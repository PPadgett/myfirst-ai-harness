<#
.SYNOPSIS
One-command end-to-end benchmark gauntlet: preflight, start the full stack,
run N seeded benchmark passes, guarantee teardown, and emit a combined
per-device scoreboard (speed + quality + determinism).

.DESCRIPTION
Closes the "manual start/run/stop" gap: previously the fleet, GPU backend,
and harnessd were started by hand (and died by hand - see the Ctrl+C
incident). This script owns the lifecycle with a try/finally so the stack is
stopped even when a step throws.

Preflight verifies AND (by default) repairs the two environment
prerequisites, since this script runs on the host where they can be fixed:
- Lemonade hybrid model absent from the catalog -> runs
  "lemonade-server pull <model>" and re-checks (registration in the catalog
  does not imply downloaded weights).
- llama-cpp-python without GPU offload support -> reinstalls the pinned
  Vulkan build (0.3.31, vendor-neutral: works on AMD and NVIDIA drivers) into
  the venv and re-probes. A CPU-only wheel silently ignores n_gpu_layers;
  that exact condition invalidated the "gpu" rows of the 2026-07-02 runs.
Pass -NoAutoFix for check-only behavior; -DryRun prints planned fixes.

.EXAMPLE
.\scripts\Invoke-HarnessBenchmarkGauntlet.ps1 -DryRun
Preflight + print every planned action without starting anything.

.EXAMPLE
.\scripts\Invoke-HarnessBenchmarkGauntlet.ps1
Full gauntlet: 2 seeded runs, MaxTokens 1024, report under benchmarks\gauntlet-<utc>.
#>
[CmdletBinding()]
param(
    [string]$Config = "harness-live-bench.yaml",
    [int]$Seed = 42,
    [ValidateRange(1, 20)]
    [int]$Runs = 2,
    # Thinking models need headroom past their <think> budget; 256 truncated
    # 100/120 rows in the first seeded runs.
    [int]$MaxTokens = 1024,
    [string]$Model = "qwen3:4b",
    [string]$ModelsRoot = "$env:USERPROFILE\.ollama\models",
    [int]$GpuLayers = 999,
    [string]$OutputRoot = "",
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 8080,
    [int]$ServerWaitSeconds = 180,
    [string]$HybridBaseUrl = "http://127.0.0.1:13305/api/v1",
    [string]$HybridModel = "Qwen2.5-3B-Instruct-Hybrid",
    [switch]$SkipPreflight,
    [switch]$SkipHybridCheck,
    [switch]$NoAutoFix,
    [switch]$KeepStackRunning,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot
. (Join-Path $scriptRoot "Invoke-HarnessOneShot.ps1")

function Write-Step { param([string]$Message) Write-Host ("[gauntlet] " + $Message) }
function Write-Plan { param([string]$Message) Write-Host ("[gauntlet:plan] " + $Message) -ForegroundColor Cyan }

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $OutputRoot = Join-Path $repoRoot ("benchmarks\gauntlet-" + $stamp)
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) { $venvPython = "python" }

# ---------------- Preflight (read-only; runs even with -DryRun) ----------------
$preflightWarnings = @()
if (-not $SkipPreflight.IsPresent) {
    Write-Step "Preflight: config"
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $Config))) {
        throw "Config not found: $Config"
    }

    function Test-GpuOffloadSupport {
        try {
            return ((& $venvPython -c "import llama_cpp,sys;sys.stdout.write(str(llama_cpp.llama_supports_gpu_offload()))" 2>$null | Out-String).Trim())
        } catch { return "probe_failed" }
    }

    Write-Step "Preflight: llama-cpp-python GPU offload capability"
    $offload = Test-GpuOffloadSupport
    if ($offload -ne "True") {
        # Direct artifact URL: an --index-url install excluded PyPI and broke
        # dependency resolution (typing-extensions), while --extra-index-url
        # would let pip pick PyPI's CPU build of the same version. The direct
        # URL pins the exact Vulkan artifact; --no-deps is safe because the
        # dependencies are already satisfied by the existing install.
        $wheelUrl = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.31-vulkan/llama_cpp_python-0.3.31-py3-none-win_amd64.whl"
        if ($NoAutoFix.IsPresent) {
            $warning = "llama-cpp-python reports gpu_offload_supported=$offload. The 'gpu' rows will run on CPU. Fix: pip install $wheelUrl --force-reinstall --no-deps --no-cache-dir"
            $preflightWarnings += $warning
            Write-Warning $warning
        } elseif ($DryRun.IsPresent) {
            Write-Plan ("AUTO-FIX would run: " + $venvPython + " -m pip install " + $wheelUrl + " --force-reinstall --no-deps --no-cache-dir")
        } else {
            Write-Step "AUTO-FIX: installing Vulkan wheel 0.3.31 (direct artifact, vendor-neutral, exact version pin)"
            & $venvPython -m pip install $wheelUrl --force-reinstall --no-deps --no-cache-dir
            if ($LASTEXITCODE -ne 0) { throw "Vulkan wheel install failed (exit $LASTEXITCODE). See docs/fix-plan-2026-07-02.md Phase B.2." }
            $offload = Test-GpuOffloadSupport
            $vulkanDll = Join-Path $repoRoot ".venv\Lib\site-packages\llama_cpp\lib\ggml-vulkan.dll"
            if ($offload -eq "True" -and (Test-Path -LiteralPath $vulkanDll)) {
                Write-Step "AUTO-FIX verified: GPU offload now supported (ggml-vulkan.dll present)"
            } else {
                $warning = "Vulkan wheel installed but offload probe reports '$offload'. Check Vulkan drivers; 'gpu' rows may still run on CPU."
                $preflightWarnings += $warning
                Write-Warning $warning
            }
        }
    } else {
        Write-Step "Preflight: GPU offload supported"
    }

    if (-not $SkipHybridCheck.IsPresent) {
        function Get-LemonadeCatalogIds {
            try {
                $catalogRaw = Invoke-WebRequest -UseBasicParsing -Uri ($HybridBaseUrl.TrimEnd("/") + "/models") -TimeoutSec 10 -ErrorAction Stop
                $catalog = $catalogRaw.Content | ConvertFrom-Json
                if ($catalog.PSObject.Properties.Name -contains "data") {
                    return @($catalog.data | ForEach-Object { [string]$_.id })
                }
                return @()
            } catch { return $null }
        }

        function Resolve-LemonadeCli {
            foreach ($name in @("lemonade-server", "lemonade-server-dev")) {
                $cmd = Get-Command $name -ErrorAction SilentlyContinue
                if ($cmd) { return $cmd.Source }
            }
            foreach ($candidate in @(
                (Join-Path $env:LOCALAPPDATA "lemonade_server\lemonade-server.exe"),
                (Join-Path $env:LOCALAPPDATA "Programs\lemonade-server\lemonade-server.exe")
            )) {
                if (Test-Path -LiteralPath $candidate) { return $candidate }
            }
            return $null
        }

        Write-Step "Preflight: Lemonade hybrid model catalog"
        $ids = Get-LemonadeCatalogIds
        if ($null -eq $ids) {
            throw ("Lemonade unreachable at " + $HybridBaseUrl + ". Start Lemonade Server (auto-starting it is intentionally out of scope), or pass -SkipHybridCheck to proceed without the hybrid row.")
        }
        if ($ids -notcontains $HybridModel) {
            $pullHint = "lemonade-server pull " + $HybridModel
            if ($NoAutoFix.IsPresent) {
                throw ("Hybrid model '" + $HybridModel + "' not in Lemonade catalog. Available: " + ($ids -join ", ") + ". Fix: " + $pullHint)
            }
            if ($DryRun.IsPresent) {
                Write-Plan ("AUTO-FIX would run: " + $pullHint + " (model weights download; can take several minutes)")
            } else {
                $cli = Resolve-LemonadeCli
                if (-not $cli) {
                    throw ("Hybrid model missing and lemonade-server CLI not found on PATH. Run manually: " + $pullHint)
                }
                Write-Step ("AUTO-FIX: pulling hybrid model via " + $cli + " (this downloads weights; be patient)")
                & $cli pull $HybridModel 2>&1 | ForEach-Object { Write-Host ("[lemonade] " + $_) }
                if ($LASTEXITCODE -ne 0) { throw ("lemonade-server pull failed (exit " + $LASTEXITCODE + "). Check Lemonade logs; see docs/fix-plan-2026-07-02.md Phase B.1 escalation paths.") }
                $ids = Get-LemonadeCatalogIds
                if ($null -eq $ids -or $ids -notcontains $HybridModel) {
                    throw ("Pull completed but '" + $HybridModel + "' still absent from the catalog. Escalate per Phase B.1 (DLL/driver signatures).")
                }
                Write-Step "AUTO-FIX verified: hybrid model now in catalog"
            }
        } else {
            Write-Step ("Preflight: hybrid model present (" + $HybridModel + ")")
        }
    }
}

# ---------------- Dry run: print the plan and exit ----------------
if ($DryRun.IsPresent) {
    Write-Plan "Start-HarnessModelBackendFleet -Devices @('npu','cpu') -Model $Model -Backend auto -ModelsRoot $ModelsRoot -LocalOnly"
    Write-Plan "Start-HarnessOwnLLMBackend -ModelBackendPort 11434 -Device gpu -Model $Model -Backend auto -ModelsRoot $ModelsRoot -LlamaCppGpuLayers $GpuLayers -LocalOnly"
    Write-Plan "Get-HarnessModelBackendFleetStatus  # require reachable = 3"
    Write-Plan "Start-HarnessBackend -ExecutionMode local -Config $Config -WaitSeconds $ServerWaitSeconds"
    for ($i = 1; $i -le $Runs; $i++) {
        $runDir = Join-Path $OutputRoot ("run-{0:d2}" -f $i)
        Write-Plan (".\scripts\Benchmark-MultiDeviceHarness.ps1 -Seed $Seed -MaxTokens $MaxTokens -WriteCsv -OutputDir $runDir -ServerHost $ServerHost -ServerPort $ServerPort")
    }
    Write-Plan ("$venvPython scripts\gauntlet_report.py <run dirs> --output-json " + (Join-Path $OutputRoot "gauntlet-report.json") + " --output-md " + (Join-Path $OutputRoot "gauntlet-report.md"))
    if (-not $KeepStackRunning.IsPresent) {
        Write-Plan "finally: Stop-HarnessBackend -ExecutionMode local -Config $Config; Stop-HarnessModelBackendFleet -Devices @('npu','gpu','cpu')"
    }
    Write-Step "Dry run complete. Re-run without -DryRun to execute."
    return
}

# ---------------- Execute with guaranteed teardown ----------------
$modelBackendsStarted = $false
$serverStarted = $false
$runDirs = @()
try {
    Write-Step "Starting model backends (npu, cpu)"
    $null = Start-HarnessModelBackendFleet -Devices @("npu", "cpu") -Model $Model -Backend auto -ModelsRoot $ModelsRoot -LocalOnly
    Write-Step "Starting GPU backend (n_gpu_layers=$GpuLayers)"
    $null = Start-HarnessOwnLLMBackend -ModelBackendPort 11434 -Device gpu -Model $Model -Backend auto -ModelsRoot $ModelsRoot -LlamaCppGpuLayers $GpuLayers -LocalOnly
    $modelBackendsStarted = $true

    $fleet = Get-HarnessModelBackendFleetStatus
    if ([int]$fleet.reachable -lt 3) {
        throw ("Model backend fleet not ready: reachable=" + $fleet.reachable + "/3")
    }
    Write-Step "Fleet ready (3/3)"

    Write-Step ("Starting harnessd (config=" + $Config + ", wait<=" + $ServerWaitSeconds + "s)")
    $null = Start-HarnessBackend -ExecutionMode local -Config $Config -WaitSeconds $ServerWaitSeconds
    $serverStarted = $true

    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    for ($i = 1; $i -le $Runs; $i++) {
        $runDir = Join-Path $OutputRoot ("run-{0:d2}" -f $i)
        Write-Step ("Benchmark run " + $i + "/" + $Runs + " -> " + $runDir)
        $null = & (Join-Path $scriptRoot "Benchmark-MultiDeviceHarness.ps1") `
            -Seed $Seed -MaxTokens $MaxTokens -WriteCsv -OutputDir $runDir `
            -ServerHost $ServerHost -ServerPort $ServerPort
        $runDirs += $runDir
    }

    Write-Step "Building combined report"
    $reportJson = Join-Path $OutputRoot "gauntlet-report.json"
    $reportMd = Join-Path $OutputRoot "gauntlet-report.md"
    & $venvPython (Join-Path $scriptRoot "gauntlet_report.py") @runDirs --output-json $reportJson --output-md $reportMd
    Write-Step ("Report: " + $reportMd)
    if ($preflightWarnings.Count -gt 0) {
        Write-Warning ("Preflight warnings applied to this run: " + ($preflightWarnings -join " | "))
    }
}
finally {
    if (-not $KeepStackRunning.IsPresent) {
        if ($serverStarted) {
            Write-Step "Teardown: harnessd"
            try { $null = Stop-HarnessBackend -ExecutionMode local -Config $Config } catch { Write-Warning ("harnessd stop failed: " + $_.Exception.Message) }
        }
        if ($modelBackendsStarted) {
            Write-Step "Teardown: model backend fleet"
            try { $null = Stop-HarnessModelBackendFleet -Devices @("npu", "gpu", "cpu") } catch { Write-Warning ("fleet stop failed: " + $_.Exception.Message) }
        }
    } else {
        Write-Step "KeepStackRunning: leaving stack up"
    }
}
