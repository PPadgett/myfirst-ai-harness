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

Preflight verifies the things no code fix can create: the Lemonade hybrid
model must be present in the catalog at the hybrid base URL, and the venv's
llama-cpp-python build is probed for GPU offload support (a CPU-only wheel
silently ignores n_gpu_layers; that exact condition invalidated the "gpu"
rows of the 2026-07-02 runs).

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

    Write-Step "Preflight: llama-cpp-python GPU offload capability"
    $offload = ""
    try {
        $offload = (& $venvPython -c "import llama_cpp,sys;sys.stdout.write(str(llama_cpp.llama_supports_gpu_offload()))" 2>$null | Out-String).Trim()
    } catch { $offload = "probe_failed" }
    if ($offload -ne "True") {
        $warning = "llama-cpp-python reports gpu_offload_supported=$offload. The 'gpu' rows will run on CPU. Fix: reinstall an accelerated wheel (see docs/fix-plan-2026-07-02.md Phase B.2)."
        $preflightWarnings += $warning
        Write-Warning $warning
    } else {
        Write-Step "Preflight: GPU offload supported"
    }

    if (-not $SkipHybridCheck.IsPresent) {
        Write-Step "Preflight: Lemonade hybrid model catalog"
        try {
            $catalogRaw = Invoke-WebRequest -UseBasicParsing -Uri ($HybridBaseUrl.TrimEnd("/") + "/models") -TimeoutSec 10 -ErrorAction Stop
            $catalog = $catalogRaw.Content | ConvertFrom-Json
            $ids = @()
            if ($catalog.PSObject.Properties.Name -contains "data") { $ids = @($catalog.data | ForEach-Object { [string]$_.id }) }
            if ($ids -notcontains $HybridModel) {
                throw ("Hybrid model '" + $HybridModel + "' not in Lemonade catalog. Available: " + ($ids -join ", ") + ". Fix: lemonade-server pull " + $HybridModel)
            }
            Write-Step ("Preflight: hybrid model present (" + $HybridModel + ")")
        } catch {
            throw ("Lemonade preflight failed at " + $HybridBaseUrl + " - " + $_.Exception.Message + " (start Lemonade Server, or pass -SkipHybridCheck to proceed without the hybrid row)")
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
