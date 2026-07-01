<#
.SYNOPSIS
Runs the full prompt benchmark in isolated processor phases.

.DESCRIPTION
Runs the caller-supplied prompt against one processor backend at a time in this
order: CPU, NPU, GPU, Hybrid. Each phase starts only the target backend, starts
the harness runtime with a target-only config, records benchmark artifacts, then
stops the runtime/model backend before moving to the next phase. Hybrid is loaded
only for the final phase and is unloaded at the end.
#>
param(
    [string]$CoreModel = "qwen2.5-7b-instruct",
    [string]$CpuModel = "Qwen2.5-0.5B-Instruct-CPU",
    [string]$NpuModel = "Qwen-2.5-1.5B-Instruct-NPU",
    [string]$GpuModel = "Qwen3-0.6B-GGUF",
    [string]$HybridModel = "Qwen25Hybrid",
    [ValidateSet("auto", "fallback", "transformers", "llamacpp", "llama_cpp", "llama-cpp")]
    [string]$LocalBackend = "auto",
    [string]$GpuLlamaCppBackend = "cuda",
    [string]$GpuLlamaCppDevice = "",
    [string]$GpuModelPath = "",
    [string]$ConfigBase = ".\state\benchmarks\harness-qwen25-hybrid-full.yaml",
    [string]$ConfigActive = ".\state\benchmarks\harness-qwen25-hybrid-full-active.yaml",
    [string]$Prompt = "",
    [string]$PromptFile = "C:\Users\AllNi\.codex\attachments\b104f735-c0cd-401f-b6e5-e8146cb207e2\pasted-text.txt",
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 8080,
    [int]$MaxTokens = 2048,
    [int]$LocalMaxTokens = 64,
    [int]$BackendTimeoutSeconds = 900,
    [int]$GenerationProbeTimeoutSeconds = 300,
    [int]$GenerationProbeMaxTokens = 8,
    [int]$MultiDeviceWarmupIterations = 2,
    [int]$MultiDeviceIterations = 10,
    [int]$FullCoverageRunsPerCase = 1,
    [switch]$SkipGenerationProbe,
    [switch]$SkipModelPull,
    [switch]$KeepStack,
    [switch]$UnloadHybridAfterRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location $repoRoot
. (Join-Path $PSScriptRoot "Invoke-HarnessOneShot.ps1")

$portMap = @{ npu = 11433; gpu = 11434; cpu = 11435; hybrid = 13305 }
$modelsRoot = Join-Path $env:USERPROFILE ".ollama\models"
$modelsRootByDevice = @{
    npu = Join-Path $modelsRoot "npu"
    gpu = Join-Path $modelsRoot "gpu"
    cpu = Join-Path $modelsRoot "cpu"
}
$lemonade = Join-Path $env:LOCALAPPDATA "lemonade_server\bin\lemonade.exe"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$script:HarnessKnownLemonadeModels = @($CpuModel, $NpuModel, $GpuModel, $HybridModel) |
    Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
    Select-Object -Unique

function ConvertFrom-HarnessJson {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true, ValueFromPipeline = $true)]
        [AllowEmptyString()]
        [string]$InputObject,
        [int]$Depth = 20
    )

    process {
        $command = Get-Command ConvertFrom-Json -ErrorAction Stop
        if ($command.Parameters.ContainsKey("Depth")) {
            return ConvertFrom-Json -InputObject $InputObject -Depth $Depth -ErrorAction Stop
        }
        return ConvertFrom-Json -InputObject $InputObject -ErrorAction Stop
    }
}

function Read-HarnessPromptText {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Prompt file not found: $Path"
    }
    $source = Get-Content -LiteralPath $Path -Raw
    $match = [regex]::Match($source, "(?s)\`$promptText\s*=\s*@'\r?\n(?<prompt>.*?)\r?\n'@")
    if ($match.Success) {
        return $match.Groups["prompt"].Value
    }
    return $source
}

function Get-HarnessOptionalProperty {
    param(
        [object]$InputObject,
        [string]$Name,
        [object]$Default = $null
    )

    if ($null -eq $InputObject -or [string]::IsNullOrWhiteSpace($Name)) {
        return $Default
    }
    if ($InputObject -is [System.Collections.IDictionary]) {
        if ($InputObject.Contains($Name)) {
            return $InputObject[$Name]
        }
        return $Default
    }
    if ($InputObject.PSObject -and $InputObject.PSObject.Properties.Name -contains $Name) {
        return $InputObject.PSObject.Properties[$Name].Value
    }
    return $Default
}

function ConvertTo-HarnessInt {
    param(
        [object]$Value,
        [int]$Default = 0
    )

    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return $Default
    }
    try {
        return [int]$Value
    } catch {
        return $Default
    }
}

function ConvertTo-HarnessDouble {
    param(
        [object]$Value,
        [double]$Default = 0.0
    )

    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return $Default
    }
    try {
        return [double]$Value
    } catch {
        return $Default
    }
}

function Estimate-HarnessTokenCount {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return 0
    }
    return [int][math]::Ceiling($Text.Length / 4.0)
}

function Stop-HarnessKnownLLMProcesses {
    $knownNames = @(
        "ollama.exe",
        "ollama app.exe",
        "lemonade.exe",
        "lemonadeserver.exe",
        "llama-server.exe",
        "llamafile.exe",
        "text-generation-launcher.exe"
    )

    Get-CimInstance Win32_Process |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $knownNames -contains ([string]$_.Name).ToLowerInvariant()
        } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Stop-HarnessFresh {
    Stop-HarnessStack `
        -Config $ConfigActive `
        -All `
        -ModelBackendDevices @("npu", "gpu", "cpu", "hybrid") `
        -ModelBackendDevicePortMap $portMap `
        -ErrorAction SilentlyContinue | Out-Null

    foreach ($modelToUnload in @($script:HarnessKnownLemonadeModels)) {
        Invoke-HarnessSafeLemonadeUnload -ModelName ([string]$modelToUnload)
    }
    Stop-HarnessKnownLLMProcesses

    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "harness\.(server|local_model_provider)" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

    $ports = @([int]$ServerPort) + @($portMap.Values | ForEach-Object { [int]$_ }) | Select-Object -Unique
    foreach ($port in $ports) {
        Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique |
            Where-Object { $_ -gt 0 -and $_ -ne $PID } |
            ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    }

    Start-Sleep -Seconds 2
}

function Assert-HarnessFleetStarted {
    param(
        [object]$FleetResult,
        [string[]]$RequiredDevices
    )

    if ($null -eq $FleetResult) {
        throw "Model backend fleet returned no result."
    }
    $failedCount = if ($FleetResult.PSObject.Properties.Name -contains "failed_count") { [int]$FleetResult.failed_count } else { 0 }
    $requiredFailureCount = if ($FleetResult.PSObject.Properties.Name -contains "required_failures_count") {
        [int]$FleetResult.required_failures_count
    } elseif ($FleetResult.PSObject.Properties.Name -contains "required_failures" -and $null -ne $FleetResult.required_failures) {
        @($FleetResult.required_failures).Count
    } else {
        0
    }
    if ($failedCount -gt 0 -or $requiredFailureCount -gt 0) {
        $FleetResult | ConvertTo-Json -Depth 20 | Write-Host
        throw "One or more required local model backends failed to start."
    }

    $backendRows = if ($FleetResult.PSObject.Properties.Name -contains "backends" -and $null -ne $FleetResult.backends) {
        @($FleetResult.backends)
    } else {
        @()
    }
    foreach ($requiredDevice in $RequiredDevices) {
        $matches = @($backendRows | Where-Object { $_.device -eq $requiredDevice })
        if ($matches.Count -eq 0) {
            $FleetResult | ConvertTo-Json -Depth 20 | Write-Host
            throw "Required backend '$requiredDevice' was not returned by fleet startup."
        }
        $startedMatches = @($matches | Where-Object {
            ($_.PSObject.Properties.Name -contains "started" -and [bool]$_.started) -or
            ($_.PSObject.Properties.Name -contains "action" -and $_.action -eq "already_running")
        })
        if ($startedMatches.Count -eq 0) {
            $FleetResult | ConvertTo-Json -Depth 20 | Write-Host
            throw "Required backend '$requiredDevice' was returned but did not report started=true."
        }
    }
}

function Get-HarnessLatestFilePath {
    param(
        [string]$Directory,
        [string]$Filter,
        [datetime]$NotBefore
    )

    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        return ""
    }

    $candidate = Get-ChildItem -LiteralPath $Directory -Filter $Filter -ErrorAction SilentlyContinue |
        Where-Object { -not $_.PSIsContainer -and $_.LastWriteTime -ge $NotBefore.AddSeconds(-5) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($candidate) {
        return $candidate.FullName
    }
    return ""
}

function Resolve-HarnessLemonadePath {
    if (Test-Path -LiteralPath $script:lemonade -PathType Leaf) {
        return $script:lemonade
    }

    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $lemonadeCommand = Get-Command lemonade -ErrorAction Stop
    $script:lemonade = $lemonadeCommand.Source
    return $script:lemonade
}

function Invoke-HarnessSafeLemonadeUnload {
    param([string]$ModelName)

    try {
        $lemonadePath = Resolve-HarnessLemonadePath
    } catch {
        return
    }

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $process = Start-Process `
            -FilePath $lemonadePath `
            -ArgumentList @("unload", $ModelName) `
            -PassThru `
            -WindowStyle Hidden
        if (-not $process.WaitForExit(15000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    } catch {
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Start-HarnessLemonadeServer {
    param(
        [string]$LemonadePath,
        [int]$Port = 13305,
        [int]$WaitSeconds = 120
    )

    $healthUri = "http://127.0.0.1:$Port/api/v1/health"
    try {
        Invoke-RestMethod -Uri $healthUri -TimeoutSec 2 | Out-Null
        return [pscustomobject][ordered]@{
            started = $false
            process_id = $null
            health_url = $healthUri
            stdout_log = ""
            stderr_log = ""
        }
    } catch {
    }

    $serverPath = Join-Path (Split-Path -Path $LemonadePath -Parent) "LemonadeServer.exe"
    if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
        throw "Lemonade server executable not found: $serverPath"
    }

    $logDir = Join-Path $repoRoot "state\logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $serverStamp = Get-Date -Format "yyyyMMddTHHmmssfffZ"
    $stdoutLog = Join-Path $logDir "lemonade-server-$serverStamp.stdout.log"
    $stderrLog = Join-Path $logDir "lemonade-server-$serverStamp.stderr.log"
    $serverProcess = Start-Process `
        -FilePath $serverPath `
        -WorkingDirectory (Split-Path -Path $serverPath -Parent) `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($serverProcess.HasExited) {
            $stdoutText = if (Test-Path -LiteralPath $stdoutLog -PathType Leaf) { Get-Content -LiteralPath $stdoutLog -Raw } else { "" }
            $stderrText = if (Test-Path -LiteralPath $stderrLog -PathType Leaf) { Get-Content -LiteralPath $stderrLog -Raw } else { "" }
            throw ("Lemonade server exited before health became reachable. ExitCode={0}. Stdout={1} Stderr={2}" -f $serverProcess.ExitCode, ($stdoutText -replace "`r?`n", " ").Trim(), ($stderrText -replace "`r?`n", " ").Trim())
        }

        try {
            Invoke-RestMethod -Uri $healthUri -TimeoutSec 2 | Out-Null
            return [pscustomobject][ordered]@{
                started = $true
                process_id = $serverProcess.Id
                health_url = $healthUri
                stdout_log = $stdoutLog
                stderr_log = $stderrLog
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    throw "Timed out waiting for Lemonade server health endpoint: $healthUri"
}

function Get-HarnessLemonadeBackendStatus {
    param(
        [string]$LemonadePath,
        [string]$Recipe,
        [string]$Backend
    )

    $logDir = Join-Path $repoRoot "state\logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $statusStamp = Get-Date -Format "yyyyMMddTHHmmssfffZ"
    $stdoutLog = Join-Path $logDir "lemonade-backends-$statusStamp.stdout.log"
    $stderrLog = Join-Path $logDir "lemonade-backends-$statusStamp.stderr.log"
    $process = Start-Process `
        -FilePath $LemonadePath `
        -ArgumentList @("backends", "--all") `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru `
        -WindowStyle Hidden

    if (-not $process.WaitForExit(60000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        Write-Warning "Timed out querying Lemonade backends; will attempt backend installation."
        return ""
    }

    $stdout = if (Test-Path -LiteralPath $stdoutLog -PathType Leaf) { Get-Content -LiteralPath $stdoutLog } else { @() }
    $stderr = if (Test-Path -LiteralPath $stderrLog -PathType Leaf) { Get-Content -LiteralPath $stderrLog -Raw } else { "" }
    if ($process.ExitCode -ne 0) {
        throw "Failed to query Lemonade backends. $stderr"
    }

    foreach ($line in @($stdout)) {
        $text = [string]$line
        if ($text -match "^\s*$([regex]::Escape($Recipe))\s+$([regex]::Escape($Backend))\s+(?<status>\S+)") {
            return [string]$matches["status"]
        }
    }
    return ""
}

function Ensure-HarnessLemonadeBackend {
    param(
        [string]$LemonadePath,
        [string]$Recipe,
        [string]$Backend
    )

    if ([string]::IsNullOrWhiteSpace($Recipe) -or [string]::IsNullOrWhiteSpace($Backend)) {
        return
    }

    $status = Get-HarnessLemonadeBackendStatus -LemonadePath $LemonadePath -Recipe $Recipe -Backend $Backend
    if ($status -eq "installed") {
        return
    }

    Write-Host ("Installing Lemonade backend: {0}:{1}" -f $Recipe, $Backend)
    & $LemonadePath backends install "$Recipe`:$Backend" | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Lemonade failed to install backend '$Recipe`:$Backend' with exit code $LASTEXITCODE."
    }
}

function Start-HarnessLemonadeModel {
    param(
        [string]$ModelName,
        [string]$Device,
        [string]$LlamaCppBackend = "",
        [string]$LlamaCppDevice = ""
    )

    $lemonadePath = Resolve-HarnessLemonadePath
    Get-Service *lemonade* -ErrorAction SilentlyContinue |
        Where-Object { $_.Status -ne "Running" } |
        Start-Service -ErrorAction SilentlyContinue

    if (-not $SkipModelPull.IsPresent) {
        Write-Host ("Ensuring Lemonade model is downloaded: {0}" -f $ModelName)
        & $lemonadePath pull $ModelName | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Lemonade failed to pull '$ModelName' with exit code $LASTEXITCODE."
        }
    }

    Invoke-HarnessSafeLemonadeUnload -ModelName $ModelName
    Start-HarnessLemonadeServer -LemonadePath $lemonadePath -Port ([int]$portMap["hybrid"]) | Format-List | Out-Host

    if (-not [string]::IsNullOrWhiteSpace($LlamaCppBackend)) {
        Ensure-HarnessLemonadeBackend -LemonadePath $lemonadePath -Recipe "llamacpp" -Backend $LlamaCppBackend
    }

    $loadArgs = @("load", $ModelName, "--ctx-size", "4096", "--pinned")
    if (-not [string]::IsNullOrWhiteSpace($LlamaCppBackend)) {
        $loadArgs += @("--llamacpp", $LlamaCppBackend)
    }
    if (-not [string]::IsNullOrWhiteSpace($LlamaCppDevice)) {
        $loadArgs += @("--llamacpp-device", $LlamaCppDevice)
    }

    Write-Host ("Loading Lemonade model for {0}: {1}" -f $Device, ($loadArgs -join " "))
    & $lemonadePath @loadArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Lemonade failed to load '$ModelName' for $Device with exit code $LASTEXITCODE."
    }

    $healthUri = "http://127.0.0.1:13305/api/v1/health"
    $deadline = (Get-Date).AddSeconds([math]::Max(30, $BackendTimeoutSeconds))
    $loadedModel = ""
    while ((Get-Date) -lt $deadline) {
        try {
            $health = Invoke-RestMethod -Uri $healthUri -TimeoutSec 30
            $loadedModel = [string](Get-HarnessOptionalProperty -InputObject $health -Name "model_loaded" -Default "")
            if ($loadedModel -eq $ModelName) {
                return
            }
        } catch {
            $loadedModel = ""
        }
        Start-Sleep -Seconds 2
    }

    throw "Lemonade model is not loaded for $Device. Expected '$ModelName', got '$loadedModel'."
}

function Resolve-HarnessGpuModelPath {
    param(
        [string]$ModelName,
        [string]$ExplicitPath
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $resolved = Resolve-Path -LiteralPath $ExplicitPath -ErrorAction Stop
        return $resolved.Path
    }

    if ($ModelName -eq "Qwen3-0.6B-GGUF") {
        $root = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--unsloth--Qwen3-0.6B-GGUF\snapshots"
        $candidate = Get-ChildItem -LiteralPath $root -Recurse -File -Filter "Qwen3-0.6B-Q4_0.gguf" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }

    throw "Could not resolve a local GGUF path for GPU model '$ModelName'. Pass -GpuModelPath with a local .gguf file."
}

function Resolve-HarnessLlamaCppServerPath {
    param([string]$Backend)

    $normalized = if ([string]::IsNullOrWhiteSpace($Backend)) { "cuda" } else { $Backend.Trim().ToLowerInvariant() }
    $serverPath = Join-Path $env:USERPROFILE ".cache\lemonade\bin\llamacpp\$normalized\llama-server.exe"
    if (-not (Test-Path -LiteralPath $serverPath -PathType Leaf)) {
        throw "llama.cpp backend '$normalized' is not installed at $serverPath. Install the backend or choose another -GpuLlamaCppBackend."
    }
    return $serverPath
}

function Start-HarnessDirectLlamaCppServer {
    param(
        [string]$ModelName,
        [string]$ModelPath,
        [string]$Backend,
        [string]$Device,
        [int]$Port
    )

    $serverPath = Resolve-HarnessLlamaCppServerPath -Backend $Backend
    $resolvedModelPath = Resolve-HarnessGpuModelPath -ModelName $ModelName -ExplicitPath $ModelPath
    $healthUri = "http://127.0.0.1:$Port/health"
    $logDir = Join-Path $repoRoot "state\logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $serverStamp = Get-Date -Format "yyyyMMddTHHmmssfffZ"
    $stdoutLog = Join-Path $logDir "llamacpp-$Backend-$serverStamp.stdout.log"
    $stderrLog = Join-Path $logDir "llamacpp-$Backend-$serverStamp.stderr.log"

    $args = @(
        "-m", $resolvedModelPath,
        "--host", "127.0.0.1",
        "--port", ([string]$Port),
        "--ctx-size", "4096",
        "--alias", $ModelName,
        "--gpu-layers", "all",
        "--reasoning", "off",
        "--reasoning-format", "none"
    )
    if (-not [string]::IsNullOrWhiteSpace($Device)) {
        $args += @("--device", $Device)
    }

    Write-Host ("Starting direct llama.cpp GPU backend: {0} {1}" -f $serverPath, ($args -join " "))
    $serverProcess = Start-Process `
        -FilePath $serverPath `
        -WorkingDirectory (Split-Path -Path $serverPath -Parent) `
        -ArgumentList $args `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds($BackendTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($serverProcess.HasExited) {
            $stdoutText = if (Test-Path -LiteralPath $stdoutLog -PathType Leaf) { Get-Content -LiteralPath $stdoutLog -Raw } else { "" }
            $stderrText = if (Test-Path -LiteralPath $stderrLog -PathType Leaf) { Get-Content -LiteralPath $stderrLog -Raw } else { "" }
            throw ("llama.cpp GPU backend exited before health became reachable. ExitCode={0}. Stdout={1} Stderr={2}" -f $serverProcess.ExitCode, ($stdoutText -replace "`r?`n", " ").Trim(), ($stderrText -replace "`r?`n", " ").Trim())
        }

        try {
            Invoke-RestMethod -Uri $healthUri -TimeoutSec 5 | Out-Null
            return [pscustomobject][ordered]@{
                process_id = $serverProcess.Id
                health_url = $healthUri
                model_path = $resolvedModelPath
                stdout_log = $stdoutLog
                stderr_log = $stderrLog
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    throw "Timed out waiting for llama.cpp GPU backend health endpoint: $healthUri"
}

function New-HarnessPhaseConfigText {
    param(
        [ValidateSet("cpu", "npu", "gpu", "hybrid")]
        [string]$Device,
        [string]$ModelName,
        [int]$PhaseMaxTokens,
        [string]$Runtime = "",
        [string]$BaseUrl = "",
        [string]$HealthEndpoint = "",
        [string]$DeviceMode = ""
    )

    $port = [int]$portMap[$Device]
    $isHybrid = $Device -eq "hybrid"
    if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
        $BaseUrl = if ($isHybrid) { "http://127.0.0.1:$port/api/v1" } else { "http://127.0.0.1:$port/v1" }
    }
    if ([string]::IsNullOrWhiteSpace($HealthEndpoint)) {
        $HealthEndpoint = if ($isHybrid) { "http://127.0.0.1:$port/api/v1/health" } else { "http://127.0.0.1:$port/health" }
    }
    if ([string]::IsNullOrWhiteSpace($Runtime)) {
        $Runtime = if ($isHybrid) { "lemonade_ryzenai" } else { "local_provider" }
    }
    if ([string]::IsNullOrWhiteSpace($DeviceMode)) {
        $DeviceMode = if ($isHybrid) { "hybrid_npu_igpu" } else { $Device }
    }
    $timeout = $BackendTimeoutSeconds
    $capabilities = switch ($Device) {
        "cpu" { '"cpu", "qwen", "reasoning"' }
        "npu" { '"npu", "qwen", "reasoning"' }
        "gpu" { '"gpu", "qwen", "reasoning"' }
        "hybrid" { '"hybrid", "npu", "igpu", "gpu", "qwen", "reasoning"' }
    }

    return @(
        "backends:",
        "  - id: $Device",
        "    name: openai",
        "    base_url: `"$baseUrl`"",
        "    model: `"$ModelName`"",
        "    api_key: null",
        "    timeout_seconds: $timeout",
        "    max_tokens: $PhaseMaxTokens",
        "    required: true",
        "    capabilities: [$capabilities]",
        "    device: $Device",
        "    device_mode: $DeviceMode",
        "    runtime: $Runtime",
        "    model_family: qwen",
        "    max_context: 4096",
        "    max_output_tokens: $PhaseMaxTokens",
        "    max_concurrency: 1",
        "    health_endpoint: `"$HealthEndpoint`"",
        "",
        "model: `"$ModelName`"",
        "corpus_dir: `"corpus`"",
        "trace_dir: `"traces`"",
        "cache_dir: `".cache`"",
        "enable_cache: false",
        "max_cache_entries: 2000",
        "",
        "route_backend_defaults:",
        "  direct:",
        "    prefer_device: $Device",
        "    fallback_chain:",
        "      - $Device",
        "    latency_class: isolated",
        "    max_output_tokens: $PhaseMaxTokens",
        "  grounded_qa:",
        "    prefer_device: $Device",
        "    fallback_chain:",
        "      - $Device",
        "  structured_extraction:",
        "    prefer_device: $Device",
        "    fallback_chain:",
        "      - $Device",
        "  tool_required:",
        "    prefer_device: $Device",
        "    fallback_chain:",
        "      - $Device",
        "",
        "route_manifest_path: `"real_harness_routes.yaml`"",
        "feature_level: `"basic`"",
        "require_evidence: false",
        "agentic_parallel_enabled: false",
        "agentic_parallel_max_workers: 1",
        "agentic_parallel_max_repair_loops: 1",
        "agentic_parallel_max_wall_clock_seconds: 120"
    ) -join "`r`n"
}

function Invoke-HarnessGenerationProbe {
    param(
        [string]$Device,
        [string]$ModelName,
        [string]$ExecutionProfile
    )

    if ($SkipGenerationProbe.IsPresent) {
        return [pscustomobject][ordered]@{
            ok = $true
            skipped = $true
            status = "skipped"
            latency_ms = 0
            error = ""
        }
    }

    $probeUrl = "http://$ServerHost`:$ServerPort/v1/chat/completions"
    $payload = [ordered]@{
        model = $ModelName
        route = "direct"
        execution_profile = $ExecutionProfile
        request_id = "phase-probe-$Device-" + [guid]::NewGuid().ToString("N")
        max_tokens = [int]$GenerationProbeMaxTokens
        temperature = 0
        messages = @(
            @{ role = "user"; content = "Reply with exactly: ready" }
        )
    }

    $start = Get-Date
    try {
        $raw = Invoke-WebRequest `
            -UseBasicParsing `
            -Uri $probeUrl `
            -Method Post `
            -TimeoutSec $GenerationProbeTimeoutSeconds `
            -ContentType "application/json" `
            -Body ($payload | ConvertTo-Json -Depth 20) `
            -ErrorAction Stop
        $latencyMs = [int][math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        $response = $raw.Content | ConvertFrom-HarnessJson -Depth 20
        $status = [string](Get-HarnessOptionalProperty -InputObject $response -Name "status" -Default "")
        $execution = Get-HarnessOptionalProperty -InputObject $response -Name "execution"
        if (-not $execution) {
            $provider = Get-HarnessOptionalProperty -InputObject $response -Name "provider"
            $execution = Get-HarnessOptionalProperty -InputObject $provider -Name "execution"
        }
        $backendId = [string](Get-HarnessOptionalProperty -InputObject $execution -Name "backend_id" -Default "")
        $profile = [string](Get-HarnessOptionalProperty -InputObject $execution -Name "profile" -Default "")
        if ($raw.StatusCode -lt 200 -or $raw.StatusCode -ge 300 -or $status -ne "ok") {
            $validation = Get-HarnessOptionalProperty -InputObject $response -Name "validation"
            $errorCode = [string](Get-HarnessOptionalProperty -InputObject $response -Name "error_code" -Default "")
            $errorText = [string](Get-HarnessOptionalProperty -InputObject $response -Name "error" -Default "")
            if ([string]::IsNullOrWhiteSpace($errorText)) {
                $errorText = ($validation | ConvertTo-Json -Depth 10 -Compress)
            }
            throw "generation probe returned status='$status' http=$($raw.StatusCode) error_code='$errorCode' error='$errorText'"
        }
        return [pscustomobject][ordered]@{
            ok = $true
            skipped = $false
            status = $status
            latency_ms = $latencyMs
            backend_id = $backendId
            execution_profile = $profile
            error = ""
        }
    } catch {
        $latencyMs = [int][math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        return [pscustomobject][ordered]@{
            ok = $false
            skipped = $false
            status = "failed"
            latency_ms = $latencyMs
            backend_id = ""
            execution_profile = ""
            error = $_.Exception.Message
        }
    }
}

function Write-HarnessResponseReport {
    param(
        [object[]]$PhaseResults,
        [string]$Path
    )

    $lines = New-Object System.Collections.ArrayList
    [void]$lines.Add("# Sequential Full Prompt LLM Responses")
    [void]$lines.Add("")
    [void]$lines.Add(("Generated: {0}" -f (Get-Date).ToUniversalTime().ToString("o")))
    [void]$lines.Add("")

    foreach ($phase in @($PhaseResults)) {
        $device = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "device" -Default "unknown")
        $label = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "label" -Default $device)
        $profile = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "execution_profile" -Default "")
        $model = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "model" -Default "")
        $runtime = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "runtime" -Default "")
        $localBackend = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "local_backend" -Default "")
        $outputJson = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "output_json" -Default "")
        $phaseOk = [bool](Get-HarnessOptionalProperty -InputObject $phase -Name "ok" -Default $false)
        $phaseError = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "error" -Default "")

        [void]$lines.Add(("## {0} ({1})" -f $label, $device))
        [void]$lines.Add("")
        [void]$lines.Add(('- Model: `{0}`' -f $model))
        [void]$lines.Add(('- Runtime: `{0}`' -f $runtime))
        [void]$lines.Add(('- Backend label: `{0}`' -f $localBackend))
        [void]$lines.Add(('- Execution profile: `{0}`' -f $profile))
        [void]$lines.Add(('- Phase passed: `{0}`' -f $phaseOk))
        if (-not [string]::IsNullOrWhiteSpace($phaseError)) {
            [void]$lines.Add(('- Phase error: `{0}`' -f ($phaseError -replace "`r?`n", " ")))
        }
        if (-not [string]::IsNullOrWhiteSpace($outputJson)) {
            [void]$lines.Add(('- Source JSON: `{0}`' -f $outputJson))
        }
        [void]$lines.Add("")

        if ([string]::IsNullOrWhiteSpace($outputJson) -or -not (Test-Path -LiteralPath $outputJson -PathType Leaf)) {
            [void]$lines.Add("_No benchmark response artifact was written for this phase._")
            [void]$lines.Add("")
            continue
        }

        try {
            $benchmark = Get-Content -LiteralPath $outputJson -Raw | ConvertFrom-HarnessJson -Depth 40
            foreach ($case in @($benchmark.cases)) {
                $caseId = [string](Get-HarnessOptionalProperty -InputObject $case -Name "case_id" -Default "unknown_case")
                $status = [string](Get-HarnessOptionalProperty -InputObject $case -Name "status" -Default "")
                $selectedBackend = [string](Get-HarnessOptionalProperty -InputObject $case -Name "selected_backend" -Default "")
                $executionProfile = [string](Get-HarnessOptionalProperty -InputObject $case -Name "execution_profile" -Default "")
                $latencyMs = Get-HarnessOptionalProperty -InputObject $case -Name "latency_ms" -Default ""
                $tokensPerSecond = Get-HarnessOptionalProperty -InputObject $case -Name "tokens_per_second" -Default ""
                $promptTokens = Get-HarnessOptionalProperty -InputObject $case -Name "prompt_tokens" -Default ""
                $completionTokens = Get-HarnessOptionalProperty -InputObject $case -Name "completion_tokens" -Default ""
                $totalTokens = Get-HarnessOptionalProperty -InputObject $case -Name "total_tokens" -Default ""
                $providerBackend = [string](Get-HarnessOptionalProperty -InputObject $case -Name "provider_generation_backend" -Default "")
                $providerFallback = Get-HarnessOptionalProperty -InputObject $case -Name "provider_fallback_active" -Default ""
                $providerWarning = [string](Get-HarnessOptionalProperty -InputObject $case -Name "provider_warning" -Default "")
                $errorCode = [string](Get-HarnessOptionalProperty -InputObject $case -Name "error_code" -Default "")
                $errorMessage = [string](Get-HarnessOptionalProperty -InputObject $case -Name "error_message" -Default "")
                $responseText = [string](Get-HarnessOptionalProperty -InputObject $case -Name "response_text" -Default "")

                [void]$lines.Add(("### {0} / {1}" -f $label, $caseId))
                [void]$lines.Add("")
                [void]$lines.Add(('- Status: `{0}`' -f $status))
                [void]$lines.Add(('- Selected backend: `{0}`' -f $selectedBackend))
                [void]$lines.Add(('- Execution profile: `{0}`' -f $executionProfile))
                [void]$lines.Add(('- Latency ms: `{0}`' -f $latencyMs))
                [void]$lines.Add(('- Tokens/sec: `{0}`' -f $tokensPerSecond))
                [void]$lines.Add(('- Tokens: prompt `{0}`, completion `{1}`, total `{2}`' -f $promptTokens, $completionTokens, $totalTokens))
                [void]$lines.Add(('- Provider generation backend: `{0}`' -f $providerBackend))
                [void]$lines.Add(('- Provider fallback active: `{0}`' -f $providerFallback))
                if (-not [string]::IsNullOrWhiteSpace($providerWarning)) {
                    [void]$lines.Add(('- Provider warning: `{0}`' -f ($providerWarning -replace "`r?`n", " ")))
                }
                if (-not [string]::IsNullOrWhiteSpace($errorCode)) {
                    [void]$lines.Add(('- Error code: `{0}`' -f $errorCode))
                }
                if (-not [string]::IsNullOrWhiteSpace($errorMessage)) {
                    [void]$lines.Add(('- Error message: `{0}`' -f ($errorMessage -replace "`r?`n", " ")))
                }
                [void]$lines.Add("")
                if ([string]::IsNullOrWhiteSpace($responseText)) {
                    [void]$lines.Add("_No assistant response text was captured for this case._")
                } else {
                    [void]$lines.Add('```text')
                    [void]$lines.Add($responseText.Trim())
                    [void]$lines.Add('```')
                }
                [void]$lines.Add("")
            }
        } catch {
            [void]$lines.Add(("_Failed to read response artifact: {0}_" -f ($_.Exception.Message -replace "`r?`n", " ")))
            [void]$lines.Add("")
        }
    }

    $lines | Set-Content -LiteralPath $Path -Encoding UTF8
}

function New-HarnessProcessorReport {
    param([object[]]$PhaseResults)

    $rows = @()
    foreach ($phase in @($PhaseResults)) {
        $device = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "device" -Default "unknown")
        $label = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "label" -Default $device)
        $profile = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "execution_profile" -Default "")
        $model = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "model" -Default "")
        $runtime = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "runtime" -Default "")
        $localBackend = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "local_backend" -Default "")
        $outputJson = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "output_json" -Default "")
        $phaseOk = [bool](Get-HarnessOptionalProperty -InputObject $phase -Name "ok" -Default $false)
        $phaseError = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "error" -Default "")

        $caseCount = 0
        $succeeded = 0
        $failed = 0
        $latencyP50Ms = 0
        $speedTokensPerSecond = 0.0
        $promptTokens = 0
        $completionTokens = 0
        $totalTokens = 0
        $fallbackSamples = 0
        $providerBackends = New-Object System.Collections.Generic.List[string]

        if (-not [string]::IsNullOrWhiteSpace($outputJson) -and (Test-Path -LiteralPath $outputJson -PathType Leaf)) {
            try {
                $benchmark = Get-Content -LiteralPath $outputJson -Raw | ConvertFrom-HarnessJson -Depth 40
                $overall = Get-HarnessOptionalProperty -InputObject $benchmark -Name "overall"
                $percentiles = Get-HarnessOptionalProperty -InputObject $overall -Name "percentiles_ms"
                $caseRows = @(Get-HarnessOptionalProperty -InputObject $benchmark -Name "cases" -Default @())

                $caseCount = ConvertTo-HarnessInt (Get-HarnessOptionalProperty -InputObject $overall -Name "total" -Default $caseRows.Count) -Default $caseRows.Count
                $succeeded = ConvertTo-HarnessInt (Get-HarnessOptionalProperty -InputObject $overall -Name "succeeded" -Default 0)
                $failed = ConvertTo-HarnessInt (Get-HarnessOptionalProperty -InputObject $overall -Name "failed" -Default 0)
                $latencyP50Ms = ConvertTo-HarnessInt (Get-HarnessOptionalProperty -InputObject $percentiles -Name "p50_ms" -Default 0)
                $speedTokensPerSecond = ConvertTo-HarnessDouble (Get-HarnessOptionalProperty -InputObject $overall -Name "tokens_per_second_avg" -Default 0)
                if ($speedTokensPerSecond -le 0) {
                    $speedTokensPerSecond = ConvertTo-HarnessDouble (Get-HarnessOptionalProperty -InputObject $percentiles -Name "tokens_per_second_p50" -Default 0)
                }

                foreach ($case in $caseRows) {
                    $casePromptTokens = ConvertTo-HarnessInt (Get-HarnessOptionalProperty -InputObject $case -Name "prompt_tokens" -Default 0)
                    $caseCompletionTokens = ConvertTo-HarnessInt (Get-HarnessOptionalProperty -InputObject $case -Name "completion_tokens" -Default 0)
                    $caseTotalTokens = ConvertTo-HarnessInt (Get-HarnessOptionalProperty -InputObject $case -Name "total_tokens" -Default 0)
                    $caseFallback = [bool](Get-HarnessOptionalProperty -InputObject $case -Name "fallback_attempted" -Default $false)
                    $caseProviderFallback = [bool](Get-HarnessOptionalProperty -InputObject $case -Name "provider_fallback_active" -Default $false)
                    $caseProviderBackend = [string](Get-HarnessOptionalProperty -InputObject $case -Name "provider_generation_backend" -Default "")
                    if ($caseTotalTokens -le 0) {
                        $caseTotalTokens = $casePromptTokens + $caseCompletionTokens
                    }
                    if ($caseTotalTokens -le 0) {
                        $caseTotalTokens = Estimate-HarnessTokenCount -Text ([string](Get-HarnessOptionalProperty -InputObject $case -Name "response_text" -Default ""))
                    }

                    $promptTokens += $casePromptTokens
                    $completionTokens += $caseCompletionTokens
                    $totalTokens += $caseTotalTokens
                    if ($caseFallback -or $caseProviderFallback) {
                        $fallbackSamples++
                    }
                    if (-not [string]::IsNullOrWhiteSpace($caseProviderBackend) -and -not $providerBackends.Contains($caseProviderBackend)) {
                        $providerBackends.Add($caseProviderBackend)
                    }
                }
            } catch {
                if ([string]::IsNullOrWhiteSpace($phaseError)) {
                    $phaseError = $_.Exception.Message
                }
            }
        }

        $rows += [pscustomobject][ordered]@{
            processor = $label
            device = $device
            execution_profile = $profile
            model = $model
            runtime = $runtime
            local_backend = $localBackend
            provider_generation_backends = (@($providerBackends) -join "|")
            fallback_samples = $fallbackSamples
            status = if ($phaseOk) { "ok" } else { "failed" }
            speed_tokens_per_second = [math]::Round($speedTokensPerSecond, 3)
            total_tokens = $totalTokens
            prompt_tokens = $promptTokens
            completion_tokens = $completionTokens
            cases = $caseCount
            succeeded = $succeeded
            failed = $failed
            latency_p50_ms = $latencyP50Ms
            output_json = $outputJson
            error = $phaseError
        }
    }

    return $rows
}

function Write-HarnessTerminalResponses {
    param([object[]]$PhaseResults)

    Write-Host ""
    Write-Host "Captured LLM answers by processor"
    foreach ($phase in @($PhaseResults)) {
        $device = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "device" -Default "unknown")
        $label = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "label" -Default $device)
        $outputJson = [string](Get-HarnessOptionalProperty -InputObject $phase -Name "output_json" -Default "")

        Write-Host ""
        Write-Host ("--- {0} ({1}) ---" -f $label, $device)
        if ([string]::IsNullOrWhiteSpace($outputJson) -or -not (Test-Path -LiteralPath $outputJson -PathType Leaf)) {
            Write-Host "No captured answer artifact for this processor."
            continue
        }

        try {
            $benchmark = Get-Content -LiteralPath $outputJson -Raw | ConvertFrom-HarnessJson -Depth 40
            $responseCase = @($benchmark.cases | Where-Object {
                -not [string]::IsNullOrWhiteSpace([string](Get-HarnessOptionalProperty -InputObject $_ -Name "response_text" -Default ""))
            } | Select-Object -First 1)

            if ($responseCase.Count -eq 0) {
                Write-Host "No assistant response text was captured for this processor."
                continue
            }

            $caseId = [string](Get-HarnessOptionalProperty -InputObject $responseCase[0] -Name "case_id" -Default "unknown_case")
            $responseText = [string](Get-HarnessOptionalProperty -InputObject $responseCase[0] -Name "response_text" -Default "")
            Write-Host ("Case: {0}" -f $caseId)
            Write-Host ($responseText.Trim())
        } catch {
            Write-Host ("Could not read captured answer: {0}" -f $_.Exception.Message)
        }
    }
}

function Invoke-HarnessProcessorPhase {
    param(
        [System.Collections.IDictionary]$Phase,
        [string]$PromptText,
        [string]$OutputRoot
    )

    $device = [string]$Phase["device"]
    $label = [string]$Phase["label"]
    $profile = [string]$Phase["execution_profile"]
    $modelName = [string]$Phase["model"]
    $isHybrid = [bool](Get-HarnessOptionalProperty -InputObject $Phase -Name "hybrid" -Default $false)
    $runtime = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "runtime" -Default "")
    $backendLabel = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "backend_label" -Default $runtime)
    $baseUrl = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "base_url" -Default "")
    $healthEndpoint = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "health_endpoint" -Default "")
    $deviceMode = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "device_mode" -Default $profile)
    $usesLemonade = [bool](Get-HarnessOptionalProperty -InputObject $Phase -Name "lemonade" -Default ($runtime -like "lemonade*"))
    $usesDirectLlamaCpp = [bool](Get-HarnessOptionalProperty -InputObject $Phase -Name "direct_llamacpp" -Default $false)
    $llamaCppBackend = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "llamacpp_backend" -Default "")
    $llamaCppDevice = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "llamacpp_device" -Default "")
    $llamaCppModelPath = [string](Get-HarnessOptionalProperty -InputObject $Phase -Name "llamacpp_model_path" -Default "")
    $phaseMaxTokens = [int]$MaxTokens
    $phaseOutputDir = Join-Path $OutputRoot $device
    $phaseStart = Get-Date
    $result = [ordered]@{
        device = $device
        label = $label
        execution_profile = $profile
        model = $modelName
        runtime = $runtime
        local_backend = $backendLabel
        base_url = $baseUrl
        health_endpoint = $healthEndpoint
        device_mode = $deviceMode
        direct_llamacpp = $usesDirectLlamaCpp
        llamacpp_backend = $llamaCppBackend
        llamacpp_device = $llamaCppDevice
        llamacpp_model_path = $llamaCppModelPath
        started_at = $phaseStart.ToUniversalTime().ToString("o")
        ended_at = ""
        ok = $false
        phase_max_tokens = $phaseMaxTokens
        probe_ok = $false
        probe_latency_ms = 0
        probe_backend_id = ""
        probe_execution_profile = ""
        probe_error = ""
        output_json = ""
        output_csv = ""
        error = ""
    }

    Write-Host ""
    Write-Host ("=== Starting isolated phase: {0} ({1}) ===" -f $label, $device)

    try {
        Stop-HarnessFresh
        New-Item -ItemType Directory -Path $phaseOutputDir -Force | Out-Null
        New-HarnessPhaseConfigText `
            -Device $device `
            -ModelName $modelName `
            -PhaseMaxTokens $phaseMaxTokens `
            -Runtime $runtime `
            -BaseUrl $baseUrl `
            -HealthEndpoint $healthEndpoint `
            -DeviceMode $deviceMode |
            Set-Content -LiteralPath $ConfigActive -Encoding UTF8

        if ($usesLemonade) {
            Start-HarnessLemonadeModel `
                -ModelName $modelName `
                -Device $device `
                -LlamaCppBackend $llamaCppBackend `
                -LlamaCppDevice $llamaCppDevice
        } elseif ($usesDirectLlamaCpp) {
            Start-HarnessDirectLlamaCppServer `
                -ModelName $modelName `
                -ModelPath $llamaCppModelPath `
                -Backend $llamaCppBackend `
                -Device $llamaCppDevice `
                -Port ([int]$portMap[$device]) |
                Format-List * |
                Out-Host
        } else {
            if ($LocalBackend -eq "fallback") {
                throw "Fallback backend is disabled for the full processor benchmark."
            }
            $backendStart = Start-HarnessOwnLLMBackendFleet `
                -Devices @($device) `
                -DevicePortMap $portMap `
                -Model $CoreModel `
                -ModelsRootByDevice $modelsRootByDevice `
                -Backend $LocalBackend `
                -LocalOnly `
                -MaxNewTokens $phaseMaxTokens `
                -WaitSeconds 240

            $backendStart.backends | Format-Table device,port,started,action,status,model,runtime -AutoSize | Out-Host
            Assert-HarnessFleetStarted -FleetResult $backendStart -RequiredDevices @($device)
        }

        Start-HarnessBackend `
            -Config $ConfigActive `
            -ServerHost $ServerHost `
            -Port $ServerPort `
            -RuntimeModel $modelName `
            -WaitSeconds 180 |
            Format-List * |
            Out-Host

        if ($usesLemonade) {
            $lemonadeHealth = Invoke-RestMethod -Uri $healthEndpoint -TimeoutSec 30
            [pscustomobject]@{
                device = $device
                runtime = $runtime
                base_url = $baseUrl
                health_reachable = $true
                model_loaded = Get-HarnessOptionalProperty -InputObject $lemonadeHealth -Name "model_loaded" -Default ""
                error = ""
            } | Format-Table -AutoSize | Out-Host
        } elseif ($usesDirectLlamaCpp) {
            $llamaHealth = Invoke-RestMethod -Uri $healthEndpoint -TimeoutSec 30
            [pscustomobject]@{
                device = $device
                runtime = $runtime
                base_url = $baseUrl
                health_reachable = $true
                status = Get-HarnessOptionalProperty -InputObject $llamaHealth -Name "status" -Default ""
                error = ""
            } | Format-Table -AutoSize | Out-Host
        } else {
            $status = Get-HarnessModelBackendFleetStatus `
                -Devices @($device) `
                -DevicePortMap $portMap `
                -RequestTimeoutSeconds 30

            $status.backends | ForEach-Object {
                $backendStatus = Get-HarnessOptionalProperty -InputObject $_ -Name "status"
                [pscustomobject]@{
                    device = Get-HarnessOptionalProperty -InputObject $_ -Name "device" -Default ""
                    port = Get-HarnessOptionalProperty -InputObject $_ -Name "port" -Default $null
                    health_reachable = Get-HarnessOptionalProperty -InputObject $backendStatus -Name "health_reachable" -Default $false
                    models_reachable = Get-HarnessOptionalProperty -InputObject $backendStatus -Name "models_reachable" -Default $false
                    model_catalog_present = Get-HarnessOptionalProperty -InputObject $backendStatus -Name "model_catalog_present" -Default $false
                    error = Get-HarnessOptionalProperty -InputObject $_ -Name "error" -Default ""
                }
            } | Format-Table -AutoSize | Out-Host
        }

        $probe = Invoke-HarnessGenerationProbe -Device $device -ModelName $modelName -ExecutionProfile $profile
        $result["probe_ok"] = [bool]$probe.ok
        $result["probe_latency_ms"] = [int]$probe.latency_ms
        $result["probe_backend_id"] = [string](Get-HarnessOptionalProperty -InputObject $probe -Name "backend_id" -Default "")
        $result["probe_execution_profile"] = [string](Get-HarnessOptionalProperty -InputObject $probe -Name "execution_profile" -Default "")
        $result["probe_error"] = [string]$probe.error
        if (-not $probe.ok) {
            throw "Generation readiness probe failed for $device after $($probe.latency_ms) ms: $($probe.error)"
        }

        .\scripts\Benchmark-MultiDeviceHarness.ps1 `
            -ServerHost $ServerHost `
            -ServerPort $ServerPort `
            -Model $modelName `
            -Prompt $PromptText `
            -ExecutionProfileOverride $profile `
            -WarmupIterations $MultiDeviceWarmupIterations `
            -Iterations $MultiDeviceIterations `
            -MaxTokens $phaseMaxTokens `
            -RequestTimeoutSeconds 900 `
            -OutputDir $phaseOutputDir `
            -WriteCsv `
            -RequireBackendEvidence `
            -FailOnBenchmarkAssertion |
            Out-Host

        $result["ok"] = $true
    } catch {
        $result["error"] = $_.Exception.Message
        Write-Warning ("Phase {0} failed: {1}" -f $device, $result["error"])
    } finally {
        $result["output_json"] = Get-HarnessLatestFilePath -Directory $phaseOutputDir -Filter "multidevice-benchmark-*.json" -NotBefore $phaseStart
        $result["output_csv"] = Get-HarnessLatestFilePath -Directory $phaseOutputDir -Filter "multidevice-benchmark-*.csv" -NotBefore $phaseStart
        $result["ended_at"] = (Get-Date).ToUniversalTime().ToString("o")
        if ($usesLemonade) {
            Invoke-HarnessSafeLemonadeUnload -ModelName $modelName
        }
        Stop-HarnessFresh
        Write-Host ("=== Finished isolated phase: {0} ({1}) ===" -f $label, $device)
    }

    return [pscustomobject]$result
}

$promptSource = "file"
$promptText = if (-not [string]::IsNullOrWhiteSpace($Prompt)) {
    $promptSource = "inline"
    $Prompt
} else {
    Read-HarnessPromptText -Path $PromptFile
}
if ([string]::IsNullOrWhiteSpace($promptText)) {
    throw "Prompt text is empty."
}

if ($KeepStack.IsPresent) {
    Write-Warning "-KeepStack is accepted for compatibility but ignored by isolated sequential mode; each processor phase is stopped before the next starts."
}
if ($LocalBackend -eq "fallback") {
    throw "-LocalBackend fallback is disabled for the full processor benchmark. Use explicit real phase models instead."
}
if ($FullCoverageRunsPerCase -ne 1) {
    Write-Warning "-FullCoverageRunsPerCase is accepted for compatibility but not used in isolated sequential mode. Use -MultiDeviceIterations to control per-processor measured runs."
}
if ($UnloadHybridAfterRun.IsPresent) {
    Write-Warning "-UnloadHybridAfterRun is accepted for compatibility; isolated mode unloads every Lemonade phase model."
}

Stop-HarnessFresh

$benchmarkOutputRoot = Join-Path (Join-Path $repoRoot "state\benchmarks") "full-prompt-sequential-$stamp"
New-Item -ItemType Directory -Path $benchmarkOutputRoot -Force | Out-Null

$lemonadeBaseUrl = "http://127.0.0.1:13305/api/v1"
$lemonadeHealthEndpoint = "$lemonadeBaseUrl/health"
$phases = @(
    [ordered]@{
        device = "cpu"
        label = "CPU LLM"
        execution_profile = "cpu"
        model = $CpuModel
        hybrid = $false
        lemonade = $true
        runtime = "lemonade_ryzenai_cpu"
        backend_label = "lemonade_ryzenai_cpu"
        base_url = $lemonadeBaseUrl
        health_endpoint = $lemonadeHealthEndpoint
        device_mode = "cpu"
    },
    [ordered]@{
        device = "npu"
        label = "NPU LLM"
        execution_profile = "npu_only"
        model = $NpuModel
        hybrid = $false
        lemonade = $true
        runtime = "lemonade_ryzenai_npu"
        backend_label = "lemonade_ryzenai_npu"
        base_url = $lemonadeBaseUrl
        health_endpoint = $lemonadeHealthEndpoint
        device_mode = "npu_only"
    },
    [ordered]@{
        device = "gpu"
        label = "GPU LLM"
        execution_profile = "gpu"
        model = $GpuModel
        hybrid = $false
        lemonade = $false
        direct_llamacpp = $true
        runtime = "llamacpp_$GpuLlamaCppBackend`_direct"
        backend_label = "llamacpp_$GpuLlamaCppBackend`_direct"
        base_url = "http://127.0.0.1:$([int]$portMap["gpu"])/v1"
        health_endpoint = "http://127.0.0.1:$([int]$portMap["gpu"])/health"
        device_mode = "gpu"
        llamacpp_backend = $GpuLlamaCppBackend
        llamacpp_device = $GpuLlamaCppDevice
        llamacpp_model_path = $GpuModelPath
    },
    [ordered]@{
        device = "hybrid"
        label = "Hybrid NPU+iGPU LLM"
        execution_profile = "hybrid_npu_igpu"
        model = $HybridModel
        hybrid = $true
        lemonade = $true
        runtime = "lemonade_ryzenai_hybrid"
        backend_label = "lemonade_ryzenai_hybrid"
        base_url = $lemonadeBaseUrl
        health_endpoint = $lemonadeHealthEndpoint
        device_mode = "hybrid_npu_igpu"
    }
)

$phaseResults = @()
try {
    foreach ($phase in $phases) {
        $phaseResult = Invoke-HarnessProcessorPhase -Phase $phase -PromptText $promptText -OutputRoot $benchmarkOutputRoot
        $phaseResults += $phaseResult
    }
} finally {
    Stop-HarnessFresh
    Invoke-HarnessSafeLemonadeUnload -ModelName $HybridModel
}

$summaryJsonPath = Join-Path $benchmarkOutputRoot "sequential-full-prompt-summary-$stamp.json"
$summaryCsvPath = Join-Path $benchmarkOutputRoot "sequential-full-prompt-summary-$stamp.csv"
$processorReportCsvPath = Join-Path $benchmarkOutputRoot "sequential-full-prompt-processor-report-$stamp.csv"
$responsesMarkdownPath = Join-Path $benchmarkOutputRoot "sequential-full-prompt-responses-$stamp.md"
$failedPhases = @($phaseResults | Where-Object { -not [bool](Get-HarnessOptionalProperty -InputObject $_ -Name "ok" -Default $false) })
Write-HarnessResponseReport -PhaseResults $phaseResults -Path $responsesMarkdownPath
$processorReport = @(New-HarnessProcessorReport -PhaseResults $phaseResults)
$fallbackRows = @($processorReport | Where-Object {
    ([int](Get-HarnessOptionalProperty -InputObject $_ -Name "fallback_samples" -Default 0)) -gt 0 -or
    ([string](Get-HarnessOptionalProperty -InputObject $_ -Name "local_backend" -Default "")) -match "fallback"
})
$summary = [ordered]@{
    metadata = [ordered]@{
        run_id = "full-prompt-sequential-$stamp"
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        order = @("cpu", "npu", "gpu", "hybrid")
        prompt_source = $promptSource
        prompt_file = if ($promptSource -eq "file") { $PromptFile } else { "" }
        prompt_chars = $promptText.Length
        prompt_estimated_tokens = [int][math]::Ceiling($promptText.Length / 4.0)
        core_model = $CoreModel
        cpu_model = $CpuModel
        npu_model = $NpuModel
        gpu_model = $GpuModel
        hybrid_model = $HybridModel
        local_backend = $LocalBackend
        gpu_llamacpp_backend = $GpuLlamaCppBackend
        gpu_llamacpp_device = $GpuLlamaCppDevice
        gpu_model_path = $GpuModelPath
        skip_model_pull = $SkipModelPull.IsPresent
        warmup_iterations = $MultiDeviceWarmupIterations
        measured_iterations = $MultiDeviceIterations
        max_tokens = $MaxTokens
        local_max_tokens = $LocalMaxTokens
        backend_timeout_seconds = $BackendTimeoutSeconds
        generation_probe_timeout_seconds = $GenerationProbeTimeoutSeconds
        generation_probe_max_tokens = $GenerationProbeMaxTokens
        output_root = $benchmarkOutputRoot
        responses_markdown = $responsesMarkdownPath
        processor_report_csv = $processorReportCsvPath
    }
    processor_report = $processorReport
    phases = $phaseResults
    failed_phases = $failedPhases
    fallback_rows = $fallbackRows
}

$summary | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $summaryJsonPath -Encoding UTF8
$phaseResults | Export-Csv -Path $summaryCsvPath -NoTypeInformation -Encoding UTF8
$processorReport | Export-Csv -Path $processorReportCsvPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Sequential full prompt benchmark summary"
$phaseResults | Format-Table device,execution_profile,model,runtime,local_backend,phase_max_tokens,probe_ok,probe_latency_ms,ok,output_json,output_csv,error -AutoSize
Write-Host ""
Write-Host "Processor performance report"
$processorReport | Format-Table processor,device,model,runtime,local_backend,speed_tokens_per_second,total_tokens,cases,succeeded,status -AutoSize
Write-HarnessTerminalResponses -PhaseResults $phaseResults
Write-Host ""
Write-Host "Wrote sequential summary JSON: $summaryJsonPath"
Write-Host "Wrote sequential summary CSV:  $summaryCsvPath"
Write-Host "Wrote processor report CSV:    $processorReportCsvPath"
Write-Host "Wrote labeled LLM responses:   $responsesMarkdownPath"

if ($failedPhases.Count -gt 0) {
    $failureText = ($failedPhases | ForEach-Object { "{0}: {1}" -f $_.device, $_.error }) -join " | "
    throw "Sequential full prompt benchmark completed with failed phase(s): $failureText"
}
if ($fallbackRows.Count -gt 0) {
    $failureText = ($fallbackRows | ForEach-Object { "{0}: fallback_samples={1}, backend={2}" -f $_.device, $_.fallback_samples, $_.local_backend }) -join " | "
    throw "No-fallback assertion failed for full prompt benchmark: $failureText"
}
