Set-StrictMode -Version Latest

$Script:HarnessRepoRoot = Split-Path -Path $PSScriptRoot -Parent
$Script:HarnessBackendStateDir = Join-Path $Script:HarnessRepoRoot "state"
$Script:HarnessBackendSessionFile = Join-Path $Script:HarnessBackendStateDir "oneshot-backend-sessions.json"
$Script:HarnessModelBackendStateDir = Join-Path $Script:HarnessRepoRoot "state"
$Script:HarnessModelBackendSessionFile = Join-Path $Script:HarnessModelBackendStateDir "oneshot-model-backend-sessions.json"
$Script:HarnessModelBackendDefaultHost = "127.0.0.1"
$Script:HarnessModelBackendDefaultPort = 11435
$Script:HarnessModelBackendDefaultModel = "local-foundation:v1"

function _Resolve-FilePath {
    param(
        [string]$Path,
        [string]$Base
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Base $Path))
}

function _Read-HarnessBackendSessions {
    $sessions = @()
    if (-not (Test-Path $Script:HarnessBackendSessionFile)) {
        return $sessions
    }
    try {
        $raw = Get-Content -Path $Script:HarnessBackendSessionFile -Raw -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $sessions
        }
        $decoded = ConvertFrom-Json -InputObject $raw -ErrorAction Stop
        if ($null -eq $decoded) {
            return $sessions
        }
        if ($decoded -is [System.Array]) {
            return @($decoded)
        }
        return @($decoded)
    } catch {
        return @()
    }
}

function _Write-HarnessBackendSessions {
    param([array]$Sessions)

    if (-not (Test-Path $Script:HarnessBackendStateDir)) {
        New-Item -ItemType Directory -Path $Script:HarnessBackendStateDir -Force | Out-Null
    }

    $payload = @($Sessions)
    if ($null -eq $payload -or $payload.Count -eq 0) {
        $payload = @()
    } else {
        $payload = @($payload)
    }

    ConvertTo-Json -InputObject $payload -Depth 10 | Set-Content -Path $Script:HarnessBackendSessionFile -Encoding UTF8
}

function _Read-HarnessModelBackendSessions {
    $sessions = @()
    if (-not (Test-Path $Script:HarnessModelBackendSessionFile)) {
        return $sessions
    }
    try {
        $raw = Get-Content -Path $Script:HarnessModelBackendSessionFile -Raw -ErrorAction Stop
        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $sessions
        }
        $decoded = ConvertFrom-Json -InputObject $raw -ErrorAction Stop
        if ($null -eq $decoded) {
            return $sessions
        }
        if ($decoded -is [System.Array]) {
            return @($decoded)
        }
        return @($decoded)
    } catch {
        return @()
    }
}

function _Write-HarnessModelBackendSessions {
    param([array]$Sessions)

    if (-not (Test-Path $Script:HarnessModelBackendStateDir)) {
        New-Item -ItemType Directory -Path $Script:HarnessModelBackendStateDir -Force | Out-Null
    }

    $payload = @($Sessions)
    if ($null -eq $payload -or $payload.Count -eq 0) {
        $payload = @()
    } else {
        $payload = @($payload)
    }

    ConvertTo-Json -InputObject $payload -Depth 10 | Set-Content -Path $Script:HarnessModelBackendSessionFile -Encoding UTF8
}

function _Prune-StaleModelSessions {
    param([array]$Sessions)

    $cleaned = @()
    $changed = $false
    foreach ($entry in $Sessions) {
        if (-not $entry.process_id) {
            $cleaned += $entry
            continue
        }
        if (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue) {
            $cleaned += $entry
            continue
        }
        $changed = $true
    }

    if ($changed) {
        _Write-HarnessModelBackendSessions -Sessions $cleaned
    }

    return $cleaned
}

function _Find-LiveModelBackendSession {
    param(
        [array]$Sessions,
        [string]$ModelBackendHost,
        [int]$ModelBackendPort
    )

    $key = "model-backend|$ModelBackendHost|$ModelBackendPort"
    foreach ($entry in $Sessions) {
        if ($entry.mode -ne "model_backend" -or $entry.key -ne $key) {
            continue
        }
        if (-not $entry.process_id) {
            continue
        }
        if (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue) {
            return $entry
        }
    }
    return $null
}

function _Format-ModelBackendListenerConflictMessage {
    param([int]$Port, [array]$Listeners)

    if ($null -eq $Listeners -or $Listeners.Count -eq 0) {
        return "Port $Port is already in use."
    }

    $first = $Listeners[0]
    $listenerPid = $first.process_id
    if ($listenerPid -le 0) {
        return "Cannot bind to port $Port because it is already in use by an external process."
    }
    $commandLine = _Get-ProcessCommandLine -ProcessId $listenerPid
    if ($commandLine) {
        return "Cannot bind to port $Port because it is already in use by process $listenerPid (command: $commandLine)."
    }
    return "Cannot bind to port $Port because it is already in use."
}

function _Resolve-ContainerProfile {
    param([string]$BackendName)

    switch -Regex ($BackendName.ToLowerInvariant()) {
        "^nvidia|^nvidia_nim|^nim$" { return "nvidia" }
        "^ollama$" { return "ollama" }
        default { return "" }
    }
}

function _Load-BackendContext {
    param([string]$ConfigPath)

    $pythonScript = @"
from harness.oneshot import runtime_backend_context
import json
import sys

print(json.dumps(runtime_backend_context(sys.argv[1]), ensure_ascii=False))
"@
    $raw = & python -c $pythonScript $ConfigPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read harness backend context from $ConfigPath"
    }
    return ($raw | ConvertFrom-Json -ErrorAction Stop)
}

function _HealthUrl {
    param(
        [string]$TargetHost,
        [int]$Port
    )

    $sanitizedHost = $TargetHost.Trim().TrimEnd("/")
    if ($sanitizedHost -match "^(https?://)") {
        return "{0}/health" -f $sanitizedHost
    }
    return "http://{0}:{1}/health" -f $sanitizedHost, $Port
}

function _Build-ModelCatalogUrl {
    param([string]$BackendBaseUrl)

    $normalized = $BackendBaseUrl.Trim().TrimEnd("/")
    if ([string]::IsNullOrWhiteSpace($normalized)) {
        return ""
    }
    if ($normalized -like "*/v1") {
        return "$normalized/models"
    }
    return "$normalized/v1/models"
}

function _Normalize-ModelsPayload {
    param([object]$CatalogPayload)

    $modelNames = @()
    if ($null -eq $CatalogPayload) {
        return $modelNames
    }

    $collection = $CatalogPayload
    if ($CatalogPayload.PSObject -and $CatalogPayload.PSObject.Properties) {
        if ($CatalogPayload.PSObject.Properties.Name -contains "data") {
            $collection = $CatalogPayload.data
        } elseif ($CatalogPayload.PSObject.Properties.Name -contains "models") {
            $collection = $CatalogPayload.models
        }
    }

    if ($null -eq $collection) {
        return $modelNames
    }

    if ($collection -isnot [System.Collections.IEnumerable] -or $collection -is [string]) {
        return $modelNames
    }

    foreach ($item in $collection) {
        if ($null -eq $item) {
            continue
        }
        if ($item -is [string]) {
            $name = $item.Trim()
            if (-not [string]::IsNullOrWhiteSpace($name)) {
                $modelNames += $name
            }
            continue
        }

        if ($item.PSObject -and $item.PSObject.Properties) {
            foreach ($candidateKey in @("id", "name", "model")) {
                if ($item.PSObject.Properties.Name -contains $candidateKey) {
                    $candidate = $item.$candidateKey
                    if ($candidate -is [string] -and (-not [string]::IsNullOrWhiteSpace($candidate))) {
                        $modelNames += $candidate.Trim()
                    }
                    break
                }
            }
        }
    }

    return $modelNames
}

function _Build-StatusPayload {
    param([object]$Body)
    if ($null -eq $Body -or [string]::IsNullOrWhiteSpace(($Body | Out-String).Trim())) {
        return $null
    }
    try {
        return ($Body | ConvertFrom-Json -ErrorAction Stop -Depth 20)
    } catch {
        return $Body
    }
}

function _Get-ExceptionResponse {
    param([object]$InvocationException)
    try {
        if ($null -eq $InvocationException) {
            return $null
        }
        if ($InvocationException.PSObject.Properties.Name -contains "Response") {
            return $InvocationException.Response
        }
        if ($InvocationException.Exception -and $InvocationException.Exception.PSObject.Properties.Name -contains "Response") {
            return $InvocationException.Exception.Response
        }
        return $null
    } catch {
        return $null
    }
}

function _Wait-HttpReady {
    param(
        [string]$HealthUrl,
        [int]$TimeoutSeconds,
        [int]$IntervalMs = 300
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $health = Invoke-WebRequest -Uri $HealthUrl -Method Get -TimeoutSec 2 -ErrorAction Stop
            if ($health.StatusCode -eq 200) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds $IntervalMs
        }
    }
    return $false
}

function _Session-Key {
    param(
        [string]$Mode,
        [string]$ConfigPath,
        [string]$ServerHost,
        [int]$Port,
        [string]$Profile
    )

    if ($Mode -eq "local") {
        return "local|$ConfigPath|$ServerHost|$Port"
    }
    return "container|$ConfigPath|$Profile|$ServerHost|$Port"
}

function _Get-ListeningProcesses {
    param([int]$Port)

    $listeners = @()
    try {
        $connections = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
        foreach ($entry in $connections) {
            if ($null -ne $entry.OwningProcess -and $entry.OwningProcess -gt 0) {
                $listeners += [pscustomobject]@{
                    process_id = [int]$entry.OwningProcess
                    local_port = [int]$entry.LocalPort
                    local_address = [string]$entry.LocalAddress
                    protocol = [string]$entry.Protocol
                }
            }
        }
    } catch {
        # Fallback to .NET listeners for hosts where Get-NetTCPConnection is unavailable.
    }

    try {
        $tcpProps = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties()
        $active = $tcpProps.GetActiveTcpListeners()
        foreach ($endpoint in $active) {
            if ([int]$endpoint.Port -ne $Port) {
                continue
            }
            $listeners += [pscustomobject]@{
                process_id = 0
                local_port = $Port
                local_address = [string]$endpoint.Address.ToString()
                protocol = "TCP"
            }
        }
    } catch {
        # fallback intentionally returns an empty list
    }

    try {
        $netstatLines = @(netstat -ano -n -p tcp 2>$null)
        foreach ($line in $netstatLines) {
            if ($line -notmatch "^\s*TCP\s+.*:\b$Port\b\s+.*LISTENING\s+\d+") {
                continue
            }
            $tokens = $line -split "\s+"
            if ($tokens.Count -lt 5) {
                continue
            }
            $pidToken = $tokens[-1]
            if ($pidToken -notmatch "^\d+$") {
                continue
            }
            $processId = [int]$pidToken
            if ($processId -le 0) {
                continue
            }

            $unknownListener = $listeners | Where-Object { $_.local_port -eq $Port -and $_.process_id -le 0 } | Select-Object -First 1
            if ($null -ne $unknownListener) {
                $unknownListener.process_id = $processId
                continue
            }
            $listeners += [pscustomobject]@{
                process_id = $processId
                local_port = $Port
                local_address = "0.0.0.0"
                protocol = "TCP"
            }
        }
    } catch {
        # fallback intentionally returns an existing list
    }

    return ,@($listeners)
}

function _Get-ProcessCommandLine {
    param([int]$ProcessId)

    try {
        $processInfo = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
        if ($null -ne $processInfo) {
            return [string]$processInfo.CommandLine
        }
    } catch {
        return ""
    }
    return ""
}

function _Prune-StaleSessions {
    param([array]$Sessions)

    $cleaned = @()
    $changed = $false
    foreach ($entry in $Sessions) {
        if ($entry.mode -eq "local" -and $entry.process_id) {
            if (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue) {
                $cleaned += $entry
                continue
            }
            $changed = $true
            continue
        }
        $cleaned += $entry
    }

    if ($changed) {
        _Write-HarnessBackendSessions -Sessions $cleaned
    }

    return $cleaned
}

function _Find-LocalSessionMatch {
    param(
        [array]$Sessions,
        [string]$ConfigPath,
        [string]$ServerHost,
        [int]$Port
    )

    $key = _Session-Key -Mode "local" -ConfigPath $ConfigPath -ServerHost $ServerHost -Port $Port -Profile ""
    foreach ($entry in $Sessions) {
        if ($entry.mode -ne "local" -or $entry.key -ne $key) {
            continue
        }
        if (-not $entry.process_id) {
            continue
        }
        if (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue) {
            return $entry
        }
    }
    return $null
}

function _Format-ListenerConflictMessage {
    param([int]$Port, [array]$Listeners)

    if ($null -eq $Listeners -or $Listeners.Count -eq 0) {
        return "Port $Port is already in use."
    }

    $first = $Listeners[0]
    $listenerPid = $first.process_id
    $commandLine = if ($listenerPid -gt 0) { _Get-ProcessCommandLine -ProcessId $listenerPid } else { "" }
    if ($listenerPid -le 0) {
        return "Cannot bind to port $Port because it is already in use by an external process with unknown PID."
    }
    if ($commandLine) {
        return "Cannot bind to port $Port because it is already in use by process $listenerPid (command: $commandLine)."
    }
    return "Cannot bind to port $Port because it is already in use."
}

function Get-HarnessBackendStatus {
    [CmdletBinding()]
    param(
        [string]$Config = "harness.yaml",
        [string]$ServerHost = "127.0.0.1",
        [int]$Port = 8080,
        [int]$RequestTimeoutSeconds = 6,
        [string]$ConfigProviderProfile = "",
        [switch]$PreferProviderOnly,
        [switch]$IncludeSession
    )

    $resolvedConfig = _Resolve-FilePath -Path $Config -Base (Get-Location).Path
    $healthUrl = _HealthUrl -TargetHost $ServerHost -Port $Port
    $context = _Load-BackendContext -ConfigPath $resolvedConfig
    $backendName = [string]$context.backend_name
    $backendModel = [string]$context.resolved_model
    $backendUrl = [string]$context.backend_url
    $backendModelsUrl = _Build-ModelCatalogUrl -BackendBaseUrl $backendUrl
    $providerStatus = [ordered]@{
        name = $backendName
        model = $backendModel
        base_url = $backendUrl
        models_url = $backendModelsUrl
        reachable = $false
        status_code = $null
        error = $null
        model_present_in_catalog = $false
        catalog_models = @()
    }

    $status = [ordered]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        config = $resolvedConfig
        server = [ordered]@{
            url = $healthUrl
            reachable = $false
            status_code = $null
            error = $null
            payload = $null
        }
        provider_plane = $providerStatus
        backend = $providerStatus
        session = [ordered]@{
            matched_sessions = @()
            mode = "local"
        }
    }

    if (-not $PreferProviderOnly) {
        try {
            $healthResponse = Invoke-WebRequest -Uri $healthUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
            $status.server.status_code = [int]$healthResponse.StatusCode
            $status.server.reachable = $healthResponse.StatusCode -ge 200 -and $healthResponse.StatusCode -lt 300
            if ($status.server.reachable) {
                $status.server.payload = _Build-StatusPayload -Body $healthResponse.Content
            } else {
                $status.server.error = "Health request returned status $($healthResponse.StatusCode)"
            }
        } catch {
            $exceptionResponse = _Get-ExceptionResponse -InvocationException $_
            if ($null -ne $exceptionResponse -and $exceptionResponse.PSObject.Properties.Name -contains "StatusCode") {
                $status.server.status_code = [int]$exceptionResponse.StatusCode
            }
            $status.server.error = "Failed to read runtime health endpoint at $healthUrl. $($_.Exception.Message)"
        }
    } else {
        $status.server.error = "Runtime health probe skipped by -PreferProviderOnly."
    }

    if (-not [string]::IsNullOrWhiteSpace($backendModelsUrl)) {
        try {
            $catalogResponse = Invoke-WebRequest -Uri $backendModelsUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
            $status.provider_plane.status_code = [int]$catalogResponse.StatusCode
            $status.provider_plane.reachable = $catalogResponse.StatusCode -ge 200 -and $catalogResponse.StatusCode -lt 300
            if ($status.provider_plane.reachable) {
                $catalogPayload = _Build-StatusPayload -Body $catalogResponse.Content
                $catalogModels = @(_Normalize-ModelsPayload -CatalogPayload $catalogPayload)
                if ($null -eq $catalogModels) {
                    $catalogModels = @()
                }
                if ($catalogModels.Count -eq 0 -and $catalogPayload -isnot [string]) {
                    # if payload parsing fails partially, keep diagnostics in model list for auditability
                    try {
                        $modelName = $catalogPayload.model -or $catalogPayload.name -or $catalogPayload.id
                        if ($modelName -is [string] -and -not [string]::IsNullOrWhiteSpace($modelName)) {
                            $catalogModels = @([string]$modelName)
                        }
                    } catch {
                        $catalogModels = @()
                    }
                }
                $status.provider_plane.catalog_models = $catalogModels
                $normalizedExpected = $backendModel.Trim().ToLowerInvariant().Replace(" ", "")
                $status.provider_plane.model_present_in_catalog = $false
                foreach ($candidate in $catalogModels) {
                    if ($candidate.Trim().ToLowerInvariant().Replace(" ", "") -eq $normalizedExpected) {
                        $status.provider_plane.model_present_in_catalog = $true
                        break
                    }
                }
            } else {
                $status.provider_plane.error = "Model catalog request returned status $($catalogResponse.StatusCode)"
            }
        } catch {
            $exceptionResponse = _Get-ExceptionResponse -InvocationException $_
            if ($null -ne $exceptionResponse -and $exceptionResponse.PSObject.Properties.Name -contains "StatusCode") {
                $status.provider_plane.status_code = [int]$exceptionResponse.StatusCode
            }
            $status.provider_plane.error = "Failed to read model catalog at $backendModelsUrl. $($_.Exception.Message)"
        }
    } else {
        $status.provider_plane.error = "Backend URL not set in configuration."
    }

    if ($IncludeSession) {
        $sessions = _Prune-StaleSessions -Sessions (_Read-HarnessBackendSessions)
        $matchedLocal = @()
        $matchedContainer = @()
        foreach ($entry in $sessions) {
            if ($entry.config -ne $resolvedConfig) {
                continue
            }
            if ($entry.mode -eq "local" -and $entry.host -eq $ServerHost -and [int]$entry.port -eq $Port) {
                if ($entry.process_id) {
                    $entry | Add-Member -NotePropertyName running -NotePropertyValue (
                        [bool](Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue)
                    ) -Force
                }
                $matchedLocal += $entry
                continue
            }
            if ($entry.mode -eq "containerized") {
                if ([string]::IsNullOrWhiteSpace($ConfigProviderProfile) -or $entry.profile -eq $ConfigProviderProfile) {
                    $matchedContainer += $entry
                }
            }
        }
        if ($matchedLocal.Count -gt 0) {
            $status.session.matched_sessions = $matchedLocal
            $status.session.mode = "local"
        } elseif ($matchedContainer.Count -gt 0) {
            $status.session.matched_sessions = $matchedContainer
            $status.session.mode = "containerized"
        }
    }

    return [PSCustomObject]$status
}

function Start-HarnessBackend {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)]
        [ValidateSet("local", "containerized")]
        [string]$ExecutionMode = "local",

        [string]$Config = "harness.yaml",
        [string]$ServerHost = "127.0.0.1",
        [int]$Port = 8080,
        [string]$ContainerProfile = "",
        [string]$ComposeFile = "docker-compose.nvidia.yaml",
        [string]$EnvFile = "",
        [switch]$NoBuild,
        [int]$WaitSeconds = 30,
        [switch]$DryRun
    )

    if ($WaitSeconds -lt 0) {
        throw "WaitSeconds must be >= 0"
    }

    $resolvedConfig = _Resolve-FilePath -Path $Config -Base (Get-Location).Path
    $context = _Load-BackendContext -ConfigPath $resolvedConfig
    $backendName = [string]$context.backend_name

    $healthUrl = _HealthUrl -TargetHost $ServerHost -Port $Port
    if ($ExecutionMode -eq "local") {
        $startArgs = @(
            "-m",
            "harness.server",
            "--config",
            $resolvedConfig,
            "--host",
            $ServerHost,
            "--port",
            $Port.ToString()
        )
        $command = "python $($startArgs -join ' ')"
        $entryKey = _Session-Key -Mode "local" -ConfigPath $resolvedConfig -ServerHost $ServerHost -Port $Port -Profile ""

        $existingSessions = _Prune-StaleSessions -Sessions (_Read-HarnessBackendSessions)
        $runningSession = _Find-LocalSessionMatch `
            -Sessions $existingSessions `
            -ConfigPath $resolvedConfig `
            -ServerHost $ServerHost `
            -Port $Port

        if ($null -ne $runningSession) {
            return [PSCustomObject]@{
                mode = "local"
                started = $false
                action = "already_running"
                config = $resolvedConfig
                host = $ServerHost
                port = $Port
                command = $command
                health_url = $healthUrl
                process_id = [int]$runningSession.process_id
                session_file = $Script:HarnessBackendSessionFile
            }
        }

        $trackedPids = @{}
        $trackedLocal = @($existingSessions | Where-Object {
                $_.mode -eq "local" -and
                $_.key -eq $entryKey -and
                $_.process_id -and $([int]$_.process_id) -gt 0
        })
        foreach ($entry in $trackedLocal) {
            $trackedPids[([int]$entry.process_id)] = $true
        }

        $listeners = _Get-ListeningProcesses -Port $Port
        $external = @(
            foreach ($listener in $listeners) {
                if ($listener.process_id -gt 0 -and $trackedPids.ContainsKey(([int]$listener.process_id))) {
                    continue
                }
                $listener
            }
        )
        if ($external.Count -gt 0) {
            throw _Format-ListenerConflictMessage -Port $Port -Listeners $external
        }
        if ($DryRun) {
            return [PSCustomObject]@{
                mode = "local"
                started = $false
                config = $resolvedConfig
                host = $ServerHost
                port = $Port
                command = $command
                health_url = $healthUrl
                session_file = $Script:HarnessBackendSessionFile
            }
        }

        $stdoutPath = [System.IO.Path]::GetTempFileName()
        $stderrPath = [System.IO.Path]::GetTempFileName()
        try {
            $process = Start-Process -FilePath "python" -ArgumentList $startArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        } catch {
            if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
            if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }
            throw "Failed to start harness server process."
        }
        if ($null -eq $process) {
            if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
            if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }
            throw "Failed to start harness server process."
        }
        if ($process.HasExited) {
            $startupError = ""
            if (Test-Path $stderrPath) {
                $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
            }
            if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
            if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }
            throw "Harness server exited immediately with code $($process.ExitCode). $startupError"
        }

        if (-not (_Wait-HttpReady -HealthUrl $healthUrl -TimeoutSeconds $WaitSeconds)) {
            $startupError = ""
            if (Test-Path $stderrPath) {
                $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
            }
            if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
            if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            throw "Harness server at $($healthUrl) did not become ready after ${WaitSeconds}s. $startupError"
        }
        if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
        if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }

        $entry = @{
            key = $entryKey
            mode = "local"
            process_id = $process.Id
            config = $resolvedConfig
            host = $ServerHost
            port = $Port
            command = $command
            health_url = $healthUrl
            started_utc = (Get-Date).ToUniversalTime().ToString("o")
        }

        $sessions = @($existingSessions | Where-Object { $_.key -ne $entry.key })
        $sessions += $entry
        _Write-HarnessBackendSessions -Sessions $sessions

        return [PSCustomObject]@{
            mode = "local"
            started = $true
            config = $resolvedConfig
            host = $ServerHost
            port = $Port
            command = $command
            health_url = $healthUrl
            process_id = $process.Id
            session_file = $Script:HarnessBackendSessionFile
        }
    }

    $profile = $ContainerProfile
    if ([string]::IsNullOrWhiteSpace($profile)) {
        $profile = _Resolve-ContainerProfile -BackendName $backendName
    }

    if ([string]::IsNullOrWhiteSpace($profile)) {
        throw "Cannot infer container profile from backend '$backendName'. Specify -ContainerProfile explicitly."
    }

    if ([string]::IsNullOrWhiteSpace($EnvFile)) {
        if ($profile -eq "nvidia") {
            $EnvFile = ".env.nvidia"
        } elseif ($profile -eq "ollama") {
            $EnvFile = ".env.ollama"
        }
    }

    $resolvedEnvFile = _Resolve-FilePath -Path $EnvFile -Base $Script:HarnessRepoRoot
    $resolvedCompose = _Resolve-FilePath -Path $ComposeFile -Base $Script:HarnessRepoRoot

    if (-not (Test-Path $resolvedCompose)) {
        throw "Compose file not found: $resolvedCompose"
    }
    if (-not (Test-Path $resolvedEnvFile)) {
        throw "Container env file not found: $resolvedEnvFile. Copy an example env file and set required values."
    }

    $composeArgs = @(
        "compose",
        "--env-file",
        $resolvedEnvFile,
        "-f",
        $resolvedCompose,
        "--profile",
        $profile,
        "up"
    )
    if (-not $NoBuild) {
        $composeArgs += "--build"
    }
    $composeArgs += "-d"

    $command = "docker $($composeArgs -join ' ')"
    if ($DryRun) {
        return [PSCustomObject]@{
            mode = "containerized"
            started = $false
            profile = $profile
            config = $resolvedConfig
            compose_file = $resolvedCompose
            env_file = $resolvedEnvFile
            command = $command
            health_url = $healthUrl
            session_file = $Script:HarnessBackendSessionFile
        }
    }

    $compose = Start-Process -FilePath "docker" -ArgumentList $composeArgs -PassThru -NoNewWindow -Wait
    if ($compose.ExitCode -ne 0) {
        throw "docker compose start failed with exit code $($compose.ExitCode)"
    }

    if (-not (_Wait-HttpReady -HealthUrl $healthUrl -TimeoutSeconds $WaitSeconds)) {
        throw "Harness backend did not become ready at $healthUrl after ${WaitSeconds}s."
    }

    $entry = @{
            key = _Session-Key -Mode "containerized" -ConfigPath $resolvedConfig -ServerHost $ServerHost -Port $Port -Profile $profile
        mode = "containerized"
        profile = $profile
        config = $resolvedConfig
        host = $ServerHost
        port = $Port
        compose_file = $resolvedCompose
        env_file = $resolvedEnvFile
        command = $command
        health_url = $healthUrl
        started_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $sessions = _Read-HarnessBackendSessions
    $sessions = @($sessions | Where-Object { $_.key -ne $entry.key })
    $sessions += $entry
    _Write-HarnessBackendSessions -Sessions $sessions

    return [PSCustomObject]@{
        mode = "containerized"
        started = $true
        profile = $profile
        config = $resolvedConfig
        compose_file = $resolvedCompose
        env_file = $resolvedEnvFile
        command = $command
        health_url = $healthUrl
        session_file = $Script:HarnessBackendSessionFile
    }
}

function Stop-HarnessBackend {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)]
        [ValidateSet("local", "containerized")]
        [string]$ExecutionMode = "local",

        [string]$Config = "harness.yaml",
        [string]$ServerHost = "127.0.0.1",
        [int]$Port = 8080,
        [string]$ContainerProfile = "",
        [string]$ComposeFile = "docker-compose.nvidia.yaml",
        [string]$EnvFile = "",
        [switch]$All,
        [switch]$DryRun
    )

    $resolvedConfig = _Resolve-FilePath -Path $Config -Base (Get-Location).Path
    $targetKey = _Session-Key -Mode $ExecutionMode -ConfigPath $resolvedConfig -ServerHost $ServerHost -Port $Port -Profile $ContainerProfile
    $rawSessions = _Read-HarnessBackendSessions
    $sessions = _Prune-StaleSessions -Sessions $rawSessions
    $matchedRaw = @(
        foreach ($entry in $rawSessions) {
            if ($entry.mode -ne "local") {
                continue
            }
            if (-not ($All -or $entry.key -eq $targetKey)) {
                continue
            }
            $entry
        }
    )
    $remaining = @()
    $removed = @()
    $removed = @(
        foreach ($entry in $matchedRaw) {
            if ($entry.process_id -and -not (Get-Process -Id $([int]$entry.process_id) -ErrorAction SilentlyContinue)) {
                $entry
            } elseif (-not ($entry.process_id)) {
                $entry
            }
        }
    )

    if ($ExecutionMode -eq "local") {
        foreach ($entry in $sessions) {
            if ($entry.mode -ne "local") {
                $remaining += $entry
                continue
            }

            $match = $All -or ($entry.key -eq $targetKey)
            if (-not $match) {
                $remaining += $entry
                continue
            }

            $removed += $entry
            if ($DryRun) {
                continue
            }
            if ($entry.process_id) {
                if (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue) {
                    Stop-Process -Id $entry.process_id -Force -ErrorAction SilentlyContinue
                }
            }
        }

        if ($DryRun) {
            return [PSCustomObject]@{
                mode = "local"
                action = "stopped"
                removed_count = $removed.Count
                removed = $removed
            }
        }

        if ($removed.Count -gt 0) {
            _Write-HarnessBackendSessions -Sessions $remaining
        }
        return [PSCustomObject]@{
            mode = "local"
            action = "stopped"
            removed_count = $removed.Count
            removed = $removed
        }
    }

    $profile = $ContainerProfile
    if ([string]::IsNullOrWhiteSpace($profile)) {
        $entryProfile = $sessions | Where-Object { $_.mode -eq "containerized" -and $_.config -eq $resolvedConfig } | Select-Object -First 1
        if ($null -ne $entryProfile -and $entryProfile.profile) {
            $profile = [string]$entryProfile.profile
        }
    }
    if ([string]::IsNullOrWhiteSpace($profile)) {
        $backendContext = _Load-BackendContext -ConfigPath $resolvedConfig
        $profile = _Resolve-ContainerProfile -BackendName [string]$backendContext.backend_name
    }
    if ([string]::IsNullOrWhiteSpace($profile)) {
        throw "Could not resolve container profile. Use -ContainerProfile explicitly."
    }

    $composeArgs = @(
        "compose",
        "--env-file",
        (if ([string]::IsNullOrWhiteSpace($EnvFile)) {
            if ($profile -eq "nvidia") { ".env.nvidia" } elseif ($profile -eq "ollama") { ".env.ollama" } else { ".env.nvidia" }
        } else { $EnvFile }),
        "-f",
        $ComposeFile,
        "--profile",
        $profile,
        "down"
    )
    $resolvedEnvFile = _Resolve-FilePath -Path $composeArgs[2] -Base $Script:HarnessRepoRoot
    if (-not (Test-Path $resolvedEnvFile)) {
        throw "Container env file not found for stop path: $resolvedEnvFile"
    }
    $resolvedCompose = _Resolve-FilePath -Path $ComposeFile -Base $Script:HarnessRepoRoot
    $composeArgs[2] = $resolvedEnvFile
    $composeArgs[4] = $resolvedCompose

    $command = "docker $($composeArgs -join ' ')"
    if ($DryRun) {
        return [PSCustomObject]@{
            mode = "containerized"
            action = "stopped"
            removed_count = $removed.Count
            command = $command
            profile = $profile
            config = $resolvedConfig
            compose_file = $resolvedCompose
            env_file = $resolvedEnvFile
        }
    }

    $remaining = @($sessions | Where-Object { $_.mode -ne "containerized" -or $_.config -ne $resolvedConfig })
    foreach ($entry in $sessions) {
        if ($entry.mode -eq "containerized" -and $entry.config -eq $resolvedConfig) {
            $removed += $entry
        }
    }

    $compose = Start-Process -FilePath "docker" -ArgumentList $composeArgs -PassThru -NoNewWindow -Wait
    if ($compose.ExitCode -ne 0) {
        throw "docker compose stop failed with exit code $($compose.ExitCode)"
    }

    if ($removed.Count -gt 0) {
        _Write-HarnessBackendSessions -Sessions $remaining
    }
    return [PSCustomObject]@{
        mode = "containerized"
        action = "stopped"
        removed_count = $removed.Count
        command = $command
        profile = $profile
        config = $resolvedConfig
    }
}

function Start-HarnessModelBackend {
    [CmdletBinding()]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [string]$Model = $Script:HarnessModelBackendDefaultModel,
        [int]$WaitSeconds = 30,
        [switch]$DryRun
    )

    if ($WaitSeconds -lt 0) {
        throw "WaitSeconds must be >= 0"
    }
    if ([string]::IsNullOrWhiteSpace($Model)) {
        $Model = $Script:HarnessModelBackendDefaultModel
    }

    $healthUrl = _HealthUrl -TargetHost $ModelBackendHost -Port $ModelBackendPort
    $entryKey = "model-backend|$ModelBackendHost|$ModelBackendPort"
    $startArgs = @(
        "-m",
        "harness.local_model_provider",
        "--host",
        $ModelBackendHost,
        "--port",
        $ModelBackendPort.ToString(),
        "--model",
        $Model
    )
    $command = "python $($startArgs -join ' ')"

    $existingSessions = _Prune-StaleModelSessions -Sessions (_Read-HarnessModelBackendSessions)
    $runningSession = _Find-LiveModelBackendSession -Sessions $existingSessions -ModelBackendHost $ModelBackendHost -ModelBackendPort $ModelBackendPort
    if ($null -ne $runningSession) {
        return [PSCustomObject]@{
            mode = "model_backend"
            started = $false
            action = "already_running"
            host = $ModelBackendHost
            port = $ModelBackendPort
            model = $Model
            command = $command
            health_url = $healthUrl
            process_id = [int]$runningSession.process_id
            session_file = $Script:HarnessModelBackendSessionFile
        }
    }

    $listeners = _Get-ListeningProcesses -Port $ModelBackendPort
    $external = @(
        foreach ($listener in $listeners) {
            if ($listener.process_id -gt 0) {
                $listener
            }
        }
    )
    if ($external.Count -gt 0) {
        throw _Format-ModelBackendListenerConflictMessage -Port $ModelBackendPort -Listeners $external
    }

    if ($DryRun) {
        return [PSCustomObject]@{
            mode = "model_backend"
            started = $false
            host = $ModelBackendHost
            port = $ModelBackendPort
            model = $Model
            command = $command
            health_url = $healthUrl
            session_file = $Script:HarnessModelBackendSessionFile
        }
    }

    $stdoutPath = [System.IO.Path]::GetTempFileName()
    $stderrPath = [System.IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath "python" -ArgumentList $startArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    } catch {
        if (Test-Path $stdoutPath) {
            Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $stderrPath) {
            Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue
        }
        throw "Failed to start local model backend process."
    }
    if ($null -eq $process) {
        if (Test-Path $stdoutPath) {
            Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $stderrPath) {
            Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue
        }
        throw "Failed to start local model backend process."
    }
    if ($process.HasExited) {
        $startupError = ""
        if (Test-Path $stderrPath) {
            $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
        }
        if (Test-Path $stdoutPath) {
            Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $stderrPath) {
            Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue
        }
        throw "Local model backend exited immediately with code $($process.ExitCode). $startupError"
    }

    if (-not (_Wait-HttpReady -HealthUrl $healthUrl -TimeoutSeconds $WaitSeconds)) {
        $startupError = ""
        if (Test-Path $stderrPath) {
            $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
        }
        if (Test-Path $stdoutPath) {
            Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $stderrPath) {
            Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "Local model backend at $($healthUrl) did not become ready after ${WaitSeconds}s. $startupError"
    }
    if (Test-Path $stdoutPath) {
        Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $stderrPath) {
        Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue
    }

    $entry = @{
        key = $entryKey
        mode = "model_backend"
        process_id = $process.Id
        host = $ModelBackendHost
        port = $ModelBackendPort
        model = $Model
        command = $command
        health_url = $healthUrl
        started_utc = (Get-Date).ToUniversalTime().ToString("o")
    }

    $sessions = @($existingSessions | Where-Object { $_.key -ne $entry.key })
    $sessions += $entry
    _Write-HarnessModelBackendSessions -Sessions $sessions

    return [PSCustomObject]@{
        mode = "model_backend"
        started = $true
        host = $ModelBackendHost
        port = $ModelBackendPort
        model = $Model
        command = $command
        health_url = $healthUrl
        process_id = $process.Id
        session_file = $Script:HarnessModelBackendSessionFile
    }
}

function Stop-HarnessModelBackend {
    [CmdletBinding()]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [switch]$All,
        [switch]$DryRun
    )

    $targetKey = "model-backend|$ModelBackendHost|$ModelBackendPort"
    $rawSessions = _Read-HarnessModelBackendSessions
    $sessions = _Prune-StaleModelSessions -Sessions $rawSessions
    $remaining = @()
    $removed = @()

    foreach ($entry in $sessions) {
        if ($entry.mode -ne "model_backend") {
            $remaining += $entry
            continue
        }

        $match = $All -or ($entry.key -eq $targetKey)
        if (-not $match) {
            $remaining += $entry
            continue
        }

        $removed += $entry
        if ($DryRun) {
            continue
        }
        if ($entry.process_id -and (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $entry.process_id -Force -ErrorAction SilentlyContinue
        }
    }

    if ($DryRun) {
        return [PSCustomObject]@{
            mode = "model_backend"
            action = "stopped"
            removed_count = $removed.Count
            removed = $removed
        }
    }

    if ($removed.Count -gt 0) {
        _Write-HarnessModelBackendSessions -Sessions $remaining
    }

    return [PSCustomObject]@{
        mode = "model_backend"
        action = "stopped"
        removed_count = $removed.Count
        removed = $removed
    }
}

function Get-HarnessModelBackendStatus {
    [CmdletBinding()]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [int]$RequestTimeoutSeconds = 6,
        [switch]$IncludeSession
    )

    $baseUrl = _HealthUrl -TargetHost $ModelBackendHost -Port $ModelBackendPort
    $modelsUrl = _Build-ModelCatalogUrl -BackendBaseUrl ($baseUrl -replace "/health$")
    $health = [ordered]@{
        url = $baseUrl
        reachable = $false
        status_code = $null
        error = $null
        payload = $null
    }
    try {
        $healthResponse = Invoke-WebRequest -Uri $baseUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
        $health.status_code = [int]$healthResponse.StatusCode
        $health.reachable = $healthResponse.StatusCode -ge 200 -and $healthResponse.StatusCode -lt 300
        if ($health.reachable) {
            $health.payload = _Build-StatusPayload -Body $healthResponse.Content
        } else {
            $health.error = "Health request returned status $($healthResponse.StatusCode)"
        }
    } catch {
        $health.error = "Failed to read model backend health at $baseUrl. $($_.Exception.Message)"
        if ($null -ne $_.Exception -and $_.Exception.PSObject.Properties.Name -contains "Response" -and $_.Exception.Response -and $_.Exception.Response.PSObject.Properties.Name -contains "StatusCode") {
            $health.status_code = [int]$_.Exception.Response.StatusCode
        }
    }

    $models = [ordered]@{
        url = $modelsUrl
        reachable = $false
        status_code = $null
        error = $null
        models = @()
        model_present = $false
    }
    try {
        $catalogResponse = Invoke-WebRequest -Uri $modelsUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
        $models.status_code = [int]$catalogResponse.StatusCode
        $models.reachable = $catalogResponse.StatusCode -ge 200 -and $catalogResponse.StatusCode -lt 300
        if ($models.reachable) {
            $catalogPayload = _Build-StatusPayload -Body $catalogResponse.Content
            $catalogModels = @(_Normalize-ModelsPayload -CatalogPayload $catalogPayload)
            if ($null -ne $catalogModels) {
                $models.models = $catalogModels
            }
            $models.model_present = $catalogModels.Count -gt 0
        } else {
            $models.error = "Model catalog request returned status $($catalogResponse.StatusCode)"
        }
    } catch {
        $models.error = "Failed to read /v1/models at $modelsUrl. $($_.Exception.Message)"
        if ($null -ne $_.Exception -and $_.Exception.PSObject.Properties.Name -contains "Response" -and $_.Exception.Response -and $_.Exception.Response.PSObject.Properties.Name -contains "StatusCode") {
            $models.status_code = [int]$_.Exception.Response.StatusCode
        }
    }

    $result = [PSCustomObject]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        host = $ModelBackendHost
        port = $ModelBackendPort
        health = $health
        models = $models
    }

    if ($IncludeSession) {
        $sessionEntries = @(
            foreach ($entry in (_Prune-StaleModelSessions -Sessions (_Read-HarnessModelBackendSessions))) {
                if ($entry.mode -eq "model_backend" -and $entry.host -eq $ModelBackendHost -and [int]$entry.port -eq $ModelBackendPort) {
                    if ($entry.process_id) {
                        $entry | Add-Member -NotePropertyName running -NotePropertyValue (
                            [bool](Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue)
                        ) -Force
                    }
                    $entry
                }
            }
        )
        $result | Add-Member -NotePropertyName session -NotePropertyValue $sessionEntries
    }

    return $result
}

function Invoke-HarnessOneShot {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)]
        [ValidateSet("runtime", "demo")]
        [string]$Mode = "runtime",

        [Parameter(Mandatory = $true, Position = 1)]
        [string]$Question,

        [string]$Config = "harness.yaml",
        [Alias("Host")]
        [string]$ServerHost = "127.0.0.1",
        [int]$Port = 8080,
        [string]$Model = "",
        [int]$StartupTimeoutSeconds = 30,
        [int]$RequestTimeoutSeconds = 120,
        [ValidateSet("basic", "hardening")]
        [string]$FeatureLevel = "",
        [ValidateSet("off", "docker")]
        [string]$ToolSandbox = "",
        [switch]$RequireEvidence,
        [switch]$EnableAdvancedRouter,
        [switch]$NoNetwork,
        [switch]$SkipBackendCheck,
        [switch]$UseExistingServer,
        [switch]$DryRun
    )

    $ErrorActionPreference = "Stop"

    function _Invoke-PythonOneshotHelper {
        param(
            [string]$QuestionText,
            [string]$ModeArg,
            [string]$ConfigPath,
            [string]$QuestionHost,
            [int]$QuestionPort,
            [string]$ExplicitModel,
            [int]$StartupTimeout,
            [int]$RequestTimeout,
            [bool]$SkipBackendCheck = $false
        )

        $env:HARNESS_ONESHOT_MODE = $ModeArg
        $env:HARNESS_ONESHOT_QUESTION = $QuestionText
        $env:HARNESS_ONESHOT_CONFIG = $ConfigPath
        $env:HARNESS_ONESHOT_HOST = $QuestionHost
        $env:HARNESS_ONESHOT_PORT = $QuestionPort.ToString()
        $env:HARNESS_ONESHOT_EXPLICIT_MODEL = $ExplicitModel
        $env:HARNESS_ONESHOT_STARTUP_TIMEOUT_SECONDS = $StartupTimeout.ToString()
        $env:HARNESS_ONESHOT_REQUEST_TIMEOUT_SECONDS = $RequestTimeout.ToString()
        if ($SkipBackendCheck) {
            $env:HARNESS_ONESHOT_SKIP_BACKEND_CHECK = "1"
        } else {
            $env:HARNESS_ONESHOT_SKIP_BACKEND_CHECK = "0"
        }

        $pythonScript = @'
import json
import os
import sys

from harness.oneshot import (
    build_chat_payload,
    build_health_url,
    runtime_backend_context,
    validate_runtime_backend,
    resolve_model,
    validate_oneshot_args,
)


mode = os.environ.get("HARNESS_ONESHOT_MODE", "runtime")
question = os.environ.get("HARNESS_ONESHOT_QUESTION", "")
config_path = os.environ.get("HARNESS_ONESHOT_CONFIG")
explicit_model = os.environ.get("HARNESS_ONESHOT_EXPLICIT_MODEL") or None
host = os.environ.get("HARNESS_ONESHOT_HOST", "127.0.0.1")
port = int(os.environ.get("HARNESS_ONESHOT_PORT", "8080"))
startup_timeout = int(os.environ.get("HARNESS_ONESHOT_STARTUP_TIMEOUT_SECONDS", "30"))
request_timeout = int(os.environ.get("HARNESS_ONESHOT_REQUEST_TIMEOUT_SECONDS", "120"))

try:
    validate_oneshot_args(
        mode=mode,
        question=question,
        host=host,
        port=port,
        startup_timeout_seconds=startup_timeout,
        request_timeout_seconds=request_timeout,
    )

    skip_backend_check = (
        os.environ.get("HARNESS_ONESHOT_SKIP_BACKEND_CHECK", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    resolved_model = resolve_model(config_path, explicit_model)
    if not skip_backend_check:
        validate_runtime_backend(
            config_path,
            timeout_seconds=max(0.5, float(request_timeout)),
            expected_model=resolved_model,
            skip_backend_check=skip_backend_check,
        )

    context = runtime_backend_context(config_path)
    payload = build_chat_payload(question, explicit_model and resolved_model or None)

    print(
        json.dumps(
            {
                "mode": mode,
                "payload": payload,
                "health_url": build_health_url(host, port),
                "resolved_model": resolved_model,
                "explicit_model": explicit_model or "",
                "backend_name": context["backend_name"],
                "backend_url": context["backend_url"],
                "config_path": context["config_path"],
            },
            ensure_ascii=False,
        )
    )
except Exception as exc:
    print(f"Runtime one-shot helper failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
'@

        $helperOutput = & python -c $pythonScript 2>&1
        $helperOutputText = $helperOutput | Out-String
        if ($LASTEXITCODE -ne 0) {
            $normalized = $helperOutputText.Trim()
            if ($normalized -match "model_backend_unavailable|model backend is unavailable|is not available in catalog|Model backend is unavailable") {
                $backendError = @"
Invoke-HarnessOneShot: model backend is unavailable at the configured /v1/models endpoint.
Model backend check failed and runtime may not be reachable.
Start the model backend, or rerun with -Mode demo or -SkipBackendCheck (intended only for controlled diagnostics).
error_code=model_backend_unavailable. $normalized
"@
                throw $backendError
            }
            throw "Invoke-HarnessOneShot: failed while resolving runtime request shape. $normalized"
        }

        return $helperOutputText | ConvertFrom-Json -Depth 20
    }

    function _Build-RuntimeEnv {
        param(
            [hashtable]$BaseEnv,
            [bool]$RequireEvidenceFlag = $false,
            [bool]$EnableAdvancedRouterFlag = $false
        )

        $runtimeEnv = @{}
        if ($BaseEnv.Count -gt 0) {
            foreach ($key in $BaseEnv.Keys) {
                $runtimeEnv[$key] = [string]$BaseEnv[$key]
            }
        }

        if ([string]::IsNullOrWhiteSpace($FeatureLevel) -eq $false) {
            $runtimeEnv["HARNESS_FEATURE_LEVEL"] = $FeatureLevel
        }
        if ([string]::IsNullOrWhiteSpace($ToolSandbox) -eq $false) {
            $runtimeEnv["HARNESS_TOOL_SANDBOX"] = $ToolSandbox
        }
        if ($RequireEvidenceFlag) {
            $runtimeEnv["HARNESS_REQUIRE_EVIDENCE"] = "1"
        }
        if ($EnableAdvancedRouterFlag) {
            $runtimeEnv["HARNESS_ENABLE_ADVANCED_ROUTER"] = "1"
        }
        return $runtimeEnv
    }

    function _Apply-ServerEnvironment {
        param([hashtable]$Environment)

        $originalEnv = @{}
        foreach ($entry in $Environment.GetEnumerator()) {
            $name = $entry.Key
            if (Test-Path "Env:$name") {
                $originalEnv[$name] = (Get-Item "Env:$name").Value
            } else {
                $originalEnv[$name] = $null
            }
            Set-Item "Env:$name" $entry.Value
        }
        return $originalEnv
    }

    function _Restore-ServerEnvironment {
        param([hashtable]$Environment)

        foreach ($entry in $Environment.GetEnumerator()) {
            $name = $entry.Key
            if ($null -eq $entry.Value) {
                Remove-Item "Env:$name" -ErrorAction SilentlyContinue
            } else {
                Set-Item "Env:$name" $entry.Value
            }
        }
    }

    function _TryReadErrorBody {
        param([object]$InvocationException)
        $errorBody = $null
        try {
            $response = _Get-ExceptionResponse -InvocationException $InvocationException
            if ($null -ne $response -and $null -ne $response.GetResponseStream) {
                $stream = $response.GetResponseStream()
                if ($null -ne $stream) {
                    $reader = New-Object System.IO.StreamReader($stream)
                    try {
                        $errorBody = $reader.ReadToEnd()
                    } finally {
                        $reader.Dispose()
                    }
                }
            }
        } catch {
            $errorBody = $null
        }
        return $errorBody
    }

    function _Wait-ForRuntimeService {
        param([string]$HealthUrl, [int]$StartupTimeoutSeconds)

        if (-not (_Wait-HttpReady -HealthUrl $HealthUrl -TimeoutSeconds $StartupTimeoutSeconds)) {
            throw "Runtime one-shot startup timed out waiting for $HealthUrl."
        }
    }

    if ($Mode -eq "runtime") {
        $helper = _Invoke-PythonOneshotHelper `
            -QuestionText $Question `
            -ModeArg $Mode `
            -ConfigPath $Config `
            -QuestionHost $ServerHost `
            -QuestionPort $Port `
            -ExplicitModel $Model `
            -StartupTimeout $StartupTimeoutSeconds `
            -RequestTimeout $RequestTimeoutSeconds `
            -SkipBackendCheck $SkipBackendCheck

        $payload = $helper.payload | ConvertTo-Json -Depth 20 -Compress
        $resolvedModel = $helper.explicit_model
        if ($resolvedModel -and $resolvedModel -ne "") {
            Write-Verbose ("Using explicit model: " + $resolvedModel)
        }

        $runtimeEnv = _Build-RuntimeEnv -BaseEnv @{} -RequireEvidenceFlag $RequireEvidence -EnableAdvancedRouterFlag $EnableAdvancedRouter
        if ($NoNetwork) {
            # keep for parity with demo path, even though the server runtime path
            # does not provide a dedicated no-network execution mode today.
        }

        if ($DryRun) {
            return [PSCustomObject]@{
                payload = $helper.payload
                resolved_model = $helper.resolved_model
                explicit_model = $helper.explicit_model
                health_url = $helper.health_url
                backend_name = $helper.backend_name
                backend_url = $helper.backend_url
                config_path = $helper.config_path
                mode = "runtime"
                runtime_env = $runtimeEnv
            }
        }

        $chatUrl = if ($ServerHost -match "^(https?://)") {
            "{0}:{1}/v1/chat/completions" -f $ServerHost.TrimEnd("/"), $Port
        } else {
            "http://{0}:{1}/v1/chat/completions" -f $ServerHost.TrimEnd("/"), $Port
        }
        $serverProcess = $null
        $envSnapshot = @{}
        $startedServer = $false
        try {
            if ($UseExistingServer) {
                if (-not (_Wait-HttpReady -HealthUrl $helper.health_url -TimeoutSeconds $StartupTimeoutSeconds)) {
                    throw "UseExistingServer was specified, but runtime service was not reachable at $($helper.health_url) within ${StartupTimeoutSeconds}s."
                }
                _Wait-ForRuntimeService -HealthUrl $helper.health_url -StartupTimeoutSeconds $StartupTimeoutSeconds
            } else {
                if ($runtimeEnv.Count -gt 0) {
                    $envSnapshot = _Apply-ServerEnvironment -Environment $runtimeEnv
                }

                $serverProcess = Start-Process -FilePath "python" -ArgumentList @(
                    "-m",
                    "harness.server",
                    "--config",
                    $Config,
                    "--host",
                    $ServerHost,
                    "--port",
                    $Port.ToString()
                ) -PassThru -NoNewWindow

                if (-not $serverProcess) {
                    throw "Failed to start harness server process."
                }
                $startedServer = $true
                _Wait-ForRuntimeService -HealthUrl $helper.health_url -StartupTimeoutSeconds $StartupTimeoutSeconds
            }

            try {
                $response = Invoke-WebRequest -Uri $chatUrl -Method Post -ContentType "application/json" -Body $payload -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
            } catch {
                $errorResponse = _Get-ExceptionResponse -InvocationException $_
                $status = $null
                if ($null -ne $errorResponse) {
                    $status = $errorResponse.StatusCode
                }
                $errorBody = _TryReadErrorBody -InvocationException $_.Exception
                if ($null -ne $status -and $null -ne $errorResponse) {
                    throw "Runtime one-shot request failed: HTTP $status. $errorBody"
                }
                throw "Runtime one-shot request failed: $($_.Exception.Message). $errorBody"
            }

            if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) {
                throw "Runtime one-shot request returned HTTP $($response.StatusCode): $($response.Content)"
            }
            return $response.Content | ConvertFrom-Json -Depth 20
        } finally {
            if ($UseExistingServer -eq $false -and $startedServer -and $serverProcess -ne $null -and -not $serverProcess.HasExited) {
                Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
            }
            if ($runtimeEnv.Count -gt 0 -and -not $UseExistingServer) {
                _Restore-ServerEnvironment -Environment $envSnapshot
            }
        }
    }

    if ($Mode -eq "demo") {
        $demoArgs = @("real_ai_harness.py", "--query", $Question)
        if ($NoNetwork) {
            $demoArgs += "--no-network"
        }
        return & python @demoArgs
    }

    throw "Unsupported mode: $Mode"
}
