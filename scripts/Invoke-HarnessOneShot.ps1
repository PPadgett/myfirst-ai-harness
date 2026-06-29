Set-StrictMode -Version Latest

$Script:HarnessRepoRoot = Split-Path -Path $PSScriptRoot -Parent
$Script:HarnessBackendStateDir = Join-Path $Script:HarnessRepoRoot "state"
$Script:HarnessBackendSessionFile = Join-Path $Script:HarnessBackendStateDir "oneshot-backend-sessions.json"
$Script:HarnessModelBackendStateDir = Join-Path $Script:HarnessRepoRoot "state"
$Script:HarnessModelBackendSessionFile = Join-Path $Script:HarnessModelBackendStateDir "oneshot-model-backend-sessions.json"
$Script:HarnessLogDir = Join-Path $Script:HarnessRepoRoot "state\logs"
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

function _Resolve-HarnessPythonPath {
    param([string]$PythonPath = "")

    if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
        if ([System.IO.Path]::IsPathRooted($PythonPath) -or $PythonPath.Contains("\") -or $PythonPath.Contains("/")) {
            return _Resolve-FilePath -Path $PythonPath -Base (Get-Location).Path
        }
        return $PythonPath
    }

    $venvPython = Join-Path $Script:HarnessRepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        return $venvPython
    }
    return "python"
}

function _Format-HarnessCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $parts = @($Executable) + @($Arguments)
    return ($parts | ForEach-Object {
            $part = [string]$_
            if ($part -match '\s') {
                '"' + $part.Replace('"', '\"') + '"'
            } else {
                $part
            }
        }) -join " "
}

function _New-HarnessLogPaths {
    param([string]$Prefix)

    if (-not (Test-Path -LiteralPath $Script:HarnessLogDir -PathType Container)) {
        New-Item -ItemType Directory -Path $Script:HarnessLogDir -Force | Out-Null
    }
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
    $id = [System.Guid]::NewGuid().ToString("N").Substring(0, 8)
    $safePrefix = ([string]$Prefix) -replace '[^a-zA-Z0-9_.-]', '-'
    return [PSCustomObject]@{
        stdout_log = Join-Path $Script:HarnessLogDir "$safePrefix-$stamp-$id.stdout.log"
        stderr_log = Join-Path $Script:HarnessLogDir "$safePrefix-$stamp-$id.stderr.log"
    }
}

function _Join-HarnessPathParts {
    param(
        [string]$Root,
        [string[]]$Parts
    )

    $current = $Root
    foreach ($part in $Parts) {
        if ([string]::IsNullOrWhiteSpace($part)) {
            continue
        }
        $current = Join-Path $current $part
    }
    return $current
}

function _Get-HarnessCandidateModelRoots {
    param([string]$ModelsRoot)

    $roots = @()
    if (-not [string]::IsNullOrWhiteSpace($ModelsRoot)) {
        $roots += _Resolve-FilePath -Path $ModelsRoot -Base (Get-Location).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($env:OLLAMA_MODELS)) {
        $roots += _Resolve-FilePath -Path $env:OLLAMA_MODELS -Base (Get-Location).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        $roots += (Join-Path $env:USERPROFILE ".ollama\models")
    }

    $unique = @()
    $seen = @{}
    foreach ($root in $roots) {
        if ([string]::IsNullOrWhiteSpace($root)) {
            continue
        }
        $key = [string]$root
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true
        $unique += $root
    }
    return $unique
}

function _Get-HarnessCandidateHuggingFaceCacheRoots {
    $roots = @()
    if (-not [string]::IsNullOrWhiteSpace($env:HF_HUB_CACHE)) {
        $roots += _Resolve-FilePath -Path $env:HF_HUB_CACHE -Base (Get-Location).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($env:HUGGINGFACE_HUB_CACHE)) {
        $roots += _Resolve-FilePath -Path $env:HUGGINGFACE_HUB_CACHE -Base (Get-Location).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($env:HF_HOME)) {
        $roots += (Join-Path (_Resolve-FilePath -Path $env:HF_HOME -Base (Get-Location).Path) "hub")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
        $roots += (Join-Path $env:USERPROFILE ".cache\huggingface\hub")
    }

    $unique = @()
    $seen = @{}
    foreach ($root in $roots) {
        if ([string]::IsNullOrWhiteSpace($root)) {
            continue
        }
        $key = [string]$root
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true
        $unique += $root
    }
    return $unique
}

function _Resolve-HarnessHuggingFaceCacheDir {
    param([string]$Model)

    $modelId = $Model.Trim()
    if ($modelId -match "(?i)^hf://") {
        $modelId = $modelId.Substring(5)
    }
    if ([string]::IsNullOrWhiteSpace($modelId) -or -not $modelId.Contains("/")) {
        return $null
    }
    $cacheName = "models--" + $modelId.Replace("/", "--")
    foreach ($root in (_Get-HarnessCandidateHuggingFaceCacheRoots)) {
        $candidate = Join-Path $root $cacheName
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            return $candidate
        }
    }
    return $null
}

function _Test-HarnessHuggingFaceCacheHasWeights {
    param([string]$CacheDir)

    if ([string]::IsNullOrWhiteSpace($CacheDir) -or -not (Test-Path -LiteralPath $CacheDir -PathType Container)) {
        return $false
    }
    $snapshots = Join-Path $CacheDir "snapshots"
    $searchRoot = if (Test-Path -LiteralPath $snapshots -PathType Container) { $snapshots } else { $CacheDir }
    foreach ($pattern in @("*.safetensors", "*.bin", "*.pt", "*.pth")) {
        $found = Get-ChildItem -LiteralPath $searchRoot -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -ne $found) {
            return $true
        }
    }
    return $false
}

function _Get-HarnessOllamaModelParts {
    param([string]$Model)

    $name = $Model.Trim()
    $tag = "latest"
    $namePart = $name
    $lastColon = $name.LastIndexOf(":")
    if ($lastColon -gt 0) {
        $namePart = $name.Substring(0, $lastColon)
        $tag = $name.Substring($lastColon + 1)
        if ([string]::IsNullOrWhiteSpace($tag)) {
            $tag = "latest"
        }
    }

    $parts = @($namePart.Trim("/") -split "/" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($parts.Count -eq 1) {
        $parts = @("registry.ollama.ai", "library", $parts[0])
    } elseif ($parts.Count -eq 2) {
        $parts = @("registry.ollama.ai", $parts[0], $parts[1])
    }

    return [PSCustomObject]@{
        parts = $parts
        tag = $tag
    }
}

function _Resolve-HarnessOllamaStoreSource {
    param(
        [string]$Model,
        [string]$ModelsRoot
    )

    $empty = [ordered]@{
        resolved = $false
        source = $null
        source_type = $null
        artifact_format = $null
        provider_store = $null
        manifest_path = $null
        generation_backend = $null
    }
    $modelParts = _Get-HarnessOllamaModelParts -Model $Model
    foreach ($root in (_Get-HarnessCandidateModelRoots -ModelsRoot $ModelsRoot)) {
        $manifest = _Join-HarnessPathParts -Root (Join-Path $root "manifests") -Parts (@($modelParts.parts) + @($modelParts.tag))
        if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
            continue
        }
        try {
            $payload = Get-Content -LiteralPath $manifest -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        } catch {
            continue
        }

        $layers = @()
        if ($payload.PSObject.Properties.Name -contains "layers") {
            $layers = @($payload.layers)
        }
        $modelLayers = @($layers | Where-Object {
            $_ -and $_.PSObject.Properties.Name -contains "mediaType" -and ([string]$_.mediaType).ToLowerInvariant().Contains("model")
        })
        if ($modelLayers.Count -eq 0) {
            $modelLayers = @($layers)
        }

        foreach ($layer in $modelLayers) {
            if ($null -eq $layer -or -not ($layer.PSObject.Properties.Name -contains "digest")) {
                continue
            }
            $digest = ([string]$layer.digest).Trim()
            if (-not $digest.StartsWith("sha256:")) {
                continue
            }
            $blob = Join-Path (Join-Path $root "blobs") ($digest.Replace(":", "-"))
            if (-not (Test-Path -LiteralPath $blob -PathType Leaf)) {
                continue
            }
            return [ordered]@{
                resolved = $true
                source = $blob
                source_type = "ollama_store"
                artifact_format = "gguf"
                provider_store = "ollama"
                manifest_path = $manifest
                generation_backend = "llamacpp"
            }
        }
    }
    return $empty
}

function _Resolve-HarnessFilesystemModelSource {
    param(
        [string]$Model,
        [string]$ModelsRoot
    )

    $empty = [ordered]@{
        resolved = $false
        source = $null
        source_type = $null
        artifact_format = $null
        provider_store = $null
        manifest_path = $null
        generation_backend = $null
    }
    if ([string]::IsNullOrWhiteSpace($ModelsRoot)) {
        return $empty
    }
    $root = _Resolve-FilePath -Path $ModelsRoot -Base (Get-Location).Path
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        return $empty
    }

    $candidates = @()
    $candidates += (Join-Path $root $Model)
    $candidates += (Join-Path $root ($Model.Replace(":", "_").Replace("/", [System.IO.Path]::DirectorySeparatorChar)))
    $lastColon = $Model.LastIndexOf(":")
    if ($lastColon -gt 0) {
        $name = $Model.Substring(0, $lastColon)
        $tag = $Model.Substring($lastColon + 1)
        $candidates += (Join-Path (Join-Path $root $name) $tag)
        $candidates += (Join-Path $root $name)
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $artifact = if ($candidate.ToLowerInvariant().EndsWith(".gguf")) { "gguf" } else { "transformers" }
            return [ordered]@{
                resolved = $true
                source = $candidate
                source_type = "models_root"
                artifact_format = $artifact
                provider_store = $null
                manifest_path = $null
                generation_backend = if ($artifact -eq "gguf") { "llamacpp" } else { "transformers" }
            }
        }
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            $config = Join-Path $candidate "config.json"
            if (Test-Path -LiteralPath $config -PathType Leaf) {
                return [ordered]@{
                    resolved = $true
                    source = $candidate
                    source_type = "models_root"
                    artifact_format = "transformers"
                    provider_store = $null
                    manifest_path = $null
                    generation_backend = "transformers"
                }
            }
            $gguf = @(Get-ChildItem -LiteralPath $candidate -Filter "*.gguf" -File -ErrorAction SilentlyContinue)
            if ($gguf.Count -eq 1) {
                return [ordered]@{
                    resolved = $true
                    source = $gguf[0].FullName
                    source_type = "models_root"
                    artifact_format = "gguf"
                    provider_store = $null
                    manifest_path = $null
                    generation_backend = "llamacpp"
                }
            }
        }
    }
    return $empty
}

function _Resolve-HarnessModelBackendSource {
    param(
        [string]$Model,
        [string]$ModelPath,
        [string]$ModelsRoot
    )

    $empty = [ordered]@{
        resolved = $false
        source = $null
        source_type = $null
        artifact_format = $null
        provider_store = $null
        manifest_path = $null
        generation_backend = $null
    }

    if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
        $resolvedPath = _Resolve-FilePath -Path $ModelPath -Base (Get-Location).Path
        $artifact = if ($resolvedPath.ToLowerInvariant().EndsWith(".gguf")) { "gguf" } else { "transformers" }
        return [ordered]@{
            resolved = [bool](Test-Path -LiteralPath $resolvedPath)
            source = $resolvedPath
            source_type = "filesystem"
            artifact_format = $artifact
            provider_store = $null
            manifest_path = $null
            generation_backend = if ($artifact -eq "gguf") { "llamacpp" } else { "transformers" }
        }
    }

    if ($Model -match "(?i)^hf://") {
        $cacheDir = _Resolve-HarnessHuggingFaceCacheDir -Model $Model
        return [ordered]@{
            resolved = $true
            source = $Model.Substring(5)
            source_type = "huggingface_cache"
            artifact_format = "transformers"
            provider_store = "huggingface"
            manifest_path = $cacheDir
            generation_backend = "transformers"
        }
    }

    $filesystem = _Resolve-HarnessFilesystemModelSource -Model $Model -ModelsRoot $ModelsRoot
    if ([bool]$filesystem.resolved) {
        return $filesystem
    }

    $ollama = _Resolve-HarnessOllamaStoreSource -Model $Model -ModelsRoot $ModelsRoot
    if ([bool]$ollama.resolved) {
        return $ollama
    }

    return $empty
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

function _Get-HarnessNamedProperty {
    param(
        [object]$InputObject,
        [string[]]$Names
    )

    if ($null -eq $InputObject -or $null -eq $Names) {
        return $null
    }
    foreach ($name in $Names) {
        if ([string]::IsNullOrWhiteSpace($name)) {
            continue
        }
        if ($InputObject -is [System.Collections.IDictionary] -and $InputObject.Contains($name)) {
            return $InputObject[$name]
        }
        if ($InputObject.PSObject -and $InputObject.PSObject.Properties.Name -contains $name) {
            return $InputObject.PSObject.Properties[$name].Value
        }
    }
    return $null
}

function _Convert-HarnessBoolean {
    param([object]$Value)

    if ($null -eq $Value) {
        return $false
    }
    if ($Value -is [bool]) {
        return [bool]$Value
    }
    $text = ([string]$Value).Trim().ToLowerInvariant()
    return $text -in @("1", "true", "yes", "on")
}

function _Get-ModelCatalogEntries {
    param([object]$CatalogPayload)

    if ($null -eq $CatalogPayload) {
        return @()
    }

    $collection = $CatalogPayload
    if ($CatalogPayload.PSObject -and $CatalogPayload.PSObject.Properties.Name -contains "data") {
        $collection = $CatalogPayload.data
    } elseif ($CatalogPayload.PSObject -and $CatalogPayload.PSObject.Properties.Name -contains "models") {
        $collection = $CatalogPayload.models
    }

    if ($null -eq $collection -or $collection -is [string] -or $collection -isnot [System.Collections.IEnumerable]) {
        return @()
    }
    return @($collection)
}

function _Get-ModelCatalogEntry {
    param(
        [object]$CatalogPayload,
        [string]$ModelName = ""
    )

    $entries = @(_Get-ModelCatalogEntries -CatalogPayload $CatalogPayload)
    if ($entries.Count -eq 0) {
        return $null
    }
    if (-not [string]::IsNullOrWhiteSpace($ModelName)) {
        $expected = $ModelName.Trim().ToLowerInvariant().Replace(" ", "")
        foreach ($entry in $entries) {
            $name = _Get-HarnessNamedProperty -InputObject $entry -Names @("id", "name", "model")
            if ($null -ne $name -and ([string]$name).Trim().ToLowerInvariant().Replace(" ", "") -eq $expected) {
                return $entry
            }
        }
    }
    return $entries[0]
}

function _Get-ProviderDiagnosticsFromPayloads {
    param(
        [object]$HealthPayload,
        [object]$ModelEntry
    )

    $configuredBackend = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("configured_backend", "backend")
    if ($null -eq $configuredBackend) {
        $configuredBackend = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("configured_backend", "backend")
    }
    $generationBackend = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("generation_backend")
    if ($null -eq $generationBackend) {
        $generationBackend = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("generation_backend")
    }
    $modelSource = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("model_source", "source")
    if ($null -eq $modelSource) {
        $modelSource = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("model_source", "source")
    }
    $modelSourceType = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("model_source_type", "source_type")
    if ($null -eq $modelSourceType) {
        $modelSourceType = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("model_source_type", "source_type")
    }
    $modelArtifactFormat = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("model_artifact_format", "artifact_format")
    if ($null -eq $modelArtifactFormat) {
        $modelArtifactFormat = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("model_artifact_format", "artifact_format")
    }
    $providerStore = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("provider_store")
    if ($null -eq $providerStore) {
        $providerStore = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("provider_store")
    }
    $manifestPath = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("manifest_path")
    if ($null -eq $manifestPath) {
        $manifestPath = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("manifest_path")
    }
    $runtimeDependency = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("runtime_dependency")
    if ($null -eq $runtimeDependency) {
        $runtimeDependency = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("runtime_dependency")
    }
    $runtimeDependencyAvailable = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("runtime_dependency_available")
    if ($null -eq $runtimeDependencyAvailable) {
        $runtimeDependencyAvailable = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("runtime_dependency_available")
    }
    $localModelLoaded = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("local_model_loaded")
    if ($null -eq $localModelLoaded) {
        $localModelLoaded = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("local_model_loaded")
    }
    $modelSourcePresent = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("model_source_present")
    if ($null -eq $modelSourcePresent) {
        $modelSourcePresent = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("model_source_present")
    }
    $modelLoadAttempted = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("model_load_attempted")
    if ($null -eq $modelLoadAttempted) {
        $modelLoadAttempted = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("model_load_attempted")
    }
    $modelLoadSucceeded = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("model_load_succeeded")
    if ($null -eq $modelLoadSucceeded) {
        $modelLoadSucceeded = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("model_load_succeeded")
    }
    $lastLoadError = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("last_load_error")
    if ($null -eq $lastLoadError) {
        $lastLoadError = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("last_load_error")
    }
    $lastGenerationError = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("last_generation_error")
    if ($null -eq $lastGenerationError) {
        $lastGenerationError = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("last_generation_error")
    }
    $templateApplied = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("template_applied")
    if ($null -eq $templateApplied) {
        $templateApplied = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("template_applied")
    }
    $finishReason = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("finish_reason")
    if ($null -eq $finishReason) {
        $finishReason = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("finish_reason")
    }
    $truncated = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("truncated")
    if ($null -eq $truncated) {
        $truncated = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("truncated")
    }
    $reasoningExtracted = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("reasoning_extracted")
    if ($null -eq $reasoningExtracted) {
        $reasoningExtracted = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("reasoning_extracted")
    }
    $fallbackActive = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("fallback_active")
    if ($null -eq $fallbackActive) {
        $fallbackActive = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("fallback_active")
    }
    $providerWarning = _Get-HarnessNamedProperty -InputObject $HealthPayload -Names @("provider_warning", "warning")
    if ($null -eq $providerWarning) {
        $providerWarning = _Get-HarnessNamedProperty -InputObject $ModelEntry -Names @("provider_warning", "warning")
    }

    return [ordered]@{
        configured_backend = if ($null -eq $configuredBackend) { $null } else { [string]$configuredBackend }
        generation_backend = if ($null -eq $generationBackend) { $null } else { [string]$generationBackend }
        model_source = if ($null -eq $modelSource) { $null } else { [string]$modelSource }
        model_source_type = if ($null -eq $modelSourceType) { $null } else { [string]$modelSourceType }
        model_artifact_format = if ($null -eq $modelArtifactFormat) { $null } else { [string]$modelArtifactFormat }
        provider_store = if ($null -eq $providerStore) { $null } else { [string]$providerStore }
        manifest_path = if ($null -eq $manifestPath) { $null } else { [string]$manifestPath }
        runtime_dependency = if ($null -eq $runtimeDependency) { $null } else { [string]$runtimeDependency }
        runtime_dependency_available = _Convert-HarnessBoolean -Value $runtimeDependencyAvailable
        local_model_loaded = _Convert-HarnessBoolean -Value $localModelLoaded
        model_source_present = _Convert-HarnessBoolean -Value $modelSourcePresent
        model_load_attempted = _Convert-HarnessBoolean -Value $modelLoadAttempted
        model_load_succeeded = _Convert-HarnessBoolean -Value $modelLoadSucceeded
        last_load_error = if ($null -eq $lastLoadError) { $null } else { [string]$lastLoadError }
        last_generation_error = if ($null -eq $lastGenerationError) { $null } else { [string]$lastGenerationError }
        template_applied = _Convert-HarnessBoolean -Value $templateApplied
        finish_reason = if ($null -eq $finishReason) { $null } else { [string]$finishReason }
        truncated = _Convert-HarnessBoolean -Value $truncated
        reasoning_extracted = _Convert-HarnessBoolean -Value $reasoningExtracted
        fallback_active = _Convert-HarnessBoolean -Value $fallbackActive
        provider_warning = if ($null -eq $providerWarning) { $null } else { [string]$providerWarning }
    }
}

function _Get-HarnessPropertyValue {
    param(
        [object]$InputObject,
        [string]$Path
    )

    if ($null -eq $InputObject -or [string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    $current = $InputObject
    foreach ($rawSegment in ($Path -split "\.")) {
        if ($null -eq $current) {
            return $null
        }
        if ([string]::IsNullOrWhiteSpace($rawSegment)) {
            continue
        }

        $segment = [string]$rawSegment
        $index = $null
        if ($segment -match "^(?<name>[^\[]+)\[(?<index>\d+)\]$") {
            $segment = $matches["name"]
            $index = [int]$matches["index"]
        }

        if ($segment -match "^\d+$" -and ($current -is [System.Collections.IEnumerable]) -and ($current -isnot [string])) {
            $items = @($current)
            $numericIndex = [int]$segment
            if ($numericIndex -lt 0 -or $numericIndex -ge $items.Count) {
                return $null
            }
            $current = $items[$numericIndex]
            continue
        }

        if ($current -is [System.Collections.IDictionary]) {
            if (-not $current.Contains($segment)) {
                return $null
            }
            $current = $current[$segment]
        } elseif ($current.PSObject -and $current.PSObject.Properties.Name -contains $segment) {
            $current = $current.$segment
        } else {
            return $null
        }

        if ($null -ne $index) {
            if (($current -isnot [System.Collections.IEnumerable]) -or ($current -is [string])) {
                return $null
            }
            $items = @($current)
            if ($index -lt 0 -or $index -ge $items.Count) {
                return $null
            }
            $current = $items[$index]
        }
    }

    return $current
}

function _Select-HarnessProperties {
    param(
        [object]$InputObject,
        [string[]]$Property
    )

    if ($null -eq $Property -or $Property.Count -eq 0) {
        return $InputObject
    }
    if ($Property.Count -eq 1 -and $Property[0] -eq "*") {
        return $InputObject
    }

    $requiresPathSelection = $false
    foreach ($name in $Property) {
        if ($name -match "[\.\[]") {
            $requiresPathSelection = $true
            break
        }
    }
    if (-not $requiresPathSelection) {
        return $InputObject | Select-Object -Property $Property
    }

    $selected = [ordered]@{}
    foreach ($name in $Property) {
        if ([string]::IsNullOrWhiteSpace($name)) {
            continue
        }
        $selected[$name] = _Get-HarnessPropertyValue -InputObject $InputObject -Path $name
    }
    return [PSCustomObject]$selected
}

function _Register-HarnessTypeData {
    param(
        [string]$TypeName,
        [string[]]$DefaultDisplayPropertySet
    )

    if ([string]::IsNullOrWhiteSpace($TypeName) -or $null -eq $DefaultDisplayPropertySet -or $DefaultDisplayPropertySet.Count -eq 0) {
        return
    }
    try {
        Update-TypeData -TypeName $TypeName -DefaultDisplayPropertySet $DefaultDisplayPropertySet -Force -ErrorAction Stop
    } catch {
        # Display metadata is best-effort and should never change command behavior.
    }
}

function _Add-HarnessTypeName {
    param(
        [object]$InputObject,
        [string]$TypeName,
        [string[]]$DefaultDisplayPropertySet
    )

    if ($null -eq $InputObject -or [string]::IsNullOrWhiteSpace($TypeName)) {
        return $InputObject
    }
    _Register-HarnessTypeData -TypeName $TypeName -DefaultDisplayPropertySet $DefaultDisplayPropertySet
    if ($InputObject.PSObject -and $InputObject.PSObject.TypeNames[0] -ne $TypeName) {
        $InputObject.PSObject.TypeNames.Insert(0, $TypeName)
    }
    return $InputObject
}

function _New-HarnessObject {
    param(
        [string]$TypeName,
        [object]$Property,
        [string[]]$DefaultDisplayPropertySet
    )

    $obj = [PSCustomObject]$Property
    return _Add-HarnessTypeName -InputObject $obj -TypeName $TypeName -DefaultDisplayPropertySet $DefaultDisplayPropertySet
}

function _Apply-HarnessOutputOptions {
    param(
        [object]$InputObject,
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [bool]$AsJson = $false,
        [int]$JsonDepth = 20
    )

    if ($JsonDepth -le 0) {
        throw "JsonDepth must be > 0"
    }

    $output = $InputObject
    if (-not [string]::IsNullOrWhiteSpace($ExpandProperty)) {
        $output = _Get-HarnessPropertyValue -InputObject $output -Path $ExpandProperty
    } elseif ($null -ne $Property -and $Property.Count -gt 0) {
        $output = _Select-HarnessProperties -InputObject $output -Property $Property
    }

    if ($AsJson) {
        return $output | ConvertTo-Json -Depth $JsonDepth
    }
    return $output
}

function _Get-HarnessOneShotAnswer {
    param([object]$Response)

    $answer = _Get-HarnessPropertyValue -InputObject $Response -Path "choices[0].message.content"
    if ($null -eq $answer) {
        return ""
    }
    return [string]$answer
}

function _Add-HarnessOneShotConvenience {
    param([object]$Response)

    if ($null -eq $Response -or -not $Response.PSObject) {
        return $Response
    }
    $answer = _Get-HarnessOneShotAnswer -Response $Response
    if ($Response.PSObject.Properties.Name -contains "answer") {
        $Response.answer = $answer
    } else {
        $Response | Add-Member -NotePropertyName answer -NotePropertyValue $answer -Force
    }
    $provider = _Get-HarnessPropertyValue -InputObject $Response -Path "meta.provider"
    if ($null -eq $provider) {
        $provider = _Get-HarnessNamedProperty -InputObject $Response -Names @("provider")
    }
    if ($null -ne $provider) {
        foreach ($field in @("configured_backend", "generation_backend", "model_source", "model_source_type", "model_artifact_format", "provider_store", "manifest_path", "runtime_dependency", "runtime_dependency_available", "local_model_loaded", "model_source_present", "model_load_attempted", "model_load_succeeded", "last_load_error", "last_generation_error", "template_applied", "fallback_active", "allow_fallback", "finish_reason", "truncated", "reasoning_extracted", "warnings", "provider_warning")) {
            $value = _Get-HarnessNamedProperty -InputObject $provider -Names @($field)
            if ($null -eq $value -and $Response.PSObject.Properties.Name -contains $field) {
                continue
            }
            if ($Response.PSObject.Properties.Name -contains $field) {
                $Response.$field = $value
            } else {
                $Response | Add-Member -NotePropertyName $field -NotePropertyValue $value -Force
            }
        }
    }
    return _Add-HarnessTypeName `
        -InputObject $Response `
        -TypeName "Harness.OneShot.Response" `
        -DefaultDisplayPropertySet @("answer", "status", "model", "route", "generation_backend", "finish_reason", "truncated", "provider_warning", "run_id", "next_action")
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

<#
.SYNOPSIS
Gets runtime and provider-plane backend status.
.DESCRIPTION
Returns scalar diagnostics plus nested health, model catalog, and optional session details for pipeline inspection.
#>
function Get-HarnessBackendStatus {
    [CmdletBinding()]
    param(
        [string]$Config = "harness.yaml",
        [string]$ServerHost = "127.0.0.1",
        [int]$Port = 8080,
        [int]$RequestTimeoutSeconds = 6,
        [string]$ConfigProviderProfile = "",
        [switch]$PreferProviderOnly,
        [switch]$IncludeSession,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
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
        catalog_payload = $null
        model_entry = $null
        configured_backend = $null
        generation_backend = $null
        model_source = $null
        model_source_type = $null
        model_artifact_format = $null
        provider_store = $null
        manifest_path = $null
        runtime_dependency = $null
        runtime_dependency_available = $false
        local_model_loaded = $false
        model_source_present = $false
        model_load_attempted = $false
        model_load_succeeded = $false
        last_load_error = $null
        last_generation_error = $null
        template_applied = $false
        finish_reason = $null
        truncated = $false
        reasoning_extracted = $false
        fallback_active = $false
        provider_warning = $null
    }

    $status = [ordered]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        config = $resolvedConfig
        backend_name = $backendName
        selected_model = $backendModel
        backend_base_url = $backendUrl
        runtime_reachable = $false
        runtime_status_code = $null
        runtime_error = $null
        provider_reachable = $false
        provider_status_code = $null
        provider_error = $null
        model_present_in_catalog = $false
        configured_backend = $null
        generation_backend = $null
        model_source = $null
        model_source_type = $null
        model_artifact_format = $null
        provider_store = $null
        manifest_path = $null
        runtime_dependency = $null
        runtime_dependency_available = $false
        local_model_loaded = $false
        model_source_present = $false
        model_load_attempted = $false
        model_load_succeeded = $false
        last_load_error = $null
        last_generation_error = $null
        template_applied = $false
        finish_reason = $null
        truncated = $false
        reasoning_extracted = $false
        fallback_active = $false
        provider_warning = $null
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
                $status.provider_plane.catalog_payload = $catalogPayload
                $status.provider_plane.model_entry = _Get-ModelCatalogEntry -CatalogPayload $catalogPayload -ModelName $backendModel
                $diagnostics = _Get-ProviderDiagnosticsFromPayloads -HealthPayload $null -ModelEntry $status.provider_plane.model_entry
                $status.provider_plane.configured_backend = $diagnostics.configured_backend
                $status.provider_plane.generation_backend = $diagnostics.generation_backend
                $status.provider_plane.model_source = $diagnostics.model_source
                $status.provider_plane.model_source_type = $diagnostics.model_source_type
                $status.provider_plane.model_artifact_format = $diagnostics.model_artifact_format
                $status.provider_plane.provider_store = $diagnostics.provider_store
                $status.provider_plane.manifest_path = $diagnostics.manifest_path
                $status.provider_plane.runtime_dependency = $diagnostics.runtime_dependency
                $status.provider_plane.runtime_dependency_available = [bool]$diagnostics.runtime_dependency_available
                $status.provider_plane.local_model_loaded = [bool]$diagnostics.local_model_loaded
                $status.provider_plane.model_source_present = [bool]$diagnostics.model_source_present
                $status.provider_plane.model_load_attempted = [bool]$diagnostics.model_load_attempted
                $status.provider_plane.model_load_succeeded = [bool]$diagnostics.model_load_succeeded
                $status.provider_plane.last_load_error = $diagnostics.last_load_error
                $status.provider_plane.last_generation_error = $diagnostics.last_generation_error
                $status.provider_plane.template_applied = [bool]$diagnostics.template_applied
                $status.provider_plane.finish_reason = $diagnostics.finish_reason
                $status.provider_plane.truncated = [bool]$diagnostics.truncated
                $status.provider_plane.reasoning_extracted = [bool]$diagnostics.reasoning_extracted
                $status.provider_plane.fallback_active = [bool]$diagnostics.fallback_active
                $status.provider_plane.provider_warning = $diagnostics.provider_warning
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

    $status.runtime_reachable = [bool]$status.server.reachable
    $status.runtime_status_code = $status.server.status_code
    $status.runtime_error = $status.server.error
    $status.provider_reachable = [bool]$status.provider_plane.reachable
    $status.provider_status_code = $status.provider_plane.status_code
    $status.provider_error = $status.provider_plane.error
    $status.model_present_in_catalog = [bool]$status.provider_plane.model_present_in_catalog
    $status.configured_backend = $status.provider_plane.configured_backend
    $status.generation_backend = $status.provider_plane.generation_backend
    $status.model_source = $status.provider_plane.model_source
    $status.model_source_type = $status.provider_plane.model_source_type
    $status.model_artifact_format = $status.provider_plane.model_artifact_format
    $status.provider_store = $status.provider_plane.provider_store
    $status.manifest_path = $status.provider_plane.manifest_path
    $status.runtime_dependency = $status.provider_plane.runtime_dependency
    $status.runtime_dependency_available = [bool]$status.provider_plane.runtime_dependency_available
    $status.local_model_loaded = [bool]$status.provider_plane.local_model_loaded
    $status.model_source_present = [bool]$status.provider_plane.model_source_present
    $status.model_load_attempted = [bool]$status.provider_plane.model_load_attempted
    $status.model_load_succeeded = [bool]$status.provider_plane.model_load_succeeded
    $status.last_load_error = $status.provider_plane.last_load_error
    $status.last_generation_error = $status.provider_plane.last_generation_error
    $status.template_applied = [bool]$status.provider_plane.template_applied
    $status.finish_reason = $status.provider_plane.finish_reason
    $status.truncated = [bool]$status.provider_plane.truncated
    $status.reasoning_extracted = [bool]$status.provider_plane.reasoning_extracted
    $status.fallback_active = [bool]$status.provider_plane.fallback_active
    $status.provider_warning = $status.provider_plane.provider_warning

    $result = _New-HarnessObject `
        -TypeName "Harness.Backend.Status" `
        -Property $status `
        -DefaultDisplayPropertySet @(
            "backend_name",
            "selected_model",
            "runtime_reachable",
            "provider_reachable",
            "model_present_in_catalog",
            "generation_backend",
            "model_source_type",
            "model_artifact_format",
            "provider_store",
            "runtime_dependency_available",
            "model_source_present",
            "model_load_attempted",
            "model_load_succeeded",
            "template_applied",
            "truncated",
            "fallback_active",
            "runtime_error",
            "provider_error",
            "provider_warning"
        )
    return _Apply-HarnessOutputOptions `
        -InputObject $result `
        -Property $Property `
        -ExpandProperty $ExpandProperty `
        -AsJson:$AsJson.IsPresent `
        -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Starts the harness runtime backend.
.DESCRIPTION
Starts the local Python runtime or the configured container profile and returns a rich object for pipeline use.
#>
function Start-HarnessBackend {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [Parameter(Position = 0)]
        [ValidateSet("local", "containerized")]
        [string]$ExecutionMode = "local",

        [string]$Config = "harness.yaml",
        [string]$ServerHost = "127.0.0.1",
        [int]$Port = 8080,
        [string]$PythonPath = "",
        [string]$ContainerProfile = "",
        [string]$ComposeFile = "docker-compose.nvidia.yaml",
        [string]$EnvFile = "",
        [switch]$NoBuild,
        [int]$WaitSeconds = 30,
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    if ($WaitSeconds -lt 0) {
        throw "WaitSeconds must be >= 0"
    }

    $resolvedConfig = _Resolve-FilePath -Path $Config -Base (Get-Location).Path
    $context = _Load-BackendContext -ConfigPath $resolvedConfig
    $backendName = [string]$context.backend_name

    $healthUrl = _HealthUrl -TargetHost $ServerHost -Port $Port
    if ($ExecutionMode -eq "local") {
        $resolvedPythonPath = _Resolve-HarnessPythonPath -PythonPath $PythonPath
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
        $command = _Format-HarnessCommand -Executable $resolvedPythonPath -Arguments $startArgs
        $entryKey = _Session-Key -Mode "local" -ConfigPath $resolvedConfig -ServerHost $ServerHost -Port $Port -Profile ""

        $existingSessions = _Prune-StaleSessions -Sessions (_Read-HarnessBackendSessions)
        $runningSession = _Find-LocalSessionMatch `
            -Sessions $existingSessions `
            -ConfigPath $resolvedConfig `
            -ServerHost $ServerHost `
            -Port $Port

        if ($null -ne $runningSession) {
            $result = _New-HarnessObject -TypeName "Harness.Backend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "host", "port", "process_id", "health_url") -Property @{
                mode = "local"
                started = $false
                action = "already_running"
                config = $resolvedConfig
                host = $ServerHost
                port = $Port
                python_path = if ($runningSession.PSObject.Properties.Name -contains "python_path") { $runningSession.python_path } else { $resolvedPythonPath }
                command = $command
                health_url = $healthUrl
                process_id = [int]$runningSession.process_id
                stdout_log = if ($runningSession.PSObject.Properties.Name -contains "stdout_log") { $runningSession.stdout_log } else { $null }
                stderr_log = if ($runningSession.PSObject.Properties.Name -contains "stderr_log") { $runningSession.stderr_log } else { $null }
                session_file = $Script:HarnessBackendSessionFile
            }
            return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
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
            $result = _New-HarnessObject -TypeName "Harness.Backend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "host", "port", "health_url") -Property @{
                mode = "local"
                started = $false
                config = $resolvedConfig
                host = $ServerHost
                port = $Port
                python_path = $resolvedPythonPath
                command = $command
                health_url = $healthUrl
                session_file = $Script:HarnessBackendSessionFile
            }
            return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
        }

        $logPaths = _New-HarnessLogPaths -Prefix "runtime-$Port"
        $stdoutPath = [string]$logPaths.stdout_log
        $stderrPath = [string]$logPaths.stderr_log
        if (-not $PSCmdlet.ShouldProcess("$ServerHost`:$Port", "Start harness runtime backend")) {
            $result = _New-HarnessObject -TypeName "Harness.Backend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "host", "port", "health_url") -Property @{
                mode = "local"
                started = $false
                action = "whatif"
                config = $resolvedConfig
                host = $ServerHost
                port = $Port
                python_path = $resolvedPythonPath
                command = $command
                health_url = $healthUrl
                stdout_log = $stdoutPath
                stderr_log = $stderrPath
                session_file = $Script:HarnessBackendSessionFile
            }
            return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
        }
        try {
            $process = Start-Process -FilePath $resolvedPythonPath -ArgumentList $startArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        } catch {
            throw "Failed to start harness server process."
        }
        if ($null -eq $process) {
            throw "Failed to start harness server process."
        }
        if ($process.HasExited) {
            $startupError = ""
            if (Test-Path $stderrPath) {
                $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
            }
            throw "Harness server exited immediately with code $($process.ExitCode). $startupError"
        }

        if (-not (_Wait-HttpReady -HealthUrl $healthUrl -TimeoutSeconds $WaitSeconds)) {
            $startupError = ""
            if (Test-Path $stderrPath) {
                $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
            }
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            throw "Harness server at $($healthUrl) did not become ready after ${WaitSeconds}s. $startupError"
        }

        $entry = @{
            key = $entryKey
            mode = "local"
            process_id = $process.Id
            config = $resolvedConfig
            host = $ServerHost
            port = $Port
            python_path = $resolvedPythonPath
            command = $command
            health_url = $healthUrl
            stdout_log = $stdoutPath
            stderr_log = $stderrPath
            started_utc = (Get-Date).ToUniversalTime().ToString("o")
        }

        $sessions = @($existingSessions | Where-Object { $_.key -ne $entry.key })
        $sessions += $entry
        _Write-HarnessBackendSessions -Sessions $sessions

        $result = _New-HarnessObject -TypeName "Harness.Backend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "host", "port", "process_id", "health_url") -Property @{
            mode = "local"
            started = $true
            config = $resolvedConfig
            host = $ServerHost
            port = $Port
            python_path = $resolvedPythonPath
            command = $command
            health_url = $healthUrl
            process_id = $process.Id
            stdout_log = $stdoutPath
            stderr_log = $stderrPath
            session_file = $Script:HarnessBackendSessionFile
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    $profile = $ContainerProfile
    if ([string]::IsNullOrWhiteSpace($profile)) {
        $profile = _Resolve-ContainerProfile -BackendName $backendName
    }

    if ([string]::IsNullOrWhiteSpace($profile)) {
        if ($backendName -eq "ollama") {
            throw "Container profile for backend 'ollama' is opt-in only. Specify -ContainerProfile ollama and an Ollama env file explicitly."
        }
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

    $command = _Format-HarnessCommand -Executable "docker" -Arguments $composeArgs
    if ($DryRun) {
        $result = _New-HarnessObject -TypeName "Harness.Backend.StartResult" -DefaultDisplayPropertySet @("mode", "started", "profile", "config", "health_url") -Property @{
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
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    if (-not $PSCmdlet.ShouldProcess($resolvedConfig, "Start harness container backend")) {
        $result = _New-HarnessObject -TypeName "Harness.Backend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "profile", "config", "health_url") -Property @{
            mode = "containerized"
            started = $false
            action = "whatif"
            profile = $profile
            config = $resolvedConfig
            compose_file = $resolvedCompose
            env_file = $resolvedEnvFile
            command = $command
            health_url = $healthUrl
            session_file = $Script:HarnessBackendSessionFile
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
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

    $result = _New-HarnessObject -TypeName "Harness.Backend.StartResult" -DefaultDisplayPropertySet @("mode", "started", "profile", "config", "health_url") -Property @{
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
    return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Stops tracked harness runtime backends.
.DESCRIPTION
Stops local runtime processes or the configured container stack and returns the targeted sessions.
#>
function Stop-HarnessBackend {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
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
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
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
        $shouldStopLocal = $DryRun.IsPresent -or $PSCmdlet.ShouldProcess("$ServerHost`:$Port", "Stop harness runtime backend")
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
            if ($DryRun -or -not $shouldStopLocal) {
                if (-not $DryRun) {
                    $remaining += $entry
                }
                continue
            }
            if ($entry.process_id) {
                if (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue) {
                    Stop-Process -Id $entry.process_id -Force -ErrorAction SilentlyContinue
                }
            }
        }

        if ($DryRun -or -not $shouldStopLocal) {
            $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count") -Property @{
                mode = "local"
                action = if ($DryRun) { "stopped" } else { "whatif" }
                removed_count = $removed.Count
                removed = $removed
            }
            return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
        }

        if ($removed.Count -gt 0) {
            _Write-HarnessBackendSessions -Sessions $remaining
        }
        $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count") -Property @{
            mode = "local"
            action = "stopped"
            removed_count = $removed.Count
            removed = $removed
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
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
        if ([string]$backendContext.backend_name -eq "ollama") {
            throw "Could not resolve container profile for backend 'ollama'. Use -ContainerProfile ollama explicitly for explicit opt-in stacks."
        }
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
        $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count", "profile", "config") -Property @{
            mode = "containerized"
            action = "stopped"
            removed_count = $removed.Count
            command = $command
            profile = $profile
            config = $resolvedConfig
            compose_file = $resolvedCompose
            env_file = $resolvedEnvFile
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    $remaining = @($sessions | Where-Object { $_.mode -ne "containerized" -or $_.config -ne $resolvedConfig })
    foreach ($entry in $sessions) {
        if ($entry.mode -eq "containerized" -and $entry.config -eq $resolvedConfig) {
            $removed += $entry
        }
    }

    if (-not $PSCmdlet.ShouldProcess($resolvedConfig, "Stop harness container backend")) {
        $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count", "profile", "config") -Property @{
            mode = "containerized"
            action = "whatif"
            removed_count = $removed.Count
            command = $command
            profile = $profile
            config = $resolvedConfig
            compose_file = $resolvedCompose
            env_file = $resolvedEnvFile
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    $compose = Start-Process -FilePath "docker" -ArgumentList $composeArgs -PassThru -NoNewWindow -Wait
    if ($compose.ExitCode -ne 0) {
        throw "docker compose stop failed with exit code $($compose.ExitCode)"
    }

    if ($removed.Count -gt 0) {
        _Write-HarnessBackendSessions -Sessions $remaining
    }
    $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count", "profile", "config") -Property @{
        mode = "containerized"
        action = "stopped"
        removed_count = $removed.Count
        command = $command
        profile = $profile
        config = $resolvedConfig
    }
    return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Starts the harness-owned local model backend.
.DESCRIPTION
Starts the OpenAI-compatible local provider for fallback, Transformers, or llama.cpp-backed local models.
#>
function Start-HarnessModelBackend {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [string]$Model = $Script:HarnessModelBackendDefaultModel,
        [ValidateSet("auto", "fallback", "transformers", "llamacpp", "llama_cpp", "llama-cpp")]
        [string]$Backend = "auto",
        [string]$ModelPath = "",
        [string]$ModelsRoot = "",
        [string[]]$ExtraModel,
        [string]$Device = "cpu",
        [int]$MaxNewTokens = 256,
        [int]$LlamaCppContext = 4096,
        [int]$LlamaCppGpuLayers = 0,
        [int]$LlamaCppThreads = 4,
        [switch]$LocalOnly,
        [switch]$AllowFallback,
        [int]$WaitSeconds = 30,
        [string]$PythonPath = "",
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    if ($WaitSeconds -lt 0) {
        throw "WaitSeconds must be >= 0"
    }
    if ([string]::IsNullOrWhiteSpace($Model)) {
        $Model = $Script:HarnessModelBackendDefaultModel
    }
    if ($Backend -in @("llama_cpp", "llama-cpp")) {
        $Backend = "llamacpp"
    }
    $modelHasInlineSource = [bool]($Model -match "(::|=)")
    $modelPathProvided = -not [string]::IsNullOrWhiteSpace($ModelPath)
    $modelsRootProvided = -not [string]::IsNullOrWhiteSpace($ModelsRoot)
    $tagLikeModelId = [bool]($Model -match "^[^\\/:=]+:[^\\/:=]+$")
    $huggingFaceCacheModelId = [bool]($Model -match "(?i)^hf://")
    $sourceResolution = _Resolve-HarnessModelBackendSource -Model $Model -ModelPath $ModelPath -ModelsRoot $ModelsRoot
    $sourceResolved = [bool]$sourceResolution.resolved
    $extraModelSourceMatched = $false
    $primaryModelHasSource = $modelPathProvided -or $modelsRootProvided -or $modelHasInlineSource
    if (-not $primaryModelHasSource -and $ExtraModel -and $ExtraModel.Count -gt 0) {
        foreach ($extra in $ExtraModel) {
            if ([string]::IsNullOrWhiteSpace($extra)) { continue }
            if ($extra -match "^\s*(?<id>[^:=]+)\s*(::|=)\s*(?<source>.+)$") {
                if ($matches["id"].Trim() -eq $Model -and -not [string]::IsNullOrWhiteSpace($matches["source"])) {
                    $extraModelSourceMatched = $true
                    $primaryModelHasSource = $true
                    break
                }
            }
        }
    }
    $fallbackAllowed = [bool]($AllowFallback.IsPresent -or $Backend -eq "fallback")
    $primaryModelCanResolveLocally = [bool]($sourceResolved -or $extraModelSourceMatched -or $modelHasInlineSource)
    $extraModelCount = if ($ExtraModel) { $ExtraModel.Count } else { 0 }
    Write-Verbose ("Start-HarnessModelBackend resolving startup: model='{0}', backend='{1}', host='{2}', port={3}, modelPathSet={4}, modelsRootSet={5}, extraModelCount={6}, allowFallback={7}, localOnly={8}" -f $Model, $Backend, $ModelBackendHost, $ModelBackendPort, $modelPathProvided, $modelsRootProvided, $extraModelCount, $fallbackAllowed, [bool]$LocalOnly.IsPresent)
    Write-Debug ("Model source detection: inlineSource={0}; modelPathSet={1}; modelsRootSet={2}; extraModelSourceMatched={3}; primaryModelHasSource={4}; primaryModelCanResolveLocally={5}; sourceResolved={6}; sourceType={7}; fallbackAllowed={8}; tagLikeModelId={9}; huggingFaceCacheModelId={10}" -f $modelHasInlineSource, $modelPathProvided, $modelsRootProvided, $extraModelSourceMatched, $primaryModelHasSource, $primaryModelCanResolveLocally, $sourceResolved, $sourceResolution.source_type, $fallbackAllowed, $tagLikeModelId, $huggingFaceCacheModelId)
    if ($modelPathProvided -and -not $sourceResolved -and -not $fallbackAllowed) {
        throw "Start-HarnessModelBackend: -ModelPath does not exist or could not be resolved: $($sourceResolution.source)"
    }
    if ($Backend -eq "auto" -and -not $fallbackAllowed -and -not $primaryModelCanResolveLocally) {
        Write-Verbose "Start-HarnessModelBackend blocked startup before launching python because -Backend auto has no local model source and fallback is not explicit."
        Write-Debug "No process was started. Provide -ModelPath/-ModelsRoot for a real local model artifact, or choose -Backend fallback/-AllowFallback for deterministic diagnostic stub mode."
        if ($tagLikeModelId) {
            Write-Debug ("Model '{0}' looks like a name:tag model identifier and can be resolved from a local model store when its manifest/blob files are present." -f $Model)
        }
        $guardMessage = @(
            "Start-HarnessModelBackend: -Backend auto requires a real local model source before it can launch."
            "Model='$Model'; Backend='$Backend'; ModelPath='<empty>'; ModelsRoot='<empty>'; ExtraModelCount=$extraModelCount; AllowFallback=$fallbackAllowed."
            "Use -ModelPath '<local Hugging Face model folder or GGUF file>', -ModelsRoot '<local model root>', an hf:// model id already present in the local Hugging Face cache, or a tag already present in a local Ollama-style model store."
            "Use -Backend fallback or -AllowFallback only when you intentionally want deterministic diagnostic stub mode."
        )
        throw ($guardMessage -join " ")
    }
    $expectedGenerationBackend = if ($Backend -eq "fallback") {
        "fallback"
    } elseif ($Backend -eq "transformers") {
        "transformers"
    } elseif ($Backend -eq "llamacpp") {
        "llamacpp"
    } elseif ($sourceResolved -and -not [string]::IsNullOrWhiteSpace($sourceResolution.generation_backend)) {
        [string]$sourceResolution.generation_backend
    } elseif ($fallbackAllowed) {
        "fallback"
    } else {
        "transformers"
    }
    $expectedModelSource = if ($sourceResolved) { $sourceResolution.source } else { $null }
    $expectedModelSourceType = if ($sourceResolved) { $sourceResolution.source_type } else { $null }
    $expectedArtifactFormat = if ($sourceResolved) { $sourceResolution.artifact_format } else { $null }
    $expectedProviderStore = if ($sourceResolved) { $sourceResolution.provider_store } else { $null }
    $expectedManifestPath = if ($sourceResolved) { $sourceResolution.manifest_path } else { $null }
    $expectedLocalModelLoaded = [bool](
        $expectedGenerationBackend -in @("transformers", "llamacpp") -and
        -not [string]::IsNullOrWhiteSpace($expectedModelSource) -and
        (
            $expectedModelSourceType -ne "huggingface_cache" -or
            (_Test-HarnessHuggingFaceCacheHasWeights -CacheDir ([string]$sourceResolution.manifest_path))
        )
    )
    $providerWarning = if ($expectedGenerationBackend -eq "fallback") {
        "Deterministic fallback provider is active. This is diagnostic stub mode, not a real LLM."
    } else {
        $null
    }
    $expectedModelSourcePresent = [bool](
        -not [string]::IsNullOrWhiteSpace($expectedModelSource) -and
        (
            $expectedModelSourceType -eq "huggingface_cache" -or
            (Test-Path -LiteralPath ([string]$expectedModelSource) -ErrorAction SilentlyContinue)
        )
    )
    $expectedTemplateApplied = [bool]($expectedGenerationBackend -eq "llamacpp" -and $expectedProviderStore -eq "ollama")

    $healthUrl = _HealthUrl -TargetHost $ModelBackendHost -Port $ModelBackendPort
    $entryKey = "model-backend|$ModelBackendHost|$ModelBackendPort"
    $resolvedPythonPath = _Resolve-HarnessPythonPath -PythonPath $PythonPath
    $startArgs = @(
        "-m",
        "harness.local_model_provider",
        "--host",
        $ModelBackendHost,
        "--port",
        $ModelBackendPort.ToString(),
        "--model",
        $Model,
        "--backend",
        $Backend,
        "--device",
        $Device,
        "--max-new-tokens",
        $MaxNewTokens.ToString(),
        "--llama-cpp-n-ctx",
        $LlamaCppContext.ToString(),
        "--llama-cpp-n-gpu-layers",
        $LlamaCppGpuLayers.ToString(),
        "--llama-cpp-n-threads",
        $LlamaCppThreads.ToString()
    )
    if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
        $startArgs += @("--model-path", $ModelPath)
    }
    if (-not [string]::IsNullOrWhiteSpace($ModelsRoot)) {
        $startArgs += @("--models-root", $ModelsRoot)
    }
    if ($ExtraModel -and $ExtraModel.Count -gt 0) {
        foreach ($extra in $ExtraModel) {
            if ([string]::IsNullOrWhiteSpace($extra)) { continue }
            $startArgs += @("--extra-model", $extra)
        }
    }
    if ($LocalOnly.IsPresent) {
        $startArgs += "--local-only"
    }
    if ($AllowFallback.IsPresent) {
        $startArgs += "--allow-fallback"
    }
    $command = _Format-HarnessCommand -Executable $resolvedPythonPath -Arguments $startArgs

    $existingSessions = _Prune-StaleModelSessions -Sessions (_Read-HarnessModelBackendSessions)
    $runningSession = _Find-LiveModelBackendSession -Sessions $existingSessions -ModelBackendHost $ModelBackendHost -ModelBackendPort $ModelBackendPort
    if ($null -ne $runningSession) {
        $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "host", "port", "model", "generation_backend", "fallback_active", "provider_warning", "process_id", "health_url") -Property @{
            mode = "model_backend"
            started = $false
            action = "already_running"
            host = $ModelBackendHost
            port = $ModelBackendPort
            model = $Model
            configured_backend = $Backend
            generation_backend = $expectedGenerationBackend
            model_source = $expectedModelSource
            model_source_type = $expectedModelSourceType
            model_artifact_format = $expectedArtifactFormat
            provider_store = $expectedProviderStore
            manifest_path = $expectedManifestPath
            local_model_loaded = $expectedLocalModelLoaded
            model_source_present = $expectedModelSourcePresent
            model_load_attempted = $false
            model_load_succeeded = $false
            last_load_error = $null
            last_generation_error = $null
            template_applied = $expectedTemplateApplied
            fallback_active = [bool]($expectedGenerationBackend -eq "fallback")
            allow_fallback = $fallbackAllowed
            provider_warning = $providerWarning
            python_path = if ($runningSession.PSObject.Properties.Name -contains "python_path") { $runningSession.python_path } else { $resolvedPythonPath }
            command = $command
            health_url = $healthUrl
            process_id = [int]$runningSession.process_id
            stdout_log = if ($runningSession.PSObject.Properties.Name -contains "stdout_log") { $runningSession.stdout_log } else { $null }
            stderr_log = if ($runningSession.PSObject.Properties.Name -contains "stderr_log") { $runningSession.stderr_log } else { $null }
            session_file = $Script:HarnessModelBackendSessionFile
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
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
        $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StartResult" -DefaultDisplayPropertySet @("mode", "started", "host", "port", "model", "generation_backend", "fallback_active", "provider_warning", "health_url") -Property @{
            mode = "model_backend"
            started = $false
            host = $ModelBackendHost
            port = $ModelBackendPort
            model = $Model
            configured_backend = $Backend
            generation_backend = $expectedGenerationBackend
            model_source = $expectedModelSource
            model_source_type = $expectedModelSourceType
            model_artifact_format = $expectedArtifactFormat
            provider_store = $expectedProviderStore
            manifest_path = $expectedManifestPath
            local_model_loaded = $expectedLocalModelLoaded
            model_source_present = $expectedModelSourcePresent
            model_load_attempted = $false
            model_load_succeeded = $false
            last_load_error = $null
            last_generation_error = $null
            template_applied = $expectedTemplateApplied
            fallback_active = [bool]($expectedGenerationBackend -eq "fallback")
            allow_fallback = $fallbackAllowed
            provider_warning = $providerWarning
            python_path = $resolvedPythonPath
            command = $command
            health_url = $healthUrl
            session_file = $Script:HarnessModelBackendSessionFile
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    $logPaths = _New-HarnessLogPaths -Prefix "model-$ModelBackendPort"
    $stdoutPath = [string]$logPaths.stdout_log
    $stderrPath = [string]$logPaths.stderr_log
    if (-not $PSCmdlet.ShouldProcess("$ModelBackendHost`:$ModelBackendPort", "Start harness local model backend")) {
        $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "host", "port", "model", "generation_backend", "fallback_active", "provider_warning", "health_url") -Property @{
            mode = "model_backend"
            started = $false
            action = "whatif"
            host = $ModelBackendHost
            port = $ModelBackendPort
            model = $Model
            configured_backend = $Backend
            generation_backend = $expectedGenerationBackend
            model_source = $expectedModelSource
            model_source_type = $expectedModelSourceType
            model_artifact_format = $expectedArtifactFormat
            provider_store = $expectedProviderStore
            manifest_path = $expectedManifestPath
            local_model_loaded = $expectedLocalModelLoaded
            model_source_present = $expectedModelSourcePresent
            model_load_attempted = $false
            model_load_succeeded = $false
            last_load_error = $null
            last_generation_error = $null
            template_applied = $expectedTemplateApplied
            fallback_active = [bool]($expectedGenerationBackend -eq "fallback")
            allow_fallback = $fallbackAllowed
            provider_warning = $providerWarning
            python_path = $resolvedPythonPath
            command = $command
            health_url = $healthUrl
            stdout_log = $stdoutPath
            stderr_log = $stderrPath
            session_file = $Script:HarnessModelBackendSessionFile
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }
    try {
        $process = Start-Process -FilePath $resolvedPythonPath -ArgumentList $startArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    } catch {
        throw "Failed to start local model backend process."
    }
    if ($null -eq $process) {
        throw "Failed to start local model backend process."
    }
    if ($process.HasExited) {
        $startupError = ""
        if (Test-Path $stderrPath) {
            $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
        }
        throw "Local model backend exited immediately with code $($process.ExitCode). $startupError"
    }

    if (-not (_Wait-HttpReady -HealthUrl $healthUrl -TimeoutSeconds $WaitSeconds)) {
        $startupError = ""
        if (Test-Path $stderrPath) {
            $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
        }
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        throw "Local model backend at $($healthUrl) did not become ready after ${WaitSeconds}s. $startupError"
    }

    $entry = @{
        key = $entryKey
        mode = "model_backend"
        process_id = $process.Id
        host = $ModelBackendHost
        port = $ModelBackendPort
        model = $Model
        configured_backend = $Backend
        generation_backend = $expectedGenerationBackend
        model_source = $expectedModelSource
        model_source_type = $expectedModelSourceType
        model_artifact_format = $expectedArtifactFormat
        provider_store = $expectedProviderStore
        manifest_path = $expectedManifestPath
        local_model_loaded = $expectedLocalModelLoaded
        model_source_present = $expectedModelSourcePresent
        model_load_attempted = $false
        model_load_succeeded = $false
        last_load_error = $null
        last_generation_error = $null
        template_applied = $expectedTemplateApplied
        fallback_active = [bool]($expectedGenerationBackend -eq "fallback")
        allow_fallback = $fallbackAllowed
        provider_warning = $providerWarning
        python_path = $resolvedPythonPath
        command = $command
        health_url = $healthUrl
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        started_utc = (Get-Date).ToUniversalTime().ToString("o")
    }

    $sessions = @($existingSessions | Where-Object { $_.key -ne $entry.key })
    $sessions += $entry
    _Write-HarnessModelBackendSessions -Sessions $sessions

    $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StartResult" -DefaultDisplayPropertySet @("mode", "started", "host", "port", "model", "generation_backend", "fallback_active", "provider_warning", "process_id", "health_url") -Property @{
        mode = "model_backend"
        started = $true
        host = $ModelBackendHost
        port = $ModelBackendPort
        model = $Model
        configured_backend = $Backend
        generation_backend = $expectedGenerationBackend
        model_source = $expectedModelSource
        model_source_type = $expectedModelSourceType
        model_artifact_format = $expectedArtifactFormat
        provider_store = $expectedProviderStore
        manifest_path = $expectedManifestPath
        local_model_loaded = $expectedLocalModelLoaded
        model_source_present = $expectedModelSourcePresent
        model_load_attempted = $false
        model_load_succeeded = $false
        last_load_error = $null
        last_generation_error = $null
        template_applied = $expectedTemplateApplied
        fallback_active = [bool]($expectedGenerationBackend -eq "fallback")
        allow_fallback = $fallbackAllowed
        provider_warning = $providerWarning
        python_path = $resolvedPythonPath
        command = $command
        health_url = $healthUrl
        process_id = $process.Id
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
        session_file = $Script:HarnessModelBackendSessionFile
    }
    return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Starts the harness-owned local LLM backend.
.DESCRIPTION
Convenience wrapper around Start-HarnessModelBackend for local model development and diagnostics.
#>
function Start-HarnessOwnLLMBackend {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [string]$Model = $Script:HarnessModelBackendDefaultModel,
        [string]$ModelPath = "",
        [string]$ModelsRoot = "",
        [string[]]$ExtraModel,
        [string]$Backend = "auto",
        [string]$Device = "cpu",
        [int]$MaxNewTokens = 256,
        [int]$LlamaCppContext = 4096,
        [int]$LlamaCppGpuLayers = 0,
        [int]$LlamaCppThreads = 4,
        [switch]$LocalOnly,
        [switch]$AllowFallback,
        [int]$WaitSeconds = 30,
        [string]$PythonPath = "",
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    return Start-HarnessModelBackend `
        -ModelBackendHost $ModelBackendHost `
        -ModelBackendPort $ModelBackendPort `
        -Model $Model `
        -Backend $Backend `
        -ModelPath $ModelPath `
        -ModelsRoot $ModelsRoot `
        -ExtraModel $ExtraModel `
        -Device $Device `
        -MaxNewTokens $MaxNewTokens `
        -LlamaCppContext $LlamaCppContext `
        -LlamaCppGpuLayers $LlamaCppGpuLayers `
        -LlamaCppThreads $LlamaCppThreads `
        -LocalOnly:$LocalOnly.IsPresent `
        -AllowFallback:$AllowFallback.IsPresent `
        -WaitSeconds $WaitSeconds `
        -PythonPath $PythonPath `
        -DryRun:$DryRun.IsPresent `
        -Property $Property `
        -ExpandProperty $ExpandProperty `
        -AsJson:$AsJson.IsPresent `
        -JsonDepth $JsonDepth
}

Set-Alias -Name Start-HarnessLocalLLMBackend -Value Start-HarnessOwnLLMBackend
Set-Alias -Name Stop-HarnessOwnLLMBackend -Value Stop-HarnessModelBackend
Set-Alias -Name Stop-HarnessLocalLLMBackend -Value Stop-HarnessModelBackend
Set-Alias -Name Get-HarnessOwnLLMBackendStatus -Value Get-HarnessModelBackendStatus
Set-Alias -Name Get-HarnessLocalLLMBackendStatus -Value Get-HarnessModelBackendStatus

<#
.SYNOPSIS
Stops tracked harness local model backends.
.DESCRIPTION
Stops the harness-owned model provider process by host/port or every tracked provider with -All.
#>
function Stop-HarnessModelBackend {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [switch]$All,
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $targetKey = "model-backend|$ModelBackendHost|$ModelBackendPort"
    $rawSessions = _Read-HarnessModelBackendSessions
    $sessions = _Prune-StaleModelSessions -Sessions $rawSessions
    $remaining = @()
    $removed = @()
    $shouldStop = $DryRun.IsPresent -or $PSCmdlet.ShouldProcess("$ModelBackendHost`:$ModelBackendPort", "Stop harness local model backend")

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
        if ($DryRun -or -not $shouldStop) {
            if (-not $DryRun) {
                $remaining += $entry
            }
            continue
        }
        if ($entry.process_id -and (Get-Process -Id $entry.process_id -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $entry.process_id -Force -ErrorAction SilentlyContinue
        }
    }

    if ($DryRun -or -not $shouldStop) {
        $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count") -Property @{
            mode = "model_backend"
            action = if ($DryRun) { "stopped" } else { "whatif" }
            removed_count = $removed.Count
            removed = $removed
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    if ($removed.Count -gt 0) {
        _Write-HarnessModelBackendSessions -Sessions $remaining
    }

    $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count") -Property @{
        mode = "model_backend"
        action = "stopped"
        removed_count = $removed.Count
        removed = $removed
    }
    return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Gets harness local model backend status.
.DESCRIPTION
Reads /health and /v1/models from the local provider and returns flattened diagnostics plus nested payloads.
#>
function Get-HarnessModelBackendStatus {
    [CmdletBinding()]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [int]$RequestTimeoutSeconds = 6,
        [switch]$IncludeSession,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
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
        payload = $null
        model_entry = $null
    }
    try {
        $catalogResponse = Invoke-WebRequest -Uri $modelsUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
        $models.status_code = [int]$catalogResponse.StatusCode
        $models.reachable = $catalogResponse.StatusCode -ge 200 -and $catalogResponse.StatusCode -lt 300
        if ($models.reachable) {
            $catalogPayload = _Build-StatusPayload -Body $catalogResponse.Content
            $models.payload = $catalogPayload
            $catalogModels = @(_Normalize-ModelsPayload -CatalogPayload $catalogPayload)
            if ($null -ne $catalogModels) {
                $models.models = $catalogModels
            }
            $models.model_present = $catalogModels.Count -gt 0
            $models.model_entry = _Get-ModelCatalogEntry -CatalogPayload $catalogPayload
        } else {
            $models.error = "Model catalog request returned status $($catalogResponse.StatusCode)"
        }
    } catch {
        $models.error = "Failed to read /v1/models at $modelsUrl. $($_.Exception.Message)"
        if ($null -ne $_.Exception -and $_.Exception.PSObject.Properties.Name -contains "Response" -and $_.Exception.Response -and $_.Exception.Response.PSObject.Properties.Name -contains "StatusCode") {
            $models.status_code = [int]$_.Exception.Response.StatusCode
        }
    }

    $selectedModel = $null
    if ($models.models -and $models.models.Count -gt 0) {
        $selectedModel = [string]$models.models[0]
    }
    if ($null -ne $models.payload) {
        $models.model_entry = _Get-ModelCatalogEntry -CatalogPayload $models.payload -ModelName $selectedModel
    }
    $diagnostics = _Get-ProviderDiagnosticsFromPayloads -HealthPayload $health.payload -ModelEntry $models.model_entry
    $sessionEntriesForStatus = @(
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
    $primarySession = $null
    if ($sessionEntriesForStatus.Count -gt 0) {
        $primarySession = $sessionEntriesForStatus[0]
    }
    $result = [PSCustomObject]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        host = $ModelBackendHost
        port = $ModelBackendPort
        model = $selectedModel
        configured_backend = $diagnostics.configured_backend
        generation_backend = $diagnostics.generation_backend
        model_source = $diagnostics.model_source
        model_source_type = $diagnostics.model_source_type
        model_artifact_format = $diagnostics.model_artifact_format
        provider_store = $diagnostics.provider_store
        manifest_path = $diagnostics.manifest_path
        runtime_dependency = $diagnostics.runtime_dependency
        runtime_dependency_available = [bool]$diagnostics.runtime_dependency_available
        local_model_loaded = [bool]$diagnostics.local_model_loaded
        model_source_present = [bool]$diagnostics.model_source_present
        model_load_attempted = [bool]$diagnostics.model_load_attempted
        model_load_succeeded = [bool]$diagnostics.model_load_succeeded
        last_load_error = $diagnostics.last_load_error
        last_generation_error = $diagnostics.last_generation_error
        template_applied = [bool]$diagnostics.template_applied
        finish_reason = $diagnostics.finish_reason
        truncated = [bool]$diagnostics.truncated
        reasoning_extracted = [bool]$diagnostics.reasoning_extracted
        fallback_active = [bool]$diagnostics.fallback_active
        provider_warning = $diagnostics.provider_warning
        python_path = if ($null -ne $primarySession -and $primarySession.PSObject.Properties.Name -contains "python_path") { $primarySession.python_path } else { $null }
        stdout_log = if ($null -ne $primarySession -and $primarySession.PSObject.Properties.Name -contains "stdout_log") { $primarySession.stdout_log } else { $null }
        stderr_log = if ($null -ne $primarySession -and $primarySession.PSObject.Properties.Name -contains "stderr_log") { $primarySession.stderr_log } else { $null }
        health_reachable = [bool]$health.reachable
        health_status_code = $health.status_code
        health_error = $health.error
        models_reachable = [bool]$models.reachable
        models_status_code = $models.status_code
        models_error = $models.error
        model_catalog_present = [bool]$models.model_present
        health = $health
        models = $models
    }

    if ($IncludeSession) {
        $result | Add-Member -NotePropertyName session -NotePropertyValue $sessionEntriesForStatus
    }

    $result = _Add-HarnessTypeName `
        -InputObject $result `
        -TypeName "Harness.ModelBackend.Status" `
        -DefaultDisplayPropertySet @(
            "host",
            "port",
            "model",
            "generation_backend",
            "model_source_type",
            "model_artifact_format",
            "provider_store",
            "runtime_dependency_available",
            "model_source_present",
            "model_load_attempted",
            "model_load_succeeded",
            "template_applied",
            "truncated",
            "fallback_active",
            "health_reachable",
            "models_reachable",
            "model_catalog_present",
            "health_error",
            "models_error",
            "provider_warning"
        )
    return _Apply-HarnessOutputOptions `
        -InputObject $result `
        -Property $Property `
        -ExpandProperty $ExpandProperty `
        -AsJson:$AsJson.IsPresent `
        -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Stops the tracked runtime and model backend stack.
.DESCRIPTION
Coordinates Stop-HarnessBackend and Stop-HarnessModelBackend while preserving object-first output.
#>
function Stop-HarnessStack {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [ValidateSet("local", "containerized")]
        [string]$ExecutionMode = "local",
        [string]$Config = "harness.yaml",
        [string]$ServerHost = "127.0.0.1",
        [int]$Port = 8080,
        [string]$ContainerProfile = "",
        [string]$ComposeFile = "docker-compose.nvidia.yaml",
        [string]$EnvFile = "",
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [switch]$All,
        [switch]$SkipRuntimeBackend,
        [switch]$SkipModelBackend,
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    if ($SkipRuntimeBackend -and $SkipModelBackend) {
        throw "At least one backend must be selected. Remove -SkipRuntimeBackend or -SkipModelBackend."
    }

    $runtimeResult = $null
    $modelResult = $null
    $errors = @()

    if (-not $SkipRuntimeBackend) {
        try {
            $runtimeResult = Stop-HarnessBackend `
                -ExecutionMode $ExecutionMode `
                -Config $Config `
                -ServerHost $ServerHost `
                -Port $Port `
                -ContainerProfile $ContainerProfile `
                -ComposeFile $ComposeFile `
                -EnvFile $EnvFile `
                -All:$All.IsPresent `
                -DryRun:$DryRun.IsPresent
        } catch {
            $errors += [PSCustomObject]@{
                target = "runtime_backend"
                message = $_.Exception.Message
            }
        }
    }

    if (-not $SkipModelBackend) {
        try {
            $modelResult = Stop-HarnessModelBackend `
                -ModelBackendHost $ModelBackendHost `
                -ModelBackendPort $ModelBackendPort `
                -All:$All.IsPresent `
                -DryRun:$DryRun.IsPresent
        } catch {
            $errors += [PSCustomObject]@{
                target = "model_backend"
                message = $_.Exception.Message
            }
        }
    }

    $runtimeRemoved = if ($null -ne $runtimeResult -and $runtimeResult.PSObject.Properties.Name -contains "removed_count") {
        [int]$runtimeResult.removed_count
    } else {
        0
    }
    $modelRemoved = if ($null -ne $modelResult -and $modelResult.PSObject.Properties.Name -contains "removed_count") {
        [int]$modelResult.removed_count
    } else {
        0
    }

    $result = _New-HarnessObject `
        -TypeName "Harness.Stack.StopResult" `
        -DefaultDisplayPropertySet @(
            "action",
            "ok",
            "dry_run",
            "runtime_removed_count",
            "model_removed_count",
            "error_count"
        ) `
        -Property ([ordered]@{
            action = "stopped"
            ok = ($errors.Count -eq 0)
            dry_run = [bool]$DryRun.IsPresent
            all = [bool]$All.IsPresent
            runtime_backend_skipped = [bool]$SkipRuntimeBackend.IsPresent
            model_backend_skipped = [bool]$SkipModelBackend.IsPresent
            runtime_removed_count = $runtimeRemoved
            model_removed_count = $modelRemoved
            error_count = $errors.Count
            runtime_backend = $runtimeResult
            model_backend = $modelResult
            errors = $errors
        })

    return _Apply-HarnessOutputOptions `
        -InputObject $result `
        -Property $Property `
        -ExpandProperty $ExpandProperty `
        -AsJson:$AsJson.IsPresent `
        -JsonDepth $JsonDepth
}

Set-Alias -Name Stop-HarnessAll -Value Stop-HarnessStack
Set-Alias -Name Stop-HarnessEverything -Value Stop-HarnessStack

<#
.SYNOPSIS
Sends one prompt through the harness runtime or demo path.
.DESCRIPTION
Builds the runtime request, optionally starts a temporary server, and returns the OpenAI-compatible response with convenience fields.
#>
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
        [string]$PythonPath = "",
        [switch]$RequireEvidence,
        [switch]$EnableAdvancedRouter,
        [switch]$NoNetwork,
        [switch]$SkipBackendCheck,
        [switch]$UseExistingServer,
        [switch]$DryRun,
        [switch]$AnswerOnly,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $ErrorActionPreference = "Stop"
    $resolvedPythonPath = _Resolve-HarnessPythonPath -PythonPath $PythonPath

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

        $helperOutput = & $resolvedPythonPath -c $pythonScript 2>&1
        $helperOutputText = $helperOutput | Out-String
        if ($LASTEXITCODE -ne 0) {
            $normalized = $helperOutputText.Trim()
            if ($normalized -match "model_backend_unavailable|model backend is unavailable|is not available in catalog|Model backend is unavailable") {
                $backendCode = "model_backend_unavailable"
                if ($normalized -match "error_code=([a-z0-9_]+)") {
                    $backendCode = $matches[1]
                }
                $backendError = @"
Invoke-HarnessOneShot: runtime backend preflight failed with code $backendCode at the configured /v1/models endpoint.
Model backend check failed and runtime may not be reachable.
Start a real local model with `Start-HarnessOwnLLMBackend -Model "<model>" -Backend auto -ModelPath "<local Hugging Face folder or GGUF file>" -LocalOnly`,
or `Start-HarnessOwnLLMBackend -Model "qwen3:4b" -Backend auto -ModelsRoot "`$env:USERPROFILE\.ollama\models" -LocalOnly`,
or `Start-HarnessOwnLLMBackend -Model "hf://vendor/model" -Backend auto -LocalOnly` when the model is already in the local Hugging Face cache,
or explicitly start diagnostic stub mode with `Start-HarnessOwnLLMBackend -Model "<model>" -Backend fallback`,
or rerun with -Mode demo or -SkipBackendCheck (intended only for controlled diagnostics).
error_code=$backendCode. $normalized
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
            $result = _New-HarnessObject -TypeName "Harness.OneShot.DryRun" -DefaultDisplayPropertySet @("mode", "resolved_model", "backend_name", "backend_url", "health_url") -Property @{
                payload = $helper.payload
                resolved_model = $helper.resolved_model
                explicit_model = $helper.explicit_model
                health_url = $helper.health_url
                backend_name = $helper.backend_name
                backend_url = $helper.backend_url
                config_path = $helper.config_path
                mode = "runtime"
                python_path = $resolvedPythonPath
                runtime_env = $runtimeEnv
            }
            if ($AnswerOnly) {
                $result = ""
            }
            return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
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

                $serverProcess = Start-Process -FilePath $resolvedPythonPath -ArgumentList @(
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
            $result = $response.Content | ConvertFrom-Json -Depth 20
            $result = _Add-HarnessOneShotConvenience -Response $result
            if ($AnswerOnly) {
                $result = $result.answer
            }
            return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
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
        $result = & python @demoArgs
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    throw "Unsupported mode: $Mode"
}
