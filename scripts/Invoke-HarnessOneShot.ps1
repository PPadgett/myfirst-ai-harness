Set-StrictMode -Version Latest

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

    $pythonCandidates = @("python3", "python")
    foreach ($candidate in $pythonCandidates) {
        try {
            $command = Get-Command -Name $candidate -ErrorAction Stop
        } catch {
            continue
        }
        $versionOutput = & $candidate --version 2>&1
        $versionText = [string]($versionOutput | Select-Object -First 1)
        if ($versionText -match 'Python\s+([0-9]+)\.([0-9]+)') {
            if ([int]$matches[1] -ge 3) {
                return $command.Source
            }
        }
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

function _Parse-CommandLine {
    [CmdletBinding()]
    param(
        [string]$CommandLine
    )

    if ([string]::IsNullOrWhiteSpace($CommandLine)) {
        return @()
    }

    $tokens = [System.Management.Automation.PSParser]::Tokenize($CommandLine, [ref]$null)
    $allowedTypes = @(
        [System.Management.Automation.PSTokenType]::Command,
        [System.Management.Automation.PSTokenType]::CommandArgument,
        [System.Management.Automation.PSTokenType]::CommandParameter,
        [System.Management.Automation.PSTokenType]::String,
        [System.Management.Automation.PSTokenType]::Number,
        [System.Management.Automation.PSTokenType]::Identifier,
        [System.Management.Automation.PSTokenType]::Generic,
        [System.Management.Automation.PSTokenType]::Variable
    )

    $parts = @()
    foreach ($token in $tokens) {
        if ($allowedTypes -contains $token.Type) {
            $parts += [string]$token.Content
        }
    }

    return $parts
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

function _Normalize-HardwareDeviceSelection {
    [CmdletBinding()]
    param(
        [string[]]$Devices
    )

    $normalizedInput = @($Devices)
    if ($null -eq $normalizedInput -or $normalizedInput.Count -eq 0) {
        return @("cpu")
    }

    $tokens = @()
    $invalidTokens = @()

    foreach ($entry in $normalizedInput) {
        if ([string]::IsNullOrWhiteSpace($entry)) {
            continue
        }
        $parts = [regex]::Split($entry.Trim(), "[,;\|+]|(?<=\S)\s+(?=\S)")
        foreach ($part in $parts) {
            $value = [string]$part
            if ([string]::IsNullOrWhiteSpace($value)) {
                continue
            }
            $normalizedValue = $value.Trim().ToLowerInvariant()
            switch ($value.Trim().ToLowerInvariant()) {
                "all" {
                    $tokens += @("npu", "gpu", "cpu")
                }
                "hybrid" {
                    $tokens += @("hybrid")
                }
                "hybrid_pair" {
                    $tokens += @("npu", "gpu")
                }
                "cpu" { $tokens += "cpu" }
                "gpu" { $tokens += "gpu" }
                "npu" { $tokens += "npu" }
                default {
                    $invalidTokens += $normalizedValue
                    continue
                }
            }
        }
    }

    if ($invalidTokens.Count -gt 0) {
        $invalidText = @($invalidTokens | Select-Object -Unique) -join ", "
        throw "Unsupported hardware device token(s) in '$($normalizedInput -join ", ")'. Expected one of: cpu, gpu, npu, all, hybrid, hybrid_pair. Received: $invalidText"
    }

    $knownOrder = @("npu", "gpu", "hybrid", "cpu")
    $seen = @{}
    foreach ($token in $tokens) {
        $seen[$token] = $true
    }

    $ordered = @()
    foreach ($device in $knownOrder) {
        if ($seen.ContainsKey($device)) {
            $ordered += $device
        }
    }

    if ($ordered.Count -eq 0) {
        $ordered = @("cpu")
    }
    return @($ordered)
}

function _Get-RequiredModelBackendDevicesFromConfig {
    [CmdletBinding()]
    param(
        [string]$ConfigPath
    )

    $resolvedConfig = _Resolve-FilePath -Path $ConfigPath -Base (Get-Location).Path
    $pythonPath = _Resolve-HarnessPythonPath
    $pythonScript = @'
import os
import sys
import yaml

config_path = sys.argv[1]
if not config_path or str(config_path).lower() in ("", "harness.yaml"):
    config_path = "harness.yaml"
if not os.path.exists(config_path):
    print("")
    raise SystemExit(0)

with open(config_path, "r", encoding="utf-8") as f:
    loaded = yaml.safe_load(f) or {}

backend_entries = loaded.get("backends")
if not isinstance(backend_entries, list) or not backend_entries:
    single = loaded.get("backend")
    if isinstance(single, dict):
        backend_entries = [single]
    else:
        backend_entries = []

seen = []
for backend in backend_entries:
    if not isinstance(backend, dict):
        continue
    if bool(backend.get("enabled", True)) is False:
        continue
    if bool(backend.get("required", True)) is False:
        continue
    device = str(backend.get("device", "") or "").strip().lower()
    if not device:
        device = str(backend.get("id", "") or "").strip().lower()
    if device and device not in seen:
        seen.append(device)
print(",".join(seen))
'@
    $raw = & $pythonPath -c $pythonScript $resolvedConfig
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read required backend devices from config: $resolvedConfig"
    }
    if ([string]::IsNullOrWhiteSpace([string]$raw)) {
        return @()
    }
    return @(
        foreach ($entry in ($raw -split ",")) {
            $value = [string]$entry
            if (-not [string]::IsNullOrWhiteSpace($value)) {
                $value
            }
        }
    )
}

function _Get-ModelBackendProfilesFromConfig {
    [CmdletBinding()]
    param(
        [string]$ConfigPath
    )

    $resolvedConfig = _Resolve-FilePath -Path $ConfigPath -Base (Get-Location).Path
    $pythonPath = _Resolve-HarnessPythonPath
    $pythonScript = @'
import os
import json
import sys
import yaml

config_path = sys.argv[1]
if not config_path or str(config_path).lower() in ("", "harness.yaml"):
    config_path = "harness.yaml"
if not os.path.exists(config_path):
    print("{}")
    raise SystemExit(0)

with open(config_path, "r", encoding="utf-8") as f:
    loaded = yaml.safe_load(f) or {}

backend_entries = loaded.get("backends")
if not isinstance(backend_entries, list) or not backend_entries:
    single = loaded.get("backend")
    if isinstance(single, dict):
        backend_entries = [single]
    else:
        backend_entries = []

result = {}
for backend in backend_entries:
    if not isinstance(backend, dict):
        continue
    if bool(backend.get("enabled", True)) is False:
        continue
    device = str(backend.get("device", "") or "").strip().lower()
    if not device:
        continue
    if device in result:
        continue
    result[device] = {
        "device": device,
        "backend_id": str(backend.get("id", "") or "").strip(),
        "runtime": str(backend.get("runtime", "") or "").strip().lower(),
        "base_url": str(backend.get("base_url", "") or "").strip(),
        "health_endpoint": str(backend.get("health_endpoint", "") or "").strip(),
        "model": str(backend.get("model", "") or "").strip(),
        "launch_mode": str(backend.get("launch_mode", "") or "").strip().lower(),
        "start_command": str(backend.get("start_command", "") or "").strip(),
        "start_working_directory": str(backend.get("start_working_directory", "") or "").strip(),
        "start_args": [
            str(arg).strip()
            for arg in list(backend.get("start_args", []) or [])
            if str(arg).strip()
        ],
        "required": bool(backend.get("required", True)),
        "enabled": bool(backend.get("enabled", True)),
    }
print(json.dumps(result, ensure_ascii=False))
'@
    $raw = & $pythonPath -c $pythonScript $resolvedConfig
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read backend profiles from config: $resolvedConfig"
    }
    if ([string]::IsNullOrWhiteSpace([string]$raw)) {
        return @{}
    }

    $decoded = ConvertFrom-Json -InputObject $raw -ErrorAction Stop
    $profiles = @{}
    foreach ($name in $decoded.PSObject.Properties.Name) {
        $entry = $decoded.$name
        if ($null -eq $entry) {
            continue
        }
        $profiles[([string]$name).ToLowerInvariant()] = @{
            device = if ($entry.PSObject.Properties.Name -contains "device") { [string]$entry.device } else { ([string]$name).ToLowerInvariant() }
            backend_id = if ($entry.PSObject.Properties.Name -contains "backend_id") { [string]$entry.backend_id } else { "" }
            runtime = if ($entry.PSObject.Properties.Name -contains "runtime") { [string]$entry.runtime } else { "" }
            base_url = if ($entry.PSObject.Properties.Name -contains "base_url") { [string]$entry.base_url } else { "" }
            health_endpoint = if ($entry.PSObject.Properties.Name -contains "health_endpoint") { [string]$entry.health_endpoint } else { "" }
            model = if ($entry.PSObject.Properties.Name -contains "model") { [string]$entry.model } else { "" }
            launch_mode = if ($entry.PSObject.Properties.Name -contains "launch_mode") { [string]$entry.launch_mode } else { "" }
            start_command = if ($entry.PSObject.Properties.Name -contains "start_command") { [string]$entry.start_command } else { "" }
            start_working_directory = if ($entry.PSObject.Properties.Name -contains "start_working_directory") {
                [string]$entry.start_working_directory
            } else {
                ""
            }
            start_args = if ($entry.PSObject.Properties.Name -contains "start_args") {
                @($entry.start_args | ForEach-Object { [string]$_ })
            } else {
                @()
            }
            required = if ($entry.PSObject.Properties.Name -contains "required") { [bool]$entry.required } else { $true }
        }
    }
    return $profiles
}

function _Normalize-BackendEndpointUrl {
    [CmdletBinding()]
    param(
        [string]$BaseHost,
        [int]$BasePort,
        [string]$Candidate,
        [string]$Fallback
    )

    if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
        $candidateValue = $Candidate.Trim()
        if ($candidateValue -match "^(https?://)") {
            return $candidateValue
        }
        if ($candidateValue.StartsWith("/")) {
            return "http://{0}:{1}{2}" -f $BaseHost, $BasePort, $candidateValue
        }
        if ($candidateValue -match "^\d+$") {
            return "http://{0}:{1}/{2}" -f $BaseHost, $BasePort, $candidateValue
        }
        return "http://{0}:{1}/{2}" -f $BaseHost, $BasePort, $candidateValue.TrimStart("/")
    }
    if (-not [string]::IsNullOrWhiteSpace($Fallback)) {
        return $Fallback
    }
    return _HealthUrl -TargetHost $BaseHost -Port $BasePort
}

function _Resolve-BackendCatalogUrl {
    [CmdletBinding()]
    param(
        [string]$BackendBaseUrl,
        [string]$FallbackCatalogUrl,
        [string]$BaseHost = "127.0.0.1",
        [int]$BasePort = 11433
    )

    if ([string]::IsNullOrWhiteSpace($BackendBaseUrl)) {
        return $FallbackCatalogUrl
    }
    $backend = $BackendBaseUrl.Trim().TrimEnd("/")
    if ([string]::IsNullOrWhiteSpace($backend)) {
        return $FallbackCatalogUrl
    }
    if ($backend -match "^(https?://)") {
        return _Build-ModelCatalogUrl -BackendBaseUrl $backend
    }
    if ($backend -match "^\d+$") {
        return "http://{0}:{1}/v1/models" -f $BaseHost, $BasePort
    }
    return _Build-ModelCatalogUrl -BackendBaseUrl "http://$backend"
}

function _Default-CompatibleModelCatalog {
    return @(
        [ordered]@{
            model_id = "llama-3.2-1b"
            family = "llama"
            supports = @("cpu", "gpu", "npu")
            default_runtime = @("fallback", "transformers")
            notes = "Starter family for cross-stack CPU+GPU+NPU experiments."
        },
        [ordered]@{
            model_id = "llama-3.2-3b"
            family = "llama"
            supports = @("cpu", "gpu", "npu")
            default_runtime = @("transformers", "llamacpp")
            notes = "Mid-size model commonly used for hybrid local runtime evaluation."
        },
        [ordered]@{
            model_id = "mistral-7b-instruct"
            family = "mistral"
            supports = @("cpu", "gpu", "npu")
            default_runtime = @("transformers", "llamacpp")
            notes = "Instruction model for multi-device baseline performance testing."
        },
        [ordered]@{
            model_id = "phi-4-mini"
            family = "phi"
            supports = @("cpu", "gpu", "npu")
            default_runtime = @("transformers", "fallback")
            notes = "Very compact stack-friendly family for latency-sensitive comparisons."
        },
        [ordered]@{
            model_id = "qwen2.5-3b-instruct"
            family = "qwen"
            supports = @("cpu", "gpu", "npu")
            default_runtime = @("transformers", "llamacpp")
            notes = "Qwen family candidate for short+batch mixed-route benchmarking."
            source = "hf://Qwen/Qwen2.5-3B-Instruct"
        },
        [ordered]@{
            model_id = "qwen2.5-7b-instruct"
            family = "qwen"
            supports = @("cpu", "gpu", "npu")
            default_runtime = @("transformers", "llamacpp")
            notes = "Larger Qwen target where CPU is usually fallback-only."
            source = "hf://Qwen/Qwen2.5-7B-Instruct"
        },
        [ordered]@{
            model_id = "qwen3:4b"
            family = "qwen"
            supports = @("cpu", "gpu", "npu")
            default_runtime = @("transformers", "llamacpp")
            notes = "Default repository example model for quick compatibility checks."
        }
    )
}

function _Extract-DeviceSupport {
    param(
        [object]$Entry
    )

    if ($null -eq $Entry) {
        return @()
    }
    if ($Entry -is [hashtable]) {
        $supports = $Entry["supports"]
        if ($supports -is [System.Array]) {
            return @($supports)
        }
        if ($supports -is [System.Collections.Generic.List[string]]) {
            return @($supports)
        }
    } elseif ($Entry -is [System.Collections.IDictionary]) {
        $supports = $Entry["supports"]
        if ($supports -is [System.Array]) {
            return @($supports)
        }
        if ($supports -is [System.Collections.Generic.List[string]]) {
            return @($supports)
        }
    }
    if ($Entry.PSObject -and $Entry.PSObject.Properties.Name -contains "supports") {
        $supports = $Entry.supports
        if ($supports -is [System.Array]) {
            return @($supports)
        }
    }
    if ($Entry.PSObject -and $Entry.PSObject.Properties.Name -contains "supported_devices") {
        $supports = $Entry.supported_devices
        if ($supports -is [System.Array]) {
            return @($supports)
        }
    }
    return @()
}

function _Get-CompatibleModelField {
    param(
        [object]$Entry,
        [string]$Field
    )

    if ($Entry -is [hashtable]) {
        if ($Entry.ContainsKey($Field)) {
            return $Entry[$Field]
        }
        return $null
    }
    if ($Entry -is [System.Collections.IDictionary]) {
        if ($Entry.Contains($Field)) {
            return $Entry[$Field]
        }
        return $null
    }
    if ($Entry.PSObject -and $Entry.PSObject.Properties.Name -contains $Field) {
        return $Entry.$Field
    }
    return $null
}

function _Normalize-CompatibleModelEntry {
    param([object]$Entry)

    if ($null -eq $Entry) {
        return [PSCustomObject]@{}
    }
    if ($Entry -isnot [hashtable] -and $Entry -isnot [System.Collections.IDictionary] -and $Entry -isnot [PSObject]) {
        return [PSCustomObject]@{}
    }

    $modelId = ""
    $rawModelId = _Get-CompatibleModelField -Entry $Entry -Field "model_id"
    if ([string]::IsNullOrWhiteSpace([string]$rawModelId)) {
        $rawModelId = _Get-CompatibleModelField -Entry $Entry -Field "model"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$rawModelId)) {
        $modelId = [string]$rawModelId
    }
    if ([string]::IsNullOrWhiteSpace($modelId)) {
        return [PSCustomObject]@{}
    }

    $rawFamily = _Get-CompatibleModelField -Entry $Entry -Field "family"
    $rawDefaultRuntime = _Get-CompatibleModelField -Entry $Entry -Field "default_runtime"
    $rawNotes = _Get-CompatibleModelField -Entry $Entry -Field "notes"
    $rawSource = _Get-CompatibleModelField -Entry $Entry -Field "source"

    return [PSCustomObject]@{
        model_id = $modelId
        family = if ($null -ne $rawFamily) {
            [string]$rawFamily
        } else {
            ""
        }
        supports = @(_Extract-DeviceSupport -Entry $Entry)
        default_runtime = if ($null -ne $rawDefaultRuntime) {
            if ($rawDefaultRuntime -is [System.Array]) {
                @($rawDefaultRuntime)
            } elseif ($rawDefaultRuntime -is [System.Collections.Generic.List[string]]) {
                @($rawDefaultRuntime)
            } else {
                @([string]$rawDefaultRuntime)
            }
        } else {
            @()
        }
        notes = if ($null -ne $rawNotes) {
            [string]$rawNotes
        } else {
            ""
        }
        source = if ($null -ne $rawSource) {
            [string]$rawSource
        } else {
            ""
        }
    }
}

function _Load-CompatibleModelCatalog {
    param([string]$CatalogPath = "")

    if ([string]::IsNullOrWhiteSpace($CatalogPath)) {
        return @(
            foreach ($entry in (_Default-CompatibleModelCatalog)) {
                _Normalize-CompatibleModelEntry -Entry $entry
            }
        )
    }
    if (-not (Test-Path -LiteralPath $CatalogPath -PathType Leaf)) {
        throw "Compatibility catalog not found at: $CatalogPath"
    }

    $payload = Get-Content -LiteralPath $CatalogPath -Raw -ErrorAction Stop
    $decoded = ConvertFrom-Json -InputObject $payload -ErrorAction Stop
    if ($decoded -is [System.Array]) {
        return @($decoded | ForEach-Object { _Normalize-CompatibleModelEntry -Entry $_ })
    }
    if ($decoded.PSObject -and $decoded.PSObject.Properties.Name -contains "models") {
        return @($decoded.models | ForEach-Object { _Normalize-CompatibleModelEntry -Entry $_ })
    }
    throw "Compatibility catalog format not recognized at: $CatalogPath"
}

function _Resolve-CompatibleModelDownloadId {
    [CmdletBinding()]
    param(
        [string]$ModelId,
        [string]$CatalogSource = ""
    )

    $normalizedModel = [string]$ModelId
    if ([string]::IsNullOrWhiteSpace($normalizedModel)) {
        return ""
    }
    $normalizedModel = $normalizedModel.Trim()

    if ($normalizedModel -match "(?i)^hf://") {
        return $normalizedModel.Substring(5)
    }

    $catalogHint = [string]$CatalogSource
    if (-not [string]::IsNullOrWhiteSpace($catalogHint)) {
        $catalogHint = $catalogHint.Trim()
        if ($catalogHint -match "(?i)^hf://") {
            return $catalogHint.Substring(5)
        }
        if ($catalogHint -match ".+/.+") {
            return $catalogHint
        }
    }

    if ($normalizedModel -match ".+/.+") {
        return $normalizedModel
    }
    if ($normalizedModel -match ":[^/]+$") {
        return ""
    }

    $aliasMap = @{
        "qwen2.5-7b-instruct" = "Qwen/Qwen2.5-7B-Instruct"
        "qwen2.5-3b-instruct" = "Qwen/Qwen2.5-3B-Instruct"
        "llama-3.2-1b" = "meta-llama/Llama-3.2-1B"
        "llama-3.2-3b" = "meta-llama/Llama-3.2-3B"
    }

    $mapped = $aliasMap[$normalizedModel.ToLowerInvariant()]
    if (-not [string]::IsNullOrWhiteSpace($mapped)) {
        return $mapped
    }

    return $normalizedModel
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
        [int]$ModelBackendPort,
        [string]$TargetDevice = ""
    )

    $key = "model-backend|$ModelBackendHost|$ModelBackendPort"
    $normalizedTargetDevice = [string]$TargetDevice.ToLowerInvariant()
    foreach ($entry in $Sessions) {
        if ($entry.mode -ne "model_backend" -or $entry.key -ne $key) {
            continue
        }
        if (-not [string]::IsNullOrWhiteSpace($normalizedTargetDevice)) {
            $entryDevice = if ($entry.PSObject.Properties.Name -contains "requested_device") {
                [string]$entry.requested_device
            } else {
                ""
            }
            if (-not [string]::IsNullOrWhiteSpace($entryDevice) -and $entryDevice.ToLowerInvariant() -ne $normalizedTargetDevice) {
                continue
            }
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

function _Normalize-ModelId {
    param([string]$Model)

    if ([string]::IsNullOrWhiteSpace($Model)) {
        return ""
    }

    return ([string]$Model).Trim().ToLowerInvariant().Replace(" ", "")
}

function _Get-ActiveRuntimeModel {
    param(
        [string]$ServerHost,
        [int]$Port,
        [int]$TimeoutSeconds = 2
    )

    $healthUrl = _HealthUrl -TargetHost $ServerHost -Port $Port
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -Method Get -TimeoutSec $TimeoutSeconds -ErrorAction Stop
        if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) {
            return $null
        }
        $healthPayload = _Build-StatusPayload -Body $response.Content
        $inlineModel = _Get-HarnessNamedProperty -InputObject $healthPayload -Names @("model", "resolved_model")
        if (-not [string]::IsNullOrWhiteSpace([string]$inlineModel)) {
            return [string]$inlineModel
        }
        return $null
    } catch {
        return $null
    }
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
        return ($Body | ConvertFrom-HarnessJson -ErrorAction Stop -Depth 20)
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
            $health = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -Method Get -TimeoutSec 2 -ErrorAction Stop
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
    try {
        $backendContext = _Load-BackendContext -ConfigPath $resolvedConfig
    } catch {
        $backendContext = $null
    }
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
            $healthResponse = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
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
            $catalogResponse = Invoke-WebRequest -UseBasicParsing -Uri $backendModelsUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
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
                if ($entry.PSObject.Properties.Name -contains "process_id" -and $entry.process_id) {
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
        [string]$RuntimeModel = "",
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
            $reuseRunningSession = $true
            if (-not [string]::IsNullOrWhiteSpace($RuntimeModel)) {
                $requestedModel = _Normalize-ModelId -Model $RuntimeModel
                $runningModel = _Normalize-ModelId -Model (_Get-ActiveRuntimeModel -ServerHost $ServerHost -Port $Port)
                if ($runningModel -ne $requestedModel) {
                    $reuseRunningSession = $false
                    Write-Verbose (
                        "Restarting runtime session on ${ServerHost}:$Port because requested model " +
                        "'$RuntimeModel' does not match running model '$runningModel'."
                    )
                    if ($runningSession.process_id) {
                        Stop-Process -Id $runningSession.process_id -Force -ErrorAction SilentlyContinue
                        Wait-Process -Id $runningSession.process_id -Timeout 3 -ErrorAction SilentlyContinue
                    }
                    $existingSessions = @($existingSessions | Where-Object { $_.key -ne $entryKey })
                    _Write-HarnessBackendSessions -Sessions $existingSessions
                    Start-Sleep -Milliseconds 250
                }
            }

            if ($reuseRunningSession) {
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
        $runtimeModelEnvSet = $false
        $runtimeModelEnvRestore = $null
        if (-not [string]::IsNullOrWhiteSpace($RuntimeModel)) {
            if (Test-Path Env:HARNESS_MODEL) {
                $runtimeModelEnvRestore = [string]$env:HARNESS_MODEL
            }
            $env:HARNESS_MODEL = $RuntimeModel
            $runtimeModelEnvSet = $true
        }
        try {
            $process = Start-Process -FilePath $resolvedPythonPath -ArgumentList $startArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        } catch {
            throw "Failed to start harness server process."
        } finally {
            if ($runtimeModelEnvSet) {
                if ($null -eq $runtimeModelEnvRestore) {
                    Remove-Item Env:HARNESS_MODEL -ErrorAction SilentlyContinue
                } else {
                    $env:HARNESS_MODEL = $runtimeModelEnvRestore
                }
            }
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
        [switch]$AutoResolveContainerProfile,
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
    $backendContext = $null
    try {
        $backendContext = _Load-BackendContext -ConfigPath $resolvedConfig
    } catch {
        Write-Verbose "Could not load backend context from $resolvedConfig during containerized stop: $($_.Exception.Message)"
        $backendContext = $null
    }
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
    $containerSessions = @(
        foreach ($entry in $sessions) {
            if ($entry.mode -eq "containerized" -and $entry.config -eq $resolvedConfig) {
                $entry
            }
        }
    )
    $knownProfiles = @(
        foreach ($entry in $containerSessions) {
            if ($entry.PSObject.Properties.Name -contains "profile" -and -not [string]::IsNullOrWhiteSpace([string]$entry.profile)) {
                [string]$entry.profile
            }
        }
    ) | Select-Object -Unique

    $knownProfiles = @($knownProfiles)
    if ([string]::IsNullOrWhiteSpace($profile)) {
        if ($AutoResolveContainerProfile) {
            if (@($knownProfiles).Count -eq 1) {
                $profile = [string]$knownProfiles[0]
            } elseif (@($knownProfiles).Count -gt 1) {
                throw "Could not auto-resolve container profile for '$resolvedConfig'. Multiple active profiles found: $($knownProfiles -join ', '). Specify -ContainerProfile explicitly."
            } else {
                if ($null -ne $backendContext -and $backendContext.backend_name) {
                    $profile = _Resolve-ContainerProfile -BackendName [string]$backendContext.backend_name
                }
            }
        } else {
            $candidateText = if (@($knownProfiles).Count -gt 0) {
                $knownProfiles -join ", "
            } else {
                "none"
            }
            throw "Container profile not provided. Use -ContainerProfile explicitly (found profiles: $candidateText). If there is exactly one known profile, re-run with -AutoResolveContainerProfile."
        }
    }
    if ([string]::IsNullOrWhiteSpace($profile)) {
        if ($null -ne $backendContext -and [string]$backendContext.backend_name -eq "ollama") {
            throw "Could not resolve container profile for backend 'ollama'. Use -ContainerProfile ollama explicitly."
        }
        $candidateText = if (@($knownProfiles).Count -gt 0) {
            $knownProfiles -join ", "
        } else {
            "none"
        }
        if (@($knownProfiles).Count -gt 0) {
            throw "Could not resolve container profile. Available profiles from session state: $candidateText. Use -ContainerProfile <profile> explicitly."
        }
        throw "Could not resolve container profile. Use -ContainerProfile explicitly with one of: nvidia or ollama. "
    }

    $composeArgs = @(
        "compose",
        "--env-file",
        $(if ([string]::IsNullOrWhiteSpace($EnvFile)) {
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
    $matchedSessions = @(
        foreach ($entry in $containerSessions) {
            $entryProfile = if ($entry.PSObject.Properties.Name -contains "profile") { [string]$entry.profile } else { "" }
            if ([string]::IsNullOrWhiteSpace($entryProfile) -or $entryProfile -eq $profile) {
                $entry
            }
        }
    )
    $remaining = @(
        foreach ($entry in $sessions) {
            if ($entry.mode -ne "containerized" -or $entry.config -ne $resolvedConfig) {
                $entry
                continue
            }
            $entryProfile = if ($entry.PSObject.Properties.Name -contains "profile") { [string]$entry.profile } else { "" }
            if ([string]::IsNullOrWhiteSpace($entryProfile) -or $entryProfile -eq $profile) {
                continue
            }
            $entry
        }
    )

    $removed = @(
        foreach ($entry in $matchedSessions) {
            $entryCopy = @{}
            foreach ($name in $entry.PSObject.Properties.Name) {
                $entryCopy[$name] = $entry.$name
            }
            $entryCopy["result_reason"] = "would_stop"
            $entryCopy["result_backend"] = "containerized"
            [pscustomobject]$entryCopy
        }
    )
    if ($removed.Count -eq 0) {
        $removed = @(
            [pscustomobject]@{
                mode = "containerized"
                config = $resolvedConfig
                profile = $profile
                host = $ServerHost
                port = $Port
                result_reason = "not_tracked"
            }
        )
    }

    if ($DryRun) {
        $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count", "profile", "config") -Property @{
            mode = "containerized"
            action = "stopped"
            removed_count = $removed.Count
            stopped_count = @($removed | Where-Object { $_.result_reason -in @("stopped", "already_stopped") }).Count
            command = $command
            profile = $profile
            config = $resolvedConfig
            compose_file = $resolvedCompose
            env_file = $resolvedEnvFile
            removed = $removed
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    if (-not $PSCmdlet.ShouldProcess($resolvedConfig, "Stop harness container backend")) {
        $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count", "profile", "config") -Property @{
            mode = "containerized"
            action = "whatif"
            removed_count = $removed.Count
            stopped_count = 0
            command = $command
            profile = $profile
            config = $resolvedConfig
            compose_file = $resolvedCompose
            env_file = $resolvedEnvFile
            removed = $removed
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    $compose = Start-Process -FilePath "docker" -ArgumentList $composeArgs -PassThru -NoNewWindow -Wait
    if ($compose.ExitCode -ne 0) {
        $removed = @(
            foreach ($entry in $removed) {
                $entryCopy = @{}
                foreach ($name in $entry.PSObject.Properties.Name) {
                    $entryCopy[$name] = $entry.$name
                }
                $entryCopy["result_reason"] = "timed_out"
                [pscustomobject]$entryCopy
            }
        )
        throw "docker compose stop failed with exit code $($compose.ExitCode)"
    }

    if ($removed.Count -gt 0) {
        $removed = @(
            foreach ($entry in $removed) {
                $entryCopy = @{}
                foreach ($name in $entry.PSObject.Properties.Name) {
                    $entryCopy[$name] = $entry.$name
                }
                if ([string]$entryCopy.result_reason -ne "not_tracked") {
                    $entryCopy["result_reason"] = "stopped"
                }
                [pscustomobject]$entryCopy
            }
        )
        _Write-HarnessBackendSessions -Sessions $remaining
    }
    $result = _New-HarnessObject -TypeName "Harness.Backend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count", "profile", "config") -Property @{
        mode = "containerized"
        action = "stopped"
        removed_count = $removed.Count
        stopped_count = @($removed | Where-Object { $_.result_reason -in @("stopped", "already_stopped") }).Count
        command = $command
        profile = $profile
        config = $resolvedConfig
        compose_file = $resolvedCompose
        env_file = $resolvedEnvFile
        removed = $removed
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
    if (-not $DryRun -and $modelPathProvided -and -not $sourceResolved -and -not $fallbackAllowed) {
        throw "Start-HarnessModelBackend: -ModelPath does not exist or could not be resolved: $($sourceResolution.source)"
    }
    if (-not $DryRun -and $Backend -eq "auto" -and -not $fallbackAllowed -and -not $primaryModelCanResolveLocally) {
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
        $requestedModel = _Normalize-ModelId -Model $Model
        $runningModel = if ($runningSession.PSObject.Properties.Name -contains "model") {
            _Normalize-ModelId -Model ([string]$runningSession.model)
        } else {
            ""
        }
        $reuseRunningSession = [bool]($runningModel -eq $requestedModel)
        if (-not $reuseRunningSession) {
            Write-Verbose (
                "Restarting model backend on ${ModelBackendHost}:$ModelBackendPort because requested model " +
                "'$Model' does not match running model '$runningModel'."
            )
            if ($runningSession.process_id) {
                Stop-Process -Id $runningSession.process_id -Force -ErrorAction SilentlyContinue
                Wait-Process -Id $runningSession.process_id -Timeout 3 -ErrorAction SilentlyContinue
            }
            $existingSessions = @($existingSessions | Where-Object { $_.key -ne $entryKey })
            _Write-HarnessModelBackendSessions -Sessions $existingSessions
            Start-Sleep -Milliseconds 250
        }

        if ($reuseRunningSession) {
            $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StartResult" -DefaultDisplayPropertySet @("mode", "action", "started", "host", "port", "model", "generation_backend", "fallback_active", "provider_warning", "process_id", "health_url") -Property @{
                mode = "model_backend"
                started = $false
                action = "already_running"
                host = $ModelBackendHost
                port = $ModelBackendPort
                model = $Model
                requested_device = $Device
                requested_backend = $Backend
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
            requested_device = $Device
            requested_backend = $Backend
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
            requested_device = $Device
            requested_backend = $Backend
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
        requested_device = $Device
        requested_backend = $Backend
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
        requested_device = $Device
        requested_backend = $Backend
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
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium", DefaultParameterSetName = "single")]
    param(
        [Parameter(ParameterSetName = "single")]
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [Parameter(ParameterSetName = "single")]
        [Parameter(ParameterSetName = "fleet")]
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [Parameter(ParameterSetName = "single")]
        [Parameter(ParameterSetName = "fleet")]
        [string]$Model = $Script:HarnessModelBackendDefaultModel,
        [Parameter(ParameterSetName = "single")]
        [Parameter(ParameterSetName = "fleet")]
        [string]$ModelPath = "",
        [Parameter(ParameterSetName = "single")]
        [Parameter(ParameterSetName = "fleet")]
        [string]$ModelsRoot = "",
        [Parameter(ParameterSetName = "single")]
        [Parameter(ParameterSetName = "fleet")]
        [string[]]$ExtraModel,
        [string]$Backend = "auto",
        [Parameter(ParameterSetName = "single")]
        [string]$Device = "cpu",
        [Parameter(ParameterSetName = "fleet")]
        [Alias("ModelBackendDevices")]
        [string[]]$Devices = @("npu", "gpu", "cpu"),
        [Parameter(ParameterSetName = "fleet")]
        [hashtable]$DevicePortMap = @{ npu = 11433; gpu = 11434; hybrid = 13305; cpu = 11435 },
        [Parameter(ParameterSetName = "fleet")]
        [hashtable]$ModelByDevice = @{},
        [Parameter(ParameterSetName = "fleet")]
        [hashtable]$ModelPathByDevice = @{},
        [Parameter(ParameterSetName = "fleet")]
        [hashtable]$ModelsRootByDevice = @{},
        [Parameter(ParameterSetName = "fleet")]
        [hashtable]$ExtraModelByDevice = @{},
        [Parameter(ParameterSetName = "fleet")]
        [hashtable]$DeviceBackendProfiles = @{},
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

    if ($PSCmdlet.ParameterSetName -eq "fleet") {
        if ($PSBoundParameters.ContainsKey("ModelBackendPort")) {
            $normalizedDeviceList = @(_Normalize-HardwareDeviceSelection -Devices $Devices)
            if ($normalizedDeviceList.Count -ne 1) {
                throw "When passing -ModelBackendPort with -Devices, provide exactly one device target."
            }
            $ModelPortDevice = $normalizedDeviceList[0]
            if ($PSBoundParameters.ContainsKey("DevicePortMap")) {
                $nextMap = @{} + $DevicePortMap
                if (-not $nextMap.ContainsKey($ModelPortDevice)) {
                    $nextMap[$ModelPortDevice] = $ModelBackendPort
                } elseif ([int]$nextMap[$ModelPortDevice] -ne $ModelBackendPort) {
                    throw "DevicePortMap[$ModelPortDevice] already maps to a different port than -ModelBackendPort."
                }
                $DevicePortMap = $nextMap
            } else {
                $DevicePortMap = @{ $ModelPortDevice = $ModelBackendPort }
            }
        }
        return Start-HarnessModelBackendFleet `
            -ModelBackendHost $ModelBackendHost `
            -Devices $Devices `
            -DevicePortMap $DevicePortMap `
            -Model $Model `
            -ModelByDevice $ModelByDevice `
            -ModelPath $ModelPath `
            -ModelPathByDevice $ModelPathByDevice `
            -ModelsRoot $ModelsRoot `
            -ModelsRootByDevice $ModelsRootByDevice `
            -ExtraModel $ExtraModel `
            -ExtraModelByDevice $ExtraModelByDevice `
            -Backend $Backend `
            -MaxNewTokens $MaxNewTokens `
            -LlamaCppContext $LlamaCppContext `
            -LlamaCppGpuLayers $LlamaCppGpuLayers `
            -LlamaCppThreads $LlamaCppThreads `
            -LocalOnly:$LocalOnly.IsPresent `
            -AllowFallback:$AllowFallback.IsPresent `
            -WaitSeconds $WaitSeconds `
            -DeviceBackendProfiles $DeviceBackendProfiles `
            -PythonPath $PythonPath `
            -DryRun:$DryRun.IsPresent `
            -Property $Property `
            -ExpandProperty $ExpandProperty `
            -AsJson:$AsJson.IsPresent `
            -JsonDepth $JsonDepth
    }

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
Returns a compatibility-aware model list and optional device filter.
#>
function Get-HarnessCompatibleModels {
    [CmdletBinding()]
    param(
        [string[]]$Devices = @("cpu", "gpu", "npu"),
        [ValidateSet("all", "any")]
        [string]$MatchMode = "any",
        [string]$CatalogPath = "",
        [string]$Filter = "",
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $requested = @(_Normalize-HardwareDeviceSelection -Devices $Devices)
    $catalog = _Load-CompatibleModelCatalog -CatalogPath $CatalogPath
    $query = [string]$Filter
    $results = @()

    foreach ($entry in $catalog) {
        $normalizedEntry = _Normalize-CompatibleModelEntry -Entry $entry
        $entryModelId = [string]$normalizedEntry.model_id
        if ([string]::IsNullOrWhiteSpace($entryModelId)) {
            continue
        }
        $entryFamily = [string]$normalizedEntry.family
        $entrySupports = $normalizedEntry.supports
        $entryDefaultRuntime = @($normalizedEntry.default_runtime)
        $entryNotes = [string]$normalizedEntry.notes
        $entrySource = [string]$normalizedEntry.source
        $modelId = $entryModelId
        if (-not [string]::IsNullOrWhiteSpace($query)) {
            if ($modelId.ToLowerInvariant() -notmatch [regex]::Escape($query.ToLowerInvariant())) {
                continue
            }
        }
        $supports = @($entrySupports | ForEach-Object { [string]$_.ToLowerInvariant() })
        $supportsSet = @{}
        foreach ($support in $supports) {
            $supportsSet[$support] = $true
        }
    if ($MatchMode -eq "all") {
            $missing = $false
            foreach ($candidate in $requested) {
                if (-not $supportsSet.ContainsKey($candidate)) {
                    $missing = $true
                    break
                }
            }
            if ($missing) {
                continue
            }
        } else {
            $matched = $false
            foreach ($candidate in $requested) {
                if ($supportsSet.ContainsKey($candidate)) {
                    $matched = $true
                    break
                }
            }
            if (-not $matched) {
                continue
            }
        }

        $results += [pscustomobject][ordered]@{
            model_id = $modelId
            model = $modelId
            family = $entryFamily
            supports = $supports
            default_runtime = @($entryDefaultRuntime)
            notes = $entryNotes
            source = $entrySource
            supports_requested = if ($supportsSet.Count -gt 0) {
                [bool]($supportsSet.Keys | Where-Object { $requested -contains $_ } )
            } else {
                $false
            }
            requested_devices = @($requested)
            match_mode = $MatchMode
        }
    }

    $sortedResults = @($results | Sort-Object -Property model_id)
    return _Apply-HarnessOutputOptions -InputObject $sortedResults -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

function _Get-HarnessRyzenAiCollectionDefinitions {
    [CmdletBinding()]
    param()

    return [ordered]@{
        hybrid = [ordered]@{
            key = "hybrid"
            slug = "ryzen-ai-171-hybrid"
            title = "Ryzen AI 1.7.1 - Hybrid"
            execution_mode = "hybrid_npu_igpu"
            supports = @("hybrid", "npu", "igpu", "gpu")
            context_window = $null
            runtime = "oga"
            notes = "AMD Hybrid OGA models for NPU+iGPU execution."
        }
        npu_4k = [ordered]@{
            key = "npu_4k"
            slug = "ryzen-ai-171-npu-4k"
            title = "Ryzen AI 1.7.1 - NPU 4K"
            execution_mode = "npu_only"
            supports = @("npu")
            context_window = 4096
            runtime = "oga"
            notes = "AMD NPU-only OGA models with up to 4K context."
        }
        npu_16k = [ordered]@{
            key = "npu_16k"
            slug = "ryzen-ai-171-npu-16k"
            title = "Ryzen AI 1.7.1 - NPU 16K"
            execution_mode = "npu_only"
            supports = @("npu")
            context_window = 16384
            runtime = "oga"
            notes = "AMD NPU-only OGA models with up to 16K context."
        }
        npu_lfm2 = [ordered]@{
            key = "npu_lfm2"
            slug = "ryzen-ai-171-npu-lfm2-models"
            title = "Ryzen AI 1.7.1 - NPU LFM2 Models"
            execution_mode = "npu_only"
            supports = @("npu")
            context_window = $null
            runtime = "oga"
            notes = "AMD NPU-only LFM2 OGA models."
        }
    }
}

function _Resolve-HarnessRyzenAiCollectionSelection {
    [CmdletBinding()]
    param(
        [string[]]$Capability,
        [string[]]$CollectionSlug
    )

    $definitions = _Get-HarnessRyzenAiCollectionDefinitions
    $selected = [ordered]@{}

    if ($CollectionSlug -and $CollectionSlug.Count -gt 0) {
        foreach ($slug in $CollectionSlug) {
            if ([string]::IsNullOrWhiteSpace($slug)) {
                continue
            }
            $safeSlug = $slug.Trim()
            $key = ($safeSlug -replace "[^A-Za-z0-9_]+", "_").Trim("_").ToLowerInvariant()
            $selected[$key] = [ordered]@{
                key = $key
                slug = $safeSlug
                title = $safeSlug
                execution_mode = "unknown"
                supports = @("unknown")
                context_window = $null
                runtime = "oga"
                notes = "User-supplied Hugging Face collection slug."
            }
        }
        return $selected
    }

    $requested = @($Capability)
    if ($requested.Count -eq 0) {
        $requested = @("All")
    }

    foreach ($raw in $requested) {
        if ([string]::IsNullOrWhiteSpace($raw)) {
            continue
        }
        switch ($raw.Trim().ToLowerInvariant()) {
            "all" {
                foreach ($key in $definitions.Keys) {
                    $selected[$key] = $definitions[$key]
                }
            }
            "hybrid" {
                $selected["hybrid"] = $definitions["hybrid"]
            }
            "npu" {
                $selected["npu_4k"] = $definitions["npu_4k"]
                $selected["npu_16k"] = $definitions["npu_16k"]
                $selected["npu_lfm2"] = $definitions["npu_lfm2"]
            }
            "npu4k" {
                $selected["npu_4k"] = $definitions["npu_4k"]
            }
            "npu_4k" {
                $selected["npu_4k"] = $definitions["npu_4k"]
            }
            "npu16k" {
                $selected["npu_16k"] = $definitions["npu_16k"]
            }
            "npu_16k" {
                $selected["npu_16k"] = $definitions["npu_16k"]
            }
            "npulfm2" {
                $selected["npu_lfm2"] = $definitions["npu_lfm2"]
            }
            "npu_lfm2" {
                $selected["npu_lfm2"] = $definitions["npu_lfm2"]
            }
            default {
                throw "Unsupported Ryzen AI model capability '$raw'. Expected All, Hybrid, Npu, Npu4K, Npu16K, or NpuLfm2."
            }
        }
    }

    if ($selected.Count -eq 0) {
        $selected["hybrid"] = $definitions["hybrid"]
        $selected["npu_4k"] = $definitions["npu_4k"]
        $selected["npu_16k"] = $definitions["npu_16k"]
        $selected["npu_lfm2"] = $definitions["npu_lfm2"]
    }

    return $selected
}

function _Get-HarnessRyzenAiModelFamily {
    [CmdletBinding()]
    param(
        [string]$ModelId
    )

    $name = if ($null -eq $ModelId) { "" } else { $ModelId.ToLowerInvariant() }
    switch -Regex ($name) {
        "llama" { return "llama" }
        "qwen" { return "qwen" }
        "phi" { return "phi" }
        "mistral" { return "mistral" }
        "gemma" { return "gemma" }
        "codellama" { return "codellama" }
        "lfm2" { return "lfm2" }
        "deepseek" { return "deepseek" }
        default { return "" }
    }
}

function _Get-HarnessObjectMemberValue {
    [CmdletBinding()]
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

function _Normalize-StringList {
    [CmdletBinding()]
    param(
        [object]$Value
    )

    if ($null -eq $Value) {
        return @()
    }

    $values = @()
    if ($Value -is [System.Array]) {
        foreach ($entry in $Value) {
            if ($null -ne $entry) {
                $text = [string]$entry
                if (-not [string]::IsNullOrWhiteSpace($text)) {
                    $values += $text
                }
            }
        }
        return $values
    }

    if ($Value -is [System.Collections.Generic.List[string]]) {
        foreach ($entry in $Value) {
            if (-not [string]::IsNullOrWhiteSpace([string]$entry)) {
                $values += [string]$entry
            }
        }
        return $values
    }

    if ($Value -is [System.Collections.IDictionary]) {
        foreach ($key in $Value.Keys) {
            $entry = $Value[$key]
            if ($null -ne $entry) {
                $text = [string]$entry
                if (-not [string]::IsNullOrWhiteSpace($text)) {
                    $values += $text
                }
            }
        }
        return $values
    }

    $single = [string]$Value
    if (-not [string]::IsNullOrWhiteSpace($single)) {
        $values = @($single)
    }
    return $values
}

function _Expand-TemplateTokens {
    [CmdletBinding()]
    param(
        [string]$Template,
        [hashtable]$Tokens
    )

    if ([string]::IsNullOrWhiteSpace($Template)) {
        return ""
    }

    $expanded = [string]$Template
    if ($Tokens) {
        foreach ($token in $Tokens.Keys) {
            $value = [string]$Tokens[$token]
            $needle = "{{$token}}"
            $expanded = $expanded.Replace($needle, $value)
            $upperNeedle = $needle.ToUpperInvariant()
            if ($upperNeedle -ne $needle) {
                $expanded = $expanded.Replace($upperNeedle, $value)
            }
            $pctNeedle = "%$token%"
            $expanded = $expanded.Replace($pctNeedle, $value)
            $expanded = $expanded.Replace("%$($token.ToUpperInvariant())%", $value)
        }
    }
    return $expanded
}

function _ConvertTo-HarnessSafeModelDirectoryName {
    [CmdletBinding()]
    param(
        [string]$ModelId
    )

    if ([string]::IsNullOrWhiteSpace($ModelId)) {
        return "model"
    }
    return ($ModelId.Trim() -replace "[\\/:\*\?`"<>\|\[\]\s]+", "_").Trim("_")
}

function _Get-HarnessRyzenAiModelDestinationBucket {
    [CmdletBinding()]
    param(
        [object]$CatalogRow,
        [string]$Capability
    )

    if ($Capability -and $Capability.Trim().ToLowerInvariant() -eq "hybrid") {
        return "hybrid"
    }
    if ($Capability -and $Capability.Trim().ToLowerInvariant().StartsWith("npu")) {
        return "npu"
    }

    $supports = @()
    if ($CatalogRow) {
        $supportsValue = $CatalogRow.PSObject.Properties["supports"]
        if ($supportsValue) {
            $supports = @($supportsValue.Value | ForEach-Object { [string]$_ })
        }
    }
    if ($supports -contains "hybrid") {
        return "hybrid"
    }
    if ($supports -contains "npu") {
        return "npu"
    }
    return "amd"
}

function _Invoke-HarnessHuggingFaceSnapshotDownload {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PythonPath,
        [Parameter(Mandatory)]
        [string]$ModelId,
        [Parameter(Mandatory)]
        [string]$DestinationPath,
        [string]$Revision = "",
        [string]$HuggingFaceToken = "",
        [switch]$Force
    )

    $pythonScript = @'
import os
import sys
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("repo_id")
parser.add_argument("local_dir")
parser.add_argument("--revision", default="", required=False)
parser.add_argument("--token", default="", required=False)
parser.add_argument("--force", action="store_true", default=False)
parser.add_argument("--max-workers", default=1, type=int, required=False)
args = parser.parse_args()

repo_id = str(args.repo_id).strip()
local_dir = str(args.local_dir).strip()
revision = str(args.revision).strip() or None
token = str(args.token).strip() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
force_download = bool(args.force)
max_workers = int(args.max_workers or 1)

if not repo_id or not local_dir:
    raise ValueError("repo_id and local_dir are required")

download_kwargs = {
    "repo_id": repo_id,
    "local_dir": local_dir,
    "revision": revision,
    "token": token,
    "force_download": force_download,
    "max_workers": max_workers,
}

try:
    snapshot_download(**download_kwargs)
except TypeError as exc:
    message = str(exc)
    if "max_workers" not in message:
        raise
    download_kwargs.pop("max_workers", None)
    snapshot_download(**download_kwargs)

print(local_dir)
'@

    $tempBase = Join-Path $Script:HarnessLogDir ("hf-dl-" + [System.Guid]::NewGuid().ToString("N"))
    $stdOutPath = "$tempBase.out"
    $stdErrPath = "$tempBase.err"
    $tempScriptSourcePath = [System.IO.Path]::GetTempFileName()
    $tempScriptPath = [System.IO.Path]::ChangeExtension($tempScriptSourcePath, ".py")
    if (-not (Test-Path -LiteralPath $Script:HarnessLogDir -PathType Container)) {
        New-Item -ItemType Directory -Path $Script:HarnessLogDir -Force | Out-Null
    }
    $stagingRoot = Join-Path $Script:HarnessLogDir ("hf-dl-stage-" + [System.Guid]::NewGuid().ToString("N"))
    if (-not (Test-Path -LiteralPath $stagingRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null
    }

    $exitCode = 1
    $output = @()
    $retryErrorPatterns = @("WinError 32", "PermissionError", "being used by another process", "Access is denied")
    $isRetryableLockError = {
        param([string]$Text)
        if ([string]::IsNullOrWhiteSpace($Text)) {
            return $false
        }
        foreach ($pattern in $retryErrorPatterns) {
            if ($Text -match $pattern) {
                return $true
            }
        }
        return $false
    }
    try {
        $maxAttempts = 3
        for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
            $attemptDestination = Join-Path $stagingRoot ("attempt-$attempt")
            if (Test-Path -LiteralPath $attemptDestination -PathType Container) {
                Remove-Item -LiteralPath $attemptDestination -Recurse -Force -ErrorAction SilentlyContinue
            }
            New-Item -ItemType Directory -Path $attemptDestination -Force | Out-Null

            try {
                Set-Content -LiteralPath $tempScriptPath -Value $pythonScript -Encoding UTF8
                $pythonArguments = @(
                    $tempScriptPath,
                    $ModelId,
                    $attemptDestination
                )
                if (-not [string]::IsNullOrWhiteSpace($Revision)) {
                    $pythonArguments += "--revision"
                    $pythonArguments += $Revision
                }
                if (-not [string]::IsNullOrWhiteSpace($HuggingFaceToken)) {
                    $pythonArguments += "--token"
                    $pythonArguments += $HuggingFaceToken
                }
                if ($Force.IsPresent -and $attempt -eq 1) {
                    $pythonArguments += "--force"
                }
                $pythonArguments += "--max-workers"
                $pythonArguments += "1"
                $proc = Start-Process -FilePath $PythonPath -ArgumentList $pythonArguments -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdOutPath -RedirectStandardError $stdErrPath -ErrorAction Stop
                $exitCode = [int]$proc.ExitCode
                $outputText = ""
                if (Test-Path -LiteralPath $stdOutPath -PathType Leaf) {
                    $outputText = Get-Content -LiteralPath $stdOutPath -Raw -ErrorAction SilentlyContinue
                }
                $errorText = ""
                if (Test-Path -LiteralPath $stdErrPath -PathType Leaf) {
                    $errorText = Get-Content -LiteralPath $stdErrPath -Raw -ErrorAction SilentlyContinue
                }
                $output = @()
                if (-not [string]::IsNullOrWhiteSpace($outputText)) {
                    $output += $outputText
                }
                if (-not [string]::IsNullOrWhiteSpace($errorText)) {
                    $output += $errorText
                }
                if (-not $output -and $exitCode -eq 0) {
                    $output = @()
                }
                if ($exitCode -eq 0) {
                    try {
                        if ($Force.IsPresent -and (Test-Path -LiteralPath $DestinationPath -PathType Container)) {
                            try {
                                Remove-Item -LiteralPath $DestinationPath -Recurse -Force -ErrorAction Stop
                            } catch {
                                $output = @($_.Exception.Message)
                                $exitCode = 1
                                if (($attempt -lt $maxAttempts) -and (& $isRetryableLockError $_.Exception.Message)) {
                                    Start-Sleep -Seconds (1 * $attempt)
                                    continue
                                }
                                break
                            }
                        }
                        if (Test-Path -LiteralPath $DestinationPath -PathType Container) {
                            throw "Destination exists and overwrite was not requested."
                        }
                        try {
                            Move-Item -LiteralPath $attemptDestination -Destination $DestinationPath -Force -ErrorAction Stop
                        } catch {
                            $output = @($_.Exception.Message)
                            $exitCode = 1
                            if (($attempt -lt $maxAttempts) -and (& $isRetryableLockError $_.Exception.Message)) {
                                Start-Sleep -Seconds (1 * $attempt)
                                continue
                            }
                            break
                        }
                        $output = @("downloaded=$DestinationPath")
                        break
                    } catch {
                        $output = @($_.Exception.Message)
                        $exitCode = 1
                    }
                }
                if ($exitCode -eq 0 -or $attempt -eq $maxAttempts) {
                    break
                }
                if ($exitCode -ne 0 -and ((& $isRetryableLockError $errorText) -or (& $isRetryableLockError $outputText))) {
                    Start-Sleep -Seconds (1 * $attempt)
                    continue
                }
                if ($exitCode -ne 0 -and -not $output) {
                    $output = @("Python process exited with code $exitCode and no captured output.")
                }
                break
            } catch {
                $exitCode = 1
                $output = @($_.Exception.Message)
                if (($attempt -lt $maxAttempts) -and (& $isRetryableLockError $_.Exception.Message)) {
                    Start-Sleep -Seconds (1 * $attempt)
                    continue
                }
                break
            }
        }
    } finally {
        if ($tempScriptSourcePath) {
            Remove-Item -LiteralPath $tempScriptSourcePath -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $tempScriptPath -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stdOutPath -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stdErrPath -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    return [ordered]@{
        exit_code = $exitCode
        output = @($output | ForEach-Object { [string]$_ })
    }
}

<#
.SYNOPSIS
Lists AMD Ryzen AI Hybrid and NPU LLM model artifacts suitable for ASUS Ryzen AI laptops.

.DESCRIPTION
Reads AMD's Ryzen AI Hugging Face collections and returns model rows tagged by
execution mode, supported hardware path, runtime, context window, and ASUS HX
370 platform notes. By default, all AMD Ryzen AI 1.7.1 Hybrid and NPU LLM
collections are queried.
#>
function Get-HarnessRyzenAiModel {
    [CmdletBinding()]
    param(
        [ValidateSet("All", "Hybrid", "Npu", "Npu4K", "Npu_4K", "Npu16K", "Npu_16K", "NpuLfm2", "Npu_Lfm2")]
        [string[]]$Capability = @("All"),
        [string]$Filter = "",
        [string[]]$CollectionSlug = @(),
        [string]$Processor = "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
        [switch]$IncludeRepositoryMetadata,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $collections = _Resolve-HarnessRyzenAiCollectionSelection -Capability $Capability -CollectionSlug $CollectionSlug
    $rowsByModel = @{}

    foreach ($collectionKey in $collections.Keys) {
        $definition = $collections[$collectionKey]
        $slug = [string]$definition.slug
        $uri = "https://huggingface.co/api/collections/amd/$slug"
        Write-Verbose "Reading Hugging Face collection $uri"
        try {
            $collection = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 30
        } catch {
            throw "Failed to read AMD Ryzen AI collection '$slug' from Hugging Face: $($_.Exception.Message)"
        }

        $items = @($collection.items)
        foreach ($item in $items) {
            $repoType = [string](_Get-HarnessObjectMemberValue -InputObject $item -Name "repoType" -Default "")
            $type = [string](_Get-HarnessObjectMemberValue -InputObject $item -Name "type" -Default "")
            if ($repoType -and $repoType -ne "model") {
                continue
            }
            if ($type -and $type -ne "model") {
                continue
            }

            $modelId = [string](_Get-HarnessObjectMemberValue -InputObject $item -Name "id" -Default "")
            if ([string]::IsNullOrWhiteSpace($modelId)) {
                continue
            }
            if (-not [string]::IsNullOrWhiteSpace($Filter)) {
                if ($modelId.ToLowerInvariant() -notmatch [regex]::Escape($Filter.ToLowerInvariant())) {
                    continue
                }
            }

            $lookupKey = $modelId.ToLowerInvariant()
            if (-not $rowsByModel.ContainsKey($lookupKey)) {
                $rowsByModel[$lookupKey] = [ordered]@{
                    model_id = $modelId
                    model_family = _Get-HarnessRyzenAiModelFamily -ModelId $modelId
                    execution_modes = @()
                    supports = @()
                    context_windows = @()
                    runtime = @()
                    collections = @()
                    collection_titles = @()
                    pipeline_tag = [string](_Get-HarnessObjectMemberValue -InputObject $item -Name "pipeline_tag" -Default "")
                    gated = [bool](_Get-HarnessObjectMemberValue -InputObject $item -Name "gated" -Default $false)
                    downloads = [int](_Get-HarnessObjectMemberValue -InputObject $item -Name "downloads" -Default 0)
                    likes = [int](_Get-HarnessObjectMemberValue -InputObject $item -Name "likes" -Default 0)
                    last_modified = [string](_Get-HarnessObjectMemberValue -InputObject $item -Name "lastModified" -Default "")
                    collection_last_updated = @()
                    source_url = "https://huggingface.co/$modelId"
                    recommended_for_device = "ASUS laptop with AMD Ryzen AI 9 HX 370 / Radeon 890M"
                    platform_match = if ($Processor -match "(?i)ryzen ai 9 hx 37|ryzen ai 300|strix") { "ryzen_ai_300_class" } else { "unknown" }
                    platform_notes = "Requires Windows 11 24H2+ and ASUS/AMD Ryzen AI NPU driver stack. Hybrid rows require AMD OGA/Lemonade-compatible Hybrid runtime; download alone does not register a harness backend."
                    repository_sha = ""
                    repository_tags = @()
                    repository_files = @()
                }
            }

            $row = $rowsByModel[$lookupKey]
            foreach ($mode in @($definition.execution_mode)) {
                if ($mode -and $row["execution_modes"] -notcontains $mode) {
                    $row["execution_modes"] += $mode
                }
            }
            foreach ($support in @($definition.supports)) {
                if ($support -and $row["supports"] -notcontains $support) {
                    $row["supports"] += $support
                }
            }
            if ($null -ne $definition.context_window -and $row["context_windows"] -notcontains [int]$definition.context_window) {
                $row["context_windows"] += [int]$definition.context_window
            }
            foreach ($runtime in @($definition.runtime)) {
                if ($runtime -and $row["runtime"] -notcontains $runtime) {
                    $row["runtime"] += $runtime
                }
            }
            if ($row["collections"] -notcontains $slug) {
                $row["collections"] += $slug
            }
            $collectionTitle = if ($collection.title) { [string]$collection.title } else { [string]$definition.title }
            if ($row["collection_titles"] -notcontains $collectionTitle) {
                $row["collection_titles"] += $collectionTitle
            }
            $collectionLastUpdated = [string](_Get-HarnessObjectMemberValue -InputObject $collection -Name "lastUpdated" -Default "")
            if ($collectionLastUpdated -and $row["collection_last_updated"] -notcontains $collectionLastUpdated) {
                $row["collection_last_updated"] += $collectionLastUpdated
            }
        }
    }

    $rows = @()
    foreach ($key in $rowsByModel.Keys) {
        $row = $rowsByModel[$key]
        if ($IncludeRepositoryMetadata.IsPresent) {
            $modelUri = "https://huggingface.co/api/models/$($row["model_id"])"
            Write-Verbose "Reading Hugging Face model metadata $modelUri"
            try {
                $modelInfo = Invoke-RestMethod -Uri $modelUri -Method Get -TimeoutSec 30
                $row["repository_sha"] = [string](_Get-HarnessObjectMemberValue -InputObject $modelInfo -Name "sha" -Default "")
                $row["repository_tags"] = @((_Get-HarnessObjectMemberValue -InputObject $modelInfo -Name "tags" -Default @()) | ForEach-Object { [string]$_ })
                $row["repository_files"] = @((_Get-HarnessObjectMemberValue -InputObject $modelInfo -Name "siblings" -Default @()) | ForEach-Object {
                    [string](_Get-HarnessObjectMemberValue -InputObject $_ -Name "rfilename" -Default "")
                } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            } catch {
                Write-Warning "Failed to read model metadata for '$($row["model_id"])': $($_.Exception.Message)"
            }
        }
        $rows += [pscustomobject]$row
    }

    return _Apply-HarnessOutputOptions -InputObject (
        $rows | Sort-Object -Property model_family, model_id
    ) -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Downloads an AMD Ryzen AI Hybrid or NPU model artifact from Hugging Face.

.DESCRIPTION
Downloads known AMD Ryzen AI model repositories discovered by
Get-HarnessRyzenAiModel into local .models folders. The cmdlet validates model
IDs against AMD's Ryzen AI collections by default and supports ShouldProcess via
-WhatIf and -Confirm.
#>
function Save-HarnessRyzenAiModel {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipeline = $true, ValueFromPipelineByPropertyName = $true)]
        [Alias("id", "model_id")]
        [string[]]$Model,
        [ValidateSet("Auto", "Hybrid", "Npu", "Npu4K", "Npu_4K", "Npu16K", "Npu_16K", "NpuLfm2", "Npu_Lfm2")]
        [string]$Capability = "Auto",
        [string]$DestinationRoot = (Join-Path $Script:HarnessRepoRoot ".models\amd-ryzen-ai"),
        [string]$Revision = "",
        [string]$HuggingFaceToken = "",
        [string]$PythonPath = "",
        [switch]$AllowUnlisted,
        [switch]$Force,
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    begin {
        $resolvedPythonPath = _Resolve-HarnessPythonPath -PythonPath $PythonPath
        $catalogCapability = if ($Capability -eq "Auto") { @("All") } else { @($Capability) }
        $catalogRows = @(Get-HarnessRyzenAiModel -Capability $catalogCapability)
        $catalogById = @{}
        foreach ($row in $catalogRows) {
            $catalogById[[string]$row.model_id] = $row
        }
        $results = @()
    }

    process {
        foreach ($rawModel in @($Model)) {
            if ([string]::IsNullOrWhiteSpace($rawModel)) {
                continue
            }
            $modelId = [string]$rawModel
            if ($modelId -match "(?i)^hf://") {
                $modelId = $modelId.Substring(5)
            }
            $modelId = $modelId.Trim()
            if ([string]::IsNullOrWhiteSpace($modelId)) {
                continue
            }

            $catalogRow = $null
            foreach ($candidate in $catalogById.Keys) {
                if ($candidate.Equals($modelId, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $catalogRow = $catalogById[$candidate]
                    $modelId = [string]$catalogRow.model_id
                    break
                }
            }

            if (-not $catalogRow -and -not $AllowUnlisted.IsPresent) {
                throw "Model '$modelId' was not found in AMD's Ryzen AI Hybrid/NPU catalog. Re-run with -AllowUnlisted to download anyway."
            }

            $bucket = _Get-HarnessRyzenAiModelDestinationBucket -CatalogRow $catalogRow -Capability $Capability
            $safeName = _ConvertTo-HarnessSafeModelDirectoryName -ModelId $modelId
            $targetPath = Join-Path (Join-Path $DestinationRoot $bucket) $safeName
            $existingFiles = @()
            if (Test-Path -LiteralPath $targetPath -PathType Container) {
                $existingFiles = @(Get-ChildItem -LiteralPath $targetPath -Force -ErrorAction SilentlyContinue | Select-Object -First 1)
            }

            $action = "download"
            if ($existingFiles.Count -gt 0 -and -not $Force.IsPresent) {
                $action = "exists"
                $results += [pscustomobject][ordered]@{
                    model_id = $modelId
                    capability = $Capability
                    bucket = $bucket
                    destination = $targetPath
                    status = "exists"
                    exit_code = 0
                    source_url = "https://huggingface.co/$modelId"
                    supports = if ($catalogRow) { @($catalogRow.supports) } else { @() }
                    execution_modes = if ($catalogRow) { @($catalogRow.execution_modes) } else { @() }
                    message = "Destination already contains files. Use -Force to refresh."
                }
                continue
            }

            if ($DryRun.IsPresent) {
                $results += [pscustomobject][ordered]@{
                    model_id = $modelId
                    capability = $Capability
                    bucket = $bucket
                    destination = $targetPath
                    status = "dry_run"
                    exit_code = 0
                    source_url = "https://huggingface.co/$modelId"
                    supports = if ($catalogRow) { @($catalogRow.supports) } else { @() }
                    execution_modes = if ($catalogRow) { @($catalogRow.execution_modes) } else { @() }
                    message = "Dry run only. No files were downloaded."
                }
                continue
            }

            if ($PSCmdlet.ShouldProcess($modelId, "Download Hugging Face snapshot to $targetPath")) {
                try {
                    if (-not (Test-Path -LiteralPath $targetPath -PathType Container)) {
                        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
                    }
                    $download = _Invoke-HarnessHuggingFaceSnapshotDownload `
                        -PythonPath $resolvedPythonPath `
                        -ModelId $modelId `
                        -DestinationPath $targetPath `
                        -Revision $Revision `
                        -HuggingFaceToken $HuggingFaceToken `
                        -Force:$Force.IsPresent

                    $status = if ([int]$download.exit_code -eq 0) { "downloaded" } else { "failed" }
                    $statusText = (@($download.output) -join "`n")
                } catch {
                    $status = "failed"
                    $statusText = $_.Exception.Message
                    $download = @{
                        exit_code = 1
                        output = @($_.Exception.Message)
                    }
                }
                if (-not $statusText) {
                    $statusText = "No output returned from downloader; check python/runtime logs."
                }

                $results += [pscustomobject][ordered]@{
                    model_id = $modelId
                    capability = $Capability
                    bucket = $bucket
                    destination = $targetPath
                    status = $status
                    exit_code = [int]$download.exit_code
                    source_url = "https://huggingface.co/$modelId"
                    supports = if ($catalogRow) { @($catalogRow.supports) } else { @() }
                    execution_modes = if ($catalogRow) { @($catalogRow.execution_modes) } else { @() }
                    message = $statusText
                }
            }
        }
    }

    end {
        return _Apply-HarnessOutputOptions -InputObject $results -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }
}

<#
.SYNOPSIS
Downloads one or more compatibility candidates into per-device local cache folders.
#>
function Sync-HarnessCompatibleModels {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [Parameter(Mandatory, Position = 0, ValueFromPipeline = $true)]
        [string[]]$Model,
        [string[]]$Devices = @("cpu", "gpu", "npu"),
        [hashtable]$DeviceModelMap = @{},
        [string]$CatalogPath = "",
        [switch]$RequireCompatibility,
        [string]$DestinationRoot = (Join-Path $Script:HarnessRepoRoot ".models"),
        [string]$PythonPath = "",
        [switch]$Force,
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $resolvedPythonPath = _Resolve-HarnessPythonPath -PythonPath $PythonPath
    $requestedDevices = @(_Normalize-HardwareDeviceSelection -Devices $Devices)
    $compatibilityRows = Get-HarnessCompatibleModels -Devices $requestedDevices -CatalogPath $CatalogPath
    $results = @()

    foreach ($rawModel in $Model) {
        if ([string]::IsNullOrWhiteSpace($rawModel)) {
            continue
        }
        $normalizedModel = [string]$rawModel
        if ($normalizedModel -match "(?i)^hf://") {
            $normalizedModel = $normalizedModel.Substring(5)
        }
        if ([string]::IsNullOrWhiteSpace($normalizedModel)) {
            continue
        }

        $matched = @(
            foreach ($entry in $compatibilityRows) {
                $entryModelId = [string](_Normalize-CompatibleModelEntry -Entry $entry).model_id
                if ($entryModelId -eq $normalizedModel) {
                    $entry
                }
            }
        )
        if ($RequireCompatibility -and $matched.Count -eq 0) {
            throw "Sync-HarnessCompatibleModels requires known compatibility match for '$normalizedModel', but no catalog entry matched with current device selection."
        }

        $catalogRowForModel = if ($matched.Count -gt 0) { $matched[0] } else { $null }
        $downloadModelId = _Resolve-CompatibleModelDownloadId -ModelId $normalizedModel -CatalogSource ([string]($catalogRowForModel.source))

        foreach ($device in $requestedDevices) {
            $safeModel = $normalizedModel -replace "[\\/:\[\]\s]+", "_"
            $portDeviceModel = if ($DeviceModelMap.ContainsKey($device)) {
                [string]$DeviceModelMap[$device]
            } else {
                $normalizedModel
            }
            $targetPath = Join-Path (Join-Path $DestinationRoot $device) $safeModel
            $action = "skipped"
            $statusCode = 0
            $statusText = ""
            $requiresPython3 = $false

            if ([string]::IsNullOrWhiteSpace($downloadModelId)) {
                $action = "error"
                $statusCode = 1
                $statusText = "No valid Hugging Face model id resolved from '$normalizedModel'. Provide hf://<owner>/<repo> or a catalog source mapping."
            } else {
                if (-not (Test-Path -LiteralPath $targetPath -PathType Container)) {
                    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
                } else {
                    $hasFiles = Get-ChildItem -Path $targetPath -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
                    if ($hasFiles -and -not $Force) {
                        $action = "cached"
                    }
                }
            }

            if (-not [string]::IsNullOrWhiteSpace($downloadModelId) -and -not $DryRun -and $action -ne "cached") {
                if ($PSCmdlet.ShouldProcess("$normalizedModel[$device]", "Download to $targetPath")) {
                    try {
                        $download = _Invoke-HarnessHuggingFaceSnapshotDownload `
                            -PythonPath $resolvedPythonPath `
                            -ModelId $downloadModelId `
                            -DestinationPath $targetPath `
                            -Force:$Force.IsPresent
                        $statusCode = [int]$download.exit_code
                        $statusText = (@($download.output) -join "`n")
                        if ($statusCode -ne 0) {
                            throw $statusText
                        }
                        if ($statusText -match "requires_python3") {
                            $requiresPython3 = $true
                        }
                        $action = "downloaded"
                    } catch {
                        $statusCode = 1
                        $statusText = $_.Exception.Message
                        if ($statusText -match "requires_python3") {
                            $requiresPython3 = $true
                        }
                        $action = "error"
                    }
                } else {
                    $action = "would_download"
                    $requiresPython3 = $false
                }
            } elseif ($DryRun -and -not [string]::IsNullOrWhiteSpace($downloadModelId)) {
                $action = "dryrun"
                $statusText = "dry-run preview for snapshot_download model=$downloadModelId destination=$targetPath"
                $requiresPython3 = $false
            }

            $results += [pscustomobject][ordered]@{
                model = $normalizedModel
                device = $device
                backend_model = if (-not [string]::IsNullOrWhiteSpace($portDeviceModel)) { $portDeviceModel } else { $normalizedModel }
                destination = $targetPath
                action = $action
                status_code = [int]$statusCode
                message = $statusText
                requires_python3 = [bool]$requiresPython3
                dry_run = [bool]$DryRun
            }
        }
    }

    return _Apply-HarnessOutputOptions -InputObject $results -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

<#
.SYNOPSIS
Downloads all models from the compatibility catalog that satisfy selected device coverage.
.DESCRIPTION
Use this to seed per-device local caches before starting a multi-device local stack.
#>
function Sync-HarnessCompatibleModelsForDevices {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [string[]]$Devices = @("cpu", "gpu", "npu"),
        [ValidateSet("all", "any")]
        [string]$MatchMode = "all",
        [string]$CatalogPath = "",
        [string]$Filter = "",
        [hashtable]$DeviceModelMap = @{},
        [string]$DestinationRoot = (Join-Path $Script:HarnessRepoRoot ".models"),
        [string]$PythonPath = "",
        [switch]$Force,
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $requested = @(_Normalize-HardwareDeviceSelection -Devices $Devices)
    $compat = @(
        Get-HarnessCompatibleModels -Devices $requested -MatchMode $MatchMode -CatalogPath $CatalogPath -Filter $Filter
    )
    $models = @(
        foreach ($entry in $compat) {
            if ($entry -and $entry.model_id) { [string]$entry.model_id } else { "" }
        }
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    $models = @($models)

    if (-not $models -or $models.Count -eq 0) {
        return _Apply-HarnessOutputOptions -InputObject @(
            [ordered]@{
                action = "no_candidates"
                requested_devices = @($requested)
                match_mode = $MatchMode
                catalog_path = $CatalogPath
                dry_run = [bool]$DryRun
                filtered = $Filter
                count = 0
            }
        ) -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    return Sync-HarnessCompatibleModels `
        -Model $models `
        -Devices $requested `
        -DeviceModelMap $DeviceModelMap `
        -CatalogPath $CatalogPath `
        -DestinationRoot $DestinationRoot `
        -PythonPath $PythonPath `
        -Force:$Force.IsPresent `
        -DryRun:$DryRun.IsPresent `
        -Property $Property `
        -ExpandProperty $ExpandProperty `
        -AsJson:$AsJson.IsPresent `
        -JsonDepth $JsonDepth
}

Set-Alias -Name Sync-HarnessCrossDeviceCompatibleModels -Value Sync-HarnessCompatibleModelsForDevices
Set-Alias -Name Sync-HarnessStackModels -Value Sync-HarnessCompatibleModelsForDevices

<#
.SYNOPSIS
Starts one or more local harness-owned backends for selected hardware devices.
.DESCRIPTION
Use this when you want to run the same or per-device model on NPU/GPU/CPU local providers.
#>
function Start-HarnessModelBackendFleet {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [string[]]$Devices = @("npu", "gpu", "cpu"),
        [hashtable]$DevicePortMap = @{ npu = 11433; gpu = 11434; hybrid = 13305; cpu = 11435 },
        [string]$Model = $Script:HarnessModelBackendDefaultModel,
        [hashtable]$ModelByDevice = @{},
        [string]$ModelPath = "",
        [hashtable]$ModelPathByDevice = @{},
    [string]$ModelsRoot = "",
    [hashtable]$ModelsRootByDevice = @{},
    [string[]]$ExtraModel,
    [hashtable]$ExtraModelByDevice = @{},
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
        [hashtable]$DeviceBackendProfiles = @{},
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $selectedDevices = @(_Normalize-HardwareDeviceSelection -Devices $Devices)
    $results = @()
    $errors = @()
    $requiredFailures = @()
    $seenPorts = @{}
    $deviceProfileMap = @{}
    foreach ($target in $selectedDevices) {
        $deviceProfileMap[$target] = if ($DeviceBackendProfiles.ContainsKey($target)) {
            $DeviceBackendProfiles[$target]
        } else {
            @{}
        }
    }

    foreach ($target in $selectedDevices) {
        $deviceProfile = $deviceProfileMap[$target]
        $profileRuntime = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "runtime" -Default "local_provider")
        $profileHealthEndpoint = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "health_endpoint" -Default "")
        $profileBaseUrl = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "base_url" -Default "")
        $profileRequired = [bool](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "required" -Default $true)
        $profileBackendId = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "backend_id" -Default "")
        $profileLaunchMode = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "launch_mode" -Default "auto")
        $profileStartCommand = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "start_command" -Default "")
        $profileStartWorkingDirectory = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "start_working_directory" -Default "")
        $profileStartArgs = _Normalize-StringList -Value (_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "start_args" -Default @())
        $deviceModel = if ($ModelByDevice.ContainsKey($target)) {
            [string]$ModelByDevice[$target]
        } else {
            $Model
        }
        $planModel = if ($profileBackendId) {
            "{0}/{1}" -f $target, $profileBackendId
        } else {
            $target
        }
        if ($target -eq "hybrid" -and $profileRequired -and -not $DeviceBackendProfiles.ContainsKey("hybrid")) {
            $expectedPort = if ($DevicePortMap.ContainsKey("hybrid")) { [int]$DevicePortMap["hybrid"] } else { 13305 }
            $expectedHealthUrl = _HealthUrl -TargetHost $ModelBackendHost -Port $expectedPort
            $expectedCatalogUrl = _Build-ModelCatalogUrl -BackendBaseUrl "http://$ModelBackendHost`:$expectedPort"
            $missingError = "Hybrid backend requested but no backend profile is registered. Add a backend with id/device 'hybrid' and a usable endpoint (expected port=$expectedPort, health=$expectedHealthUrl, catalog=$expectedCatalogUrl) to the config before invoking."
            $errors += $missingError
            $requiredFailures += $missingError
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $expectedPort
                required = $profileRequired
                profile_present = $false
                runtime = $profileRuntime
                model = $deviceModel
                status = "skipped"
                action = "missing_profile"
                launch_mode = "external"
                host = $ModelBackendHost
                health_url = $expectedHealthUrl
                catalog_url = $expectedCatalogUrl
                response = [ordered]@{
                    required = $profileRequired
                    runtime = "missing"
                    health_reachable = $false
                    catalog_reachable = $false
                    required_endpoint = $profileRequired
                    plan_backend = $planModel
                }
            }
            continue
        }
        if (-not $DevicePortMap.ContainsKey($target)) {
            $missingPortError = if ($target -eq "hybrid") {
                "No port mapping configured for required device 'hybrid'. Add DevicePortMap['hybrid'] = 13305 before invoking."
            } else {
                "No port mapping configured for device '$target'."
            }
            $errors += $missingPortError
            if ($profileRequired) {
                $requiredFailures += $missingPortError
            }
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $null
                required = $profileRequired
                profile_present = $target -eq "hybrid" -and $DeviceBackendProfiles.ContainsKey("hybrid")
                runtime = $profileRuntime
                model = $deviceModel
                status = "skipped"
                action = "missing_port"
                launch_mode = "unconfigured"
                host = $ModelBackendHost
                response = [ordered]@{
                    required = $profileRequired
                    runtime = $profileRuntime
                    required_endpoint = $profileRequired
                    plan_backend = $planModel
                }
            }
            continue
        }
        $port = [int]$DevicePortMap[$target]
        if ($seenPorts.ContainsKey($port)) {
            $duplicateError = "Duplicate port mapping detected ($port) for device '$target'. Each requested device requires a unique port."
            $errors += $duplicateError
            if ($profileRequired) {
                $requiredFailures += $duplicateError
            }
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                required = $profileRequired
                status = "conflict"
                action = "port_collision"
                launch_mode = "unconfigured"
                host = $ModelBackendHost
                runtime = $profileRuntime
                model = $deviceModel
                health_url = _HealthUrl -TargetHost $ModelBackendHost -Port $port
                response = [ordered]@{
                    required = $profileRequired
                    runtime = $profileRuntime
                    required_endpoint = $profileRequired
                    plan_backend = $planModel
                }
            }
            continue
        }
        $seenPorts[$port] = $target
        $deviceModelPath = if ($ModelPathByDevice.ContainsKey($target)) {
            [string]$ModelPathByDevice[$target]
        } else {
            $ModelPath
        }
        $deviceModelsRoot = if ($ModelsRootByDevice.ContainsKey($target)) {
            [string]$ModelsRootByDevice[$target]
        } else {
            $ModelsRoot
        }
        $profileModel = [string](_Get-HarnessObjectMemberValue -InputObject $deviceProfile -Name "model" -Default "")
        if ([string]::IsNullOrWhiteSpace($profileModel)) {
            $profileModel = $deviceModel
        }
        $launchMode = [string]$profileLaunchMode.Trim().ToLowerInvariant()
        $plannedModel = if ([string]::IsNullOrWhiteSpace($profileModel)) { $deviceModel } else { $profileModel }
        $healthProbeUrl = _Normalize-BackendEndpointUrl -BaseHost $ModelBackendHost -BasePort $port -Candidate $profileHealthEndpoint -Fallback (_HealthUrl -TargetHost $ModelBackendHost -Port $port)
        if ([string]::IsNullOrWhiteSpace($profileBaseUrl)) {
            $catalogProbeUrl = _Build-ModelCatalogUrl -BackendBaseUrl ($healthProbeUrl -replace "/health$")
        } else {
            $catalogProbeUrl = _Build-ModelCatalogUrl -BackendBaseUrl $profileBaseUrl
        }
        $templateTokens = @{
            model = $plannedModel
            backend = $target
            device = $target
            port = $port.ToString()
            host = $ModelBackendHost
            runtime = $profileRuntime
        }
        if ([string]::IsNullOrWhiteSpace($launchMode) -or $launchMode -eq "auto") {
            if (-not [string]::IsNullOrWhiteSpace($profileStartCommand)) {
                $launchMode = "command"
            } elseif (($target -eq "hybrid" -and $profileRuntime -notin @("local_provider", "openai", "auto")) -or
                ($target -ne "hybrid" -and $profileRuntime -and $profileRuntime -ne "local_provider")) {
                $launchMode = "external"
            } elseif ($profileRuntime -eq "llamacpp" -or [string]::IsNullOrWhiteSpace($profileStartCommand)) {
                $launchMode = "local"
            } else {
                $launchMode = "command"
            }
        }
        if ($launchMode -notin @("local", "external", "command")) {
            $launchMode = "external"
        }

        if ($launchMode -eq "external") {
            $plannedModel = if ([string]::IsNullOrWhiteSpace($profileModel)) { $deviceModel } else { $profileModel }
            $healthProbeUrl = _Normalize-BackendEndpointUrl -BaseHost $ModelBackendHost -BasePort $port -Candidate $profileHealthEndpoint -Fallback (_HealthUrl -TargetHost $ModelBackendHost -Port $port)
            $catalogProbeUrl = _Build-ModelCatalogUrl -BackendBaseUrl $(
                if ([string]::IsNullOrWhiteSpace($profileBaseUrl)) {
                    $healthProbeUrl -replace "/health$", ""
                } else {
                    $profileBaseUrl
                }
            )
            $healthReachable = $false
            $catalogReachable = $false
            try {
                $healthReachable = _Wait-HttpReady -HealthUrl $healthProbeUrl -TimeoutSeconds $WaitSeconds
            } catch {
                $healthReachable = $false
            }
            try {
                if ($healthReachable -and -not [string]::IsNullOrWhiteSpace($catalogProbeUrl)) {
                    $catalogReachable = _Wait-HttpReady -HealthUrl $catalogProbeUrl -TimeoutSeconds $WaitSeconds
                }
            } catch {
                $catalogReachable = $false
            }

            if (-not ($healthReachable -and $catalogReachable)) {
                $missingMessage = "External backend for '$target' was not reachable at health='$healthProbeUrl' or catalog='$catalogProbeUrl'."
                if ($profileRequired) {
                    $errors += "[${target}:$port] $missingMessage"
                    $requiredFailures += "[${target}:$port] $missingMessage"
                }
                $results += [pscustomobject][ordered]@{
                    device = $target
                    port = $port
                    started = $false
                    action = "external_unreachable"
                    launch_mode = "external"
                    host = $ModelBackendHost
                    model = $plannedModel
                    health_url = $healthProbeUrl
                    catalog_url = $catalogProbeUrl
                    required = $profileRequired
                    runtime = $profileRuntime
                    profile = $planModel
                    response = [ordered]@{
                        required = $profileRequired
                        runtime = $profileRuntime
                        health_reachable = $healthReachable
                        catalog_reachable = $catalogReachable
                        required_endpoint = $profileRequired
                    }
                }
                continue
            }
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                started = $true
                    action = "external_ready"
                    launch_mode = "external"
                    host = $ModelBackendHost
                    model = $plannedModel
                    health_url = $healthProbeUrl
                    catalog_url = $catalogProbeUrl
                    required = $profileRequired
                    runtime = $profileRuntime
                    profile = $planModel
                    response = [ordered]@{
                        required = $profileRequired
                        runtime = $profileRuntime
                        health_reachable = $healthReachable
                        catalog_reachable = $catalogReachable
                    base_url = $profileBaseUrl
                }
            }
            continue
        }

        if ($launchMode -eq "command") {
            $resolvedCommand = if (-not [string]::IsNullOrWhiteSpace($profileStartCommand)) { $profileStartCommand } else { "" }
            $resolvedCommand = $resolvedCommand.Replace("{HOST}", $ModelBackendHost).Replace("{host}", $ModelBackendHost)
            $resolvedCommand = $resolvedCommand.Replace("{PORT}", [string]$port).Replace("{port}", [string]$port)
            $resolvedCommand = $resolvedCommand.Replace("{MODEL}", $plannedModel).Replace("{model}", $plannedModel)
            $resolvedCommand = $resolvedCommand.Replace("{DEVICE}", $target).Replace("{device}", $target)
            $commandParts = _Parse-CommandLine -CommandLine $resolvedCommand
            if ($commandParts.Count -eq 0) {
                $commandParseMessage = "Command launch for '$target' has an empty command string after tokenization."
                $errors += "[${target}:$port] $commandParseMessage"
                if ($profileRequired) {
                    $requiredFailures += "[${target}:$port] $commandParseMessage"
                }
                $results += [pscustomobject][ordered]@{
                    device = $target
                    port = $port
                    started = $false
                    action = "invalid_command"
                    launch_mode = "command"
                    host = $ModelBackendHost
                    model = $plannedModel
                    required = $profileRequired
                    runtime = $profileRuntime
                    profile = $planModel
                    health_url = $healthProbeUrl
                    catalog_url = $catalogProbeUrl
                    response = [ordered]@{
                        required = $profileRequired
                        runtime = $profileRuntime
                        health_reachable = $false
                        catalog_reachable = $false
                        required_endpoint = $profileRequired
                        error = $commandParseMessage
                        plan_backend = $planModel
                    }
                }
                continue
            }
            $commandExecutable = [string]$commandParts[0]
            $commandArgs = @()
            if ($commandParts.Count -gt 1) {
                $commandArgs = @($commandParts[1..($commandParts.Count - 1)])
            }
            if ($profileStartArgs.Count -gt 0) {
                $commandArgs += $profileStartArgs
            }
            if ($commandArgs -notcontains "--host") {
                $commandArgs += @("--host", $ModelBackendHost)
            }
            if ($commandArgs -notcontains "--port") {
                $commandArgs += @("--port", $port.ToString())
            }
            if ($commandArgs -notcontains "--model") {
                $commandArgs += @("--model", $plannedModel)
            }
            $commandForDebug = _Format-HarnessCommand -Executable $commandExecutable -Arguments $commandArgs

            if ($DryRun) {
                $results += [pscustomobject][ordered]@{
                    device = $target
                    port = $port
                    started = $false
                    action = "would_start"
                    launch_mode = "command"
                    host = $ModelBackendHost
                    model = $plannedModel
                    required = $profileRequired
                    runtime = $profileRuntime
                    profile = $planModel
                    health_url = $healthProbeUrl
                    catalog_url = $catalogProbeUrl
                    response = [ordered]@{
                        required = $profileRequired
                        runtime = $profileRuntime
                        health_reachable = $false
                        catalog_reachable = $false
                        required_endpoint = $profileRequired
                        command = $commandForDebug
                    }
                }
                continue
            }

            try {
                $shouldStartCommand = $PSCmdlet.ShouldProcess("$ModelBackendHost`:$port", "Start command backend for $target")
                if (-not $shouldStartCommand) {
                    $results += [pscustomobject][ordered]@{
                        device = $target
                        port = $port
                        started = $false
                        action = "whatif"
                        launch_mode = "command"
                        host = $ModelBackendHost
                        model = $plannedModel
                        required = $profileRequired
                        runtime = $profileRuntime
                        profile = $planModel
                        health_url = $healthProbeUrl
                        catalog_url = $catalogProbeUrl
                        response = [ordered]@{
                            required = $profileRequired
                            runtime = $profileRuntime
                            health_reachable = $false
                            catalog_reachable = $false
                            required_endpoint = $profileRequired
                            command = $commandForDebug
                        }
                    }
                    continue
                }

                $resolvedCommandWorkDir = ""
                if (-not [string]::IsNullOrWhiteSpace($profileStartWorkingDirectory)) {
                    $resolvedCommandWorkDir = _Resolve-FilePath -Path $profileStartWorkingDirectory -Base (Get-Location).Path
                }
                $logPaths = _New-HarnessLogPaths -Prefix "model-command-$target-$port"
                $stdoutPath = [string]$logPaths.stdout_log
                $stderrPath = [string]$logPaths.stderr_log
                $commandProcess = Start-Process -FilePath $commandExecutable -ArgumentList $commandArgs -PassThru -NoNewWindow `
                    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WorkingDirectory $resolvedCommandWorkDir
                if ($null -eq $commandProcess) {
                    throw "Failed to start command backend process."
                }
                if ($commandProcess.HasExited) {
                    $startupError = ""
                    if (Test-Path $stderrPath) {
                        $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
                    }
                    throw "Command backend process exited during startup. $startupError"
                }
                if (-not (_Wait-HttpReady -HealthUrl $healthProbeUrl -TimeoutSeconds $WaitSeconds)) {
                    $startupError = ""
                    if (Test-Path $stderrPath) {
                        $startupError = (Get-Content $stderrPath -Raw -ErrorAction SilentlyContinue).Trim()
                    }
                    Stop-Process -Id $commandProcess.Id -Force -ErrorAction SilentlyContinue
                    throw "Command backend at $healthProbeUrl did not become ready after ${WaitSeconds}s. $startupError"
                }
                $healthReachable = $false
                $catalogReachable = $false
                try {
                    $healthReachable = _Wait-HttpReady -HealthUrl $healthProbeUrl -TimeoutSeconds $WaitSeconds
                    if ($healthReachable -and -not [string]::IsNullOrWhiteSpace($catalogProbeUrl)) {
                        $catalogReachable = _Wait-HttpReady -HealthUrl $catalogProbeUrl -TimeoutSeconds $WaitSeconds
                    }
                } catch {
                    $healthReachable = $false
                    $catalogReachable = $false
                }
                if (-not ($healthReachable -and $catalogReachable)) {
                    throw "Command backend reachable check failed at health='$healthProbeUrl' or catalog='$catalogProbeUrl'."
                }

                $entryKey = "model-backend|$ModelBackendHost|$port"
                $existingCommandSessions = _Prune-StaleModelSessions -Sessions (_Read-HarnessModelBackendSessions)
                $entry = @{
                    key = $entryKey
                    mode = "model_backend"
                    process_id = $commandProcess.Id
                    host = $ModelBackendHost
                    port = $port
                    model = $plannedModel
                    requested_device = $target
                    requested_backend = $Backend
                    configured_backend = $Backend
                    generation_backend = "command"
                    model_source = $null
                    model_source_type = $null
                    model_artifact_format = $null
                    provider_store = $null
                    manifest_path = $null
                    local_model_loaded = $false
                    model_source_present = $false
                    model_load_attempted = $false
                    model_load_succeeded = $false
                    last_load_error = $null
                    last_generation_error = $null
                    template_applied = $false
                    fallback_active = $false
                    allow_fallback = $false
                    provider_warning = $null
                    python_path = $null
                    command = $commandForDebug
                    health_url = $healthProbeUrl
                    started_utc = (Get-Date).ToUniversalTime().ToString("o")
                }
                $sessions = @($existingCommandSessions | Where-Object { $_.key -ne $entryKey })
                $sessions += $entry
                _Write-HarnessModelBackendSessions -Sessions $sessions

                $results += [pscustomobject][ordered]@{
                    device = $target
                    port = $port
                    started = $true
                    action = "command_ready"
                    launch_mode = "command"
                    host = $ModelBackendHost
                    model = $plannedModel
                    required = $profileRequired
                    runtime = $profileRuntime
                    profile = $planModel
                    health_url = $healthProbeUrl
                    catalog_url = $catalogProbeUrl
                    response = [ordered]@{
                        required = $profileRequired
                        runtime = $profileRuntime
                        health_reachable = $healthReachable
                        catalog_reachable = $catalogReachable
                        required_endpoint = $profileRequired
                        command = $commandForDebug
                        command_workdir = $resolvedCommandWorkDir
                    }
                }
            } catch {
                if ($commandProcess -and $commandProcess.Id -gt 0) {
                    Stop-Process -Id $commandProcess.Id -Force -ErrorAction SilentlyContinue
                }
                $backendStartError = "[${target}:$port] $($_.Exception.Message)"
                $errors += $backendStartError
                if ($profileRequired) {
                    $requiredFailures += $backendStartError
                }
                $results += [pscustomobject][ordered]@{
                    device = $target
                    port = $port
                    started = $false
                    action = "start_error"
                    launch_mode = "command"
                    host = $ModelBackendHost
                    model = $plannedModel
                    required = $profileRequired
                    runtime = $profileRuntime
                    profile = $planModel
                    response = [ordered]@{
                        required = $profileRequired
                        runtime = $profileRuntime
                        required_endpoint = $profileRequired
                        error = $_.Exception.Message
                        plan_backend = $planModel
                        command = $commandForDebug
                    }
                }
            }
            continue
        }

        try {
            $single = Start-HarnessOwnLLMBackend `
                -ModelBackendHost $ModelBackendHost `
                -ModelBackendPort $port `
                -Model $plannedModel `
                -ModelPath $deviceModelPath `
                -ModelsRoot $deviceModelsRoot `
                -ExtraModel $ExtraModelByDevice[$target] `
                -Backend $Backend `
                -Device $target `
                -MaxNewTokens $MaxNewTokens `
                -LlamaCppContext $LlamaCppContext `
                -LlamaCppGpuLayers $LlamaCppGpuLayers `
                -LlamaCppThreads $LlamaCppThreads `
                -LocalOnly:$LocalOnly.IsPresent `
                -AllowFallback:$AllowFallback.IsPresent `
                -WaitSeconds $WaitSeconds `
                -PythonPath $PythonPath `
                -DryRun:$DryRun.IsPresent
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                started = [bool]$single.started
                action = if ($single.PSObject.Properties.Name -contains "action") { [string]$single.action } else { "started" }
                launch_mode = "local"
                host = $single.host
                model = $deviceModel
                required = $profileRequired
                runtime = $profileRuntime
                profile = $planModel
                health_url = $single.health_url
                response = $single
            }
        } catch {
            $backendStartError = "[${target}:$port] $($_.Exception.Message)"
            $errors += $backendStartError
            if ($profileRequired) {
                $requiredFailures += $backendStartError
            }
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                started = $false
                action = "start_error"
                launch_mode = "local"
                host = $ModelBackendHost
                model = $deviceModel
                required = $profileRequired
                runtime = $profileRuntime
                profile = $planModel
                response = [ordered]@{
                    required = $profileRequired
                    runtime = $profileRuntime
                    required_endpoint = $profileRequired
                    error = $_.Exception.Message
                    plan_backend = $planModel
                }
            }
        }
    }

    $requestedDevicesOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($selectedDevices)) {
        [void]$requestedDevicesOutput.Add($entry)
    }
    $requiredFailuresOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($requiredFailures)) {
        [void]$requiredFailuresOutput.Add($entry)
    }
    $backendsOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($results)) {
        [void]$backendsOutput.Add($entry)
    }
    $errorsOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($errors)) {
        [void]$errorsOutput.Add($entry)
    }

    $summary = [pscustomobject][ordered]@{
        mode = "model_backend_fleet"
        requested_devices = $requestedDevicesOutput
        started_count = @(
            foreach ($entry in $results) {
                if ($entry.PSObject.Properties.Name -contains "started" -and [bool]$entry.started) {
                    $entry
                }
            }
        ).Count
        total = $selectedDevices.Count
        failed_count = $errors.Count
        required_failures = $requiredFailuresOutput
        required_failures_count = $requiredFailures.Count
        dry_run = [bool]$DryRun
        backends = $backendsOutput
        errors = $errorsOutput
    }

    return _Apply-HarnessOutputOptions -InputObject $summary -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

Set-Alias -Name Start-HarnessOwnLLMBackendFleet -Value Start-HarnessModelBackendFleet
Set-Alias -Name Start-HarnessLocalLLMBackendFleet -Value Start-HarnessModelBackendFleet

<#
.SYNOPSIS
Stops one or more local harness-owned model backends for selected hardware devices.
#>
function Stop-HarnessModelBackendFleet {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [string[]]$Devices = @("npu", "gpu", "cpu"),
        [hashtable]$DevicePortMap = @{ npu = 11433; gpu = 11434; hybrid = 13305; cpu = 11435 },
        [switch]$All,
        [switch]$DryRun,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $selectedDevices = @(_Normalize-HardwareDeviceSelection -Devices $Devices)
    $results = @()
    foreach ($target in $selectedDevices) {
        if (-not $DevicePortMap.ContainsKey($target)) {
            $planError = "No port mapping configured for device '$target'."
            if ($target -eq "hybrid") {
                $planError = "No port mapping configured for required device 'hybrid'."
            }
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $null
                status = "not_configured"
                result_reason = "missing_port_map"
                response = $planError
            }
            continue
        }
        $port = [int]$DevicePortMap[$target]
        try {
            $single = Stop-HarnessModelBackend `
                -ModelBackendHost $ModelBackendHost `
                -ModelBackendPort $port `
                -TargetDevice $target `
                -All:$All.IsPresent `
                -DryRun:$DryRun.IsPresent
            $removedReasons = @()
            if (
                $single.PSObject.Properties.Name -contains "removed" -and
                $single.removed -and
                $single.removed.Count -gt 0
            ) {
                foreach ($entry in $single.removed) {
                    if ($entry.PSObject.Properties.Name -contains "result_reason") {
                        $removedReasons += [string]$entry.result_reason
                    }
                }
            }
            if ($removedReasons.Count -eq 0 -and $single.PSObject.Properties.Name -contains "action") {
                if ($single.action -eq "whatif") {
                    $removedReasons = @("would_stop")
                } else {
                    $removedReasons = @("unknown")
                }
            } elseif ($removedReasons.Count -eq 0) {
                $removedReasons = @("not_tracked")
            }
            $statusResult = "requested"
            if ($single.action -eq "whatif") {
                $statusResult = "not_stopped"
            } elseif ($removedReasons -contains "stopped" -or $removedReasons -contains "stopped_fallback") {
                $statusResult = "stopped"
            } elseif ($removedReasons -contains "timed_out") {
                $statusResult = "timed_out"
            } elseif ($removedReasons -contains "not_running") {
                $statusResult = "not_running"
            } elseif ($removedReasons -contains "already_stopped") {
                $statusResult = "already_stopped"
            } else {
                $statusResult = "not_stopped"
            }
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                status = $statusResult
                result_reason = if ($removedReasons -contains "stopped") {
                    "stopped"
                } elseif ($removedReasons -contains "stopped_fallback") {
                    "stopped_fallback"
                } elseif ($removedReasons -contains "already_stopped") {
                    "already_stopped"
                } elseif ($removedReasons -contains "timed_out") {
                    "timed_out"
                } elseif ($removedReasons -contains "would_stop") {
                    "would_stop"
                } elseif ($removedReasons -contains "would_stop_fallback") {
                    "would_stop"
                } else {
                    $removedReasons[0]
                }
                removed_reasons = $removedReasons
                response = $single
            }
        } catch {
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                status = "error"
                result_reason = "stop_error"
                response = $_.Exception.Message
            }
        }
    }

    $requestedDevicesOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($selectedDevices)) {
        [void]$requestedDevicesOutput.Add($entry)
    }
    $backendsOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($results)) {
        [void]$backendsOutput.Add($entry)
    }

    $summary = [pscustomobject][ordered]@{
        mode = "model_backend_fleet_stop"
        requested_devices = $requestedDevicesOutput
        requested_count = $selectedDevices.Count
        removed_count = @($results | Where-Object {
            $_.status -in @("requested", "stopped", "stopped_fallback", "already_stopped", "not_running", "timed_out")
        }).Count
        stopped_count = @($results | Where-Object {
            $_.result_reason -in @("stopped", "stopped_fallback")
        }).Count
        error_count = @($results | Where-Object { $_.status -eq "error" }).Count
        dry_run = [bool]$DryRun
        backends = $backendsOutput
    }
    return _Apply-HarnessOutputOptions -InputObject $summary -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

Set-Alias -Name Stop-HarnessOwnLLMBackendFleet -Value Stop-HarnessModelBackendFleet
Set-Alias -Name Stop-HarnessLocalLLMBackendFleet -Value Stop-HarnessModelBackendFleet

<#
.SYNOPSIS
Gets fleet status for one or more local model backend targets.
.DESCRIPTION
Queries one status object per requested hardware device/port.
#>
function Get-HarnessModelBackendFleetStatus {
    [CmdletBinding()]
    param(
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [string[]]$Devices = @("npu", "gpu", "cpu"),
        [hashtable]$DevicePortMap = @{ npu = 11433; gpu = 11434; hybrid = 13305; cpu = 11435 },
        [switch]$IncludeSession,
        [int]$RequestTimeoutSeconds = 6,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    $selectedDevices = @(_Normalize-HardwareDeviceSelection -Devices $Devices)
    $results = @()
    foreach ($target in $selectedDevices) {
        if (-not $DevicePortMap.ContainsKey($target)) {
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $null
                error = "No port mapping configured for device '$target'."
                status = $null
            }
            continue
        }
        $port = [int]$DevicePortMap[$target]
        try {
            $status = Get-HarnessModelBackendStatus `
                -ModelBackendHost $ModelBackendHost `
                -ModelBackendPort $port `
                -RequestTimeoutSeconds $RequestTimeoutSeconds `
                -IncludeSession:$IncludeSession.IsPresent `
                -Property $Property `
                -ExpandProperty $ExpandProperty `
                -AsJson:$false
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                status = $status
            }
        } catch {
            $results += [pscustomobject][ordered]@{
                device = $target
                port = $port
                error = $_.Exception.Message
            }
        }
    }

    $requestedDevicesOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($selectedDevices)) {
        [void]$requestedDevicesOutput.Add($entry)
    }
    $backendsOutput = [System.Collections.ArrayList]::new()
    foreach ($entry in @($results)) {
        [void]$backendsOutput.Add($entry)
    }

    $summary = [pscustomobject][ordered]@{
        mode = "model_backend_fleet_status"
        requested_devices = $requestedDevicesOutput
        count = $results.Count
        reachable = @($results | Where-Object { $_.status -and $_.status.health_reachable }).Count
        backends = $backendsOutput
    }
    return _Apply-HarnessOutputOptions -InputObject $summary -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
}

Set-Alias -Name Get-HarnessOwnLLMBackendFleetStatus -Value Get-HarnessModelBackendFleetStatus
Set-Alias -Name Get-HarnessLocalLLMBackendFleetStatus -Value Get-HarnessModelBackendFleetStatus

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
        [string]$TargetDevice = "",
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

    $targeted = @()
    $normalizedTargetDevice = [string]$TargetDevice.ToLowerInvariant()
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
        if (-not [string]::IsNullOrWhiteSpace($normalizedTargetDevice)) {
            $entryDevice = if ($entry.PSObject.Properties.Name -contains "requested_device") {
                [string]$entry.requested_device
            } else {
                ""
            }
            if (-not [string]::IsNullOrWhiteSpace($entryDevice) -and $entryDevice.ToLowerInvariant() -ne $normalizedTargetDevice) {
                $remaining += $entry
                continue
            }
        }

        $targeted += $entry
    }

    if ($targeted.Count -gt 0) {
        foreach ($entry in $targeted) {
            $entryResult = [ordered]@{}
            foreach ($name in $entry.PSObject.Properties.Name) {
                $entryResult[$name] = $entry.$name
            }
            $reason = "not_tracked"
            $entryHasPid = $entry.PSObject.Properties.Name -contains "process_id"
            $pid = if ($entryHasPid) { [int]$entry.process_id } else { 0 }
            if ($pid -gt 0) {
                $running = [bool](Get-Process -Id $pid -ErrorAction SilentlyContinue)
                if ($running) {
                    if ($DryRun -or -not $shouldStop) {
                        $reason = "would_stop"
                    } else {
                        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                        Start-Sleep -Milliseconds 250
                        $stillRunning = [bool](Get-Process -Id $pid -ErrorAction SilentlyContinue)
                        if ($stillRunning) {
                            $reason = "timed_out"
                        } else {
                            $reason = "stopped"
                        }
                    }
                } else {
                    $reason = "already_stopped"
                }
            } else {
                $reason = if ($DryRun -or -not $shouldStop) { "already_stopped" } else { "not_running"}
            }
            $entryResult["result_reason"] = $reason
            if ($reason -eq "timed_out" -and $shouldStop) {
                $remaining += $entry
            }
            $removed += [pscustomobject]$entryResult
        }
    } else {
        $fallbackProcesses = @()
        $fallbackPidSet = @{}
        foreach ($listener in _Get-ListeningProcesses -Port $ModelBackendPort) {
            if (-not ($listener.PSObject.Properties.Name -contains "process_id")) {
                continue
            }
            if ([int]$listener.process_id -le 0) {
                continue
            }
            $commandLine = _Get-ProcessCommandLine -ProcessId $listener.process_id
            if ([string]::IsNullOrWhiteSpace($commandLine)) {
                continue
            }
            if ($commandLine -notmatch "harness\.local_model_provider" -and $commandLine -notmatch "local_model_provider\.py") {
                continue
            }
            $fallbackPid = [int]$listener.process_id
            if ($fallbackPidSet.ContainsKey($fallbackPid)) {
                continue
            }
            $fallbackPidSet[$fallbackPid] = $true
            $fallbackProcesses += $fallbackPid
        }
        if ($fallbackProcesses.Count -eq 0) {
            $fallbackProcesses = @()
        } else {
            $fallbackProcesses = @($fallbackProcesses | Sort-Object -Unique)
        }

        if ($fallbackProcesses.Count -gt 0) {
            foreach ($fallbackPid in $fallbackProcesses) {
                $fallbackReason = if ($DryRun -or -not $shouldStop) {
                    "would_stop_fallback"
                } else {
                    Stop-Process -Id $fallbackPid -Force -ErrorAction SilentlyContinue
                    Start-Sleep -Milliseconds 250
                    if ([bool](Get-Process -Id $fallbackPid -ErrorAction SilentlyContinue)) {
                        "timed_out"
                    } else {
                        "stopped_fallback"
                    }
                }

                $removed += [pscustomobject]@{
                    mode = "model_backend"
                    key = $targetKey
                    host = $ModelBackendHost
                    port = $ModelBackendPort
                    process_id = $fallbackPid
                    command = _Get-ProcessCommandLine -ProcessId $fallbackPid
                    result_reason = $fallbackReason
                    result_backend = "fallback_listener"
                }
            }
        } else {
            $removed += [pscustomobject]@{
                mode = "model_backend"
                key = $targetKey
                host = $ModelBackendHost
                port = $ModelBackendPort
                process_id = 0
                command = ""
                result_reason = "not_tracked"
            }
        }
    }

    if (-not ($DryRun -or -not $shouldStop)) {
        _Write-HarnessModelBackendSessions -Sessions $remaining
    }

    if ($DryRun -or -not $shouldStop) {
        $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count") -Property @{
            mode = "model_backend"
            action = if ($DryRun) { "stopped" } else { "whatif" }
            removed_count = $removed.Count
            stopped_count = @($removed | Where-Object { $_.result_reason -in @("stopped", "already_stopped", "stopped_fallback") }).Count
            removed = $removed
        }
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    $result = _New-HarnessObject -TypeName "Harness.ModelBackend.StopResult" -DefaultDisplayPropertySet @("mode", "action", "removed_count") -Property @{
        mode = "model_backend"
        action = "stopped"
        removed_count = $removed.Count
        stopped_count = @($removed | Where-Object { $_.result_reason -in @("stopped", "stopped_fallback") }).Count
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
        $healthResponse = Invoke-WebRequest -UseBasicParsing -Uri $baseUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
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
        $catalogResponse = Invoke-WebRequest -UseBasicParsing -Uri $modelsUrl -Method Get -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
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
        [switch]$AutoResolveContainerProfile,
        [string]$ComposeFile = "docker-compose.nvidia.yaml",
        [string]$EnvFile = "",
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [int]$ModelBackendPort = $Script:HarnessModelBackendDefaultPort,
        [string[]]$ModelBackendDevices = @("npu", "gpu", "cpu"),
        [hashtable]$ModelBackendDevicePortMap = @{ npu = 11433; gpu = 11434; hybrid = 13305; cpu = 11435 },
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
    $resolvedConfigPath = _Resolve-FilePath -Path $Config -Base (Get-Location).Path
    $resolvedModelBackendDevices = @($ModelBackendDevices)
    if (-not $PSBoundParameters.ContainsKey("ModelBackendDevices")) {
        try {
            $requiredConfigDevices = _Get-RequiredModelBackendDevicesFromConfig -ConfigPath $resolvedConfigPath
            if ($requiredConfigDevices.Count -gt 0) {
                $resolvedModelBackendDevices = @($resolvedModelBackendDevices + $requiredConfigDevices)
            }
        } catch {
            Write-Verbose "Could not read required devices from config $resolvedConfigPath while resolving stop targets: $($_.Exception.Message)"
        }
    }

    if (-not $SkipRuntimeBackend) {
        try {
            $runtimeResult = Stop-HarnessBackend `
                -ExecutionMode $ExecutionMode `
                -Config $Config `
                -ServerHost $ServerHost `
                -Port $Port `
                -ContainerProfile $ContainerProfile `
                -AutoResolveContainerProfile:$AutoResolveContainerProfile.IsPresent `
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
            if ($PSBoundParameters.ContainsKey("ModelBackendDevices")) {
                $modelResult = Stop-HarnessModelBackendFleet `
                    -ModelBackendHost $ModelBackendHost `
                    -Devices @(_Normalize-HardwareDeviceSelection -Devices $resolvedModelBackendDevices) `
                    -DevicePortMap $ModelBackendDevicePortMap `
                    -All:$All.IsPresent `
                    -DryRun:$DryRun.IsPresent
            } elseif ($PSBoundParameters.ContainsKey("ModelBackendPort")) {
                $modelResult = Stop-HarnessModelBackend `
                    -ModelBackendHost $ModelBackendHost `
                    -ModelBackendPort $ModelBackendPort `
                    -All:$All.IsPresent `
                    -DryRun:$DryRun.IsPresent
            } else {
                $modelResult = Stop-HarnessModelBackendFleet `
                    -ModelBackendHost $ModelBackendHost `
                    -Devices @(_Normalize-HardwareDeviceSelection -Devices $resolvedModelBackendDevices) `
                    -DevicePortMap $ModelBackendDevicePortMap `
                    -All:$All.IsPresent `
                    -DryRun:$DryRun.IsPresent
            }
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
            errors = [object[]]@($errors)
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
Runs one-shot prompt execution using a full local stack.

.DESCRIPTION
Starts model backend(s) and runtime together, executes one runtime prompt, and optionally keeps the stack alive.
Useful for a true one-command start->run->cleanup workflow.
#>
function Invoke-HarnessOneShotStack {
    [CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
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
        [ValidateSet("local", "containerized")]
        [string]$ExecutionMode = "local",
        [string]$ContainerProfile = "",
        [switch]$AutoResolveContainerProfile,
        [string]$ComposeFile = "docker-compose.nvidia.yaml",
        [string]$EnvFile = "",
        [int]$RuntimeWaitSeconds = 30,

        [string]$Model = "",
        [string]$ModelBackendHost = $Script:HarnessModelBackendDefaultHost,
        [string[]]$ModelBackendDevices = @("npu", "gpu", "cpu"),
        [hashtable]$ModelBackendDevicePortMap = @{ npu = 11433; gpu = 11434; hybrid = 13305; cpu = 11435 },
        [hashtable]$ModelByDevice = @{},
        [string]$ModelPath = "",
        [hashtable]$ModelPathByDevice = @{},
        [string]$ModelsRoot = "",
        [hashtable]$ModelsRootByDevice = @{},
        [string[]]$ExtraModel,
        [hashtable]$ExtraModelByDevice = @{},
        [ValidateSet("auto", "fallback", "transformers", "llamacpp", "llama_cpp", "llama-cpp")]
        [string]$Backend = "auto",
        [string]$Device = "cpu",
        [int]$MaxNewTokens = 256,
        [int]$LlamaCppContext = 4096,
        [int]$LlamaCppGpuLayers = 0,
        [int]$LlamaCppThreads = 4,
        [switch]$LocalOnly,
        [switch]$AllowFallback,
        [int]$ModelBackendWaitSeconds = 30,
        [string]$PythonPath = "",

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
        [switch]$SkipRuntimeBackend,
        [switch]$SkipModelBackend,
        [switch]$KeepStack,
        [switch]$AnswerOnly,
        [switch]$DryRun,
        [switch]$PlanOnly,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    if ($Mode -ne "runtime") {
        $demoParams = @{
            Mode = $Mode
            Question = $Question
            Config = $Config
            ServerHost = $ServerHost
            Port = $Port
            Model = $Model
            StartupTimeoutSeconds = $StartupTimeoutSeconds
            RequestTimeoutSeconds = $RequestTimeoutSeconds
            PythonPath = $PythonPath
            RequireEvidence = $RequireEvidence.IsPresent
            EnableAdvancedRouter = $EnableAdvancedRouter.IsPresent
            NoNetwork = $NoNetwork.IsPresent
            SkipBackendCheck = $SkipBackendCheck.IsPresent
            UseExistingServer = $UseExistingServer.IsPresent
            DryRun = $DryRun.IsPresent
            AnswerOnly = $AnswerOnly.IsPresent
            Property = $Property
            ExpandProperty = $ExpandProperty
            AsJson = $AsJson.IsPresent
            JsonDepth = $JsonDepth
        }
        if (-not [string]::IsNullOrWhiteSpace($FeatureLevel)) {
            $demoParams["FeatureLevel"] = $FeatureLevel
        }
        if (-not [string]::IsNullOrWhiteSpace($ToolSandbox)) {
            $demoParams["ToolSandbox"] = $ToolSandbox
        }
        return Invoke-HarnessOneShot @demoParams
    }

    $resolvedConfigPath = _Resolve-FilePath -Path $Config -Base (Get-Location).Path
    $modelBackendProfiles = @{}
    try {
        $modelBackendProfiles = _Get-ModelBackendProfilesFromConfig -ConfigPath $resolvedConfigPath
    } catch {
        Write-Verbose ("Could not read model backend profiles from config {0}: {1}" -f $resolvedConfigPath, $_.Exception.Message)
    }
    $configuredDevices = @($ModelBackendDevices)
    if (-not $PSBoundParameters.ContainsKey("ModelBackendDevices")) {
        try {
            $requiredConfigDevices = _Get-RequiredModelBackendDevicesFromConfig -ConfigPath $resolvedConfigPath
            if ($requiredConfigDevices.Count -gt 0) {
                Write-Verbose "Auto-augmenting model-backend devices from required config entries: $($requiredConfigDevices -join ', ')"
                $configuredDevices = @($configuredDevices + $requiredConfigDevices)
            }
        } catch {
                Write-Verbose ("Could not read required devices from config {0}: {1}" -f $resolvedConfigPath, $_.Exception.Message)
        }
    }

    $resolvedModel = if ([string]::IsNullOrWhiteSpace($Model)) { $Script:HarnessModelBackendDefaultModel } else { $Model }
    $normalizedModelBackendDevices = @(_Normalize-HardwareDeviceSelection -Devices $configuredDevices)
    if (($normalizedModelBackendDevices -contains "hybrid") -and ($modelBackendProfiles.Count -eq 0 -or -not $modelBackendProfiles.ContainsKey("hybrid"))) {
        $expectedHybridPort = if ($ModelBackendDevicePortMap.ContainsKey("hybrid")) { [int]$ModelBackendDevicePortMap["hybrid"] } else { 13305 }
        throw ("Hybrid backend requested (directly or via required config) but no profile is registered for device 'hybrid'. Add backend device id 'hybrid' to {0} with a reachable endpoint (for example base_url: http://127.0.0.1:{1}/api/v1, health_endpoint: /api/v1/health) or start via explicit hybrid backend first." -f $resolvedConfigPath, $expectedHybridPort)
    }
    $modelBackendRequiredByDevice = @{}
    $modelBackendRuntimeByDevice = @{}
    $modelBackendProfileByDevice = @{}
    $modelBackendPlan = @(
        foreach ($target in $normalizedModelBackendDevices) {
            $profile = if ($modelBackendProfiles.ContainsKey($target)) { $modelBackendProfiles[$target] } else { @{} }
            $required = if ($profile.ContainsKey("required")) { [bool]$profile.required } else { $true }
            $runtime = if ($profile.ContainsKey("runtime")) { [string]$profile.runtime } else { "" }
            $modelBackendRequiredByDevice[$target] = $required
            $modelBackendRuntimeByDevice[$target] = $runtime
            $modelBackendProfileByDevice[$target] = $profile
            if (-not $ModelBackendDevicePortMap.ContainsKey($target)) {
                [ordered]@{
                    device = $target
                    port = $null
                    status = "misconfigured"
                    reason = if ($required) { "Required port mapping missing for device '$target'." } else { "Optional port mapping missing for device '$target'." }
                    required = $required
                    planned_model = if ($ModelByDevice.ContainsKey($target)) { [string]$ModelByDevice[$target] } else { $resolvedModel }
                    runtime = $runtime
                    health_url = $null
                }
                continue
            }
            $targetPort = [int]$ModelBackendDevicePortMap[$target]
            [ordered]@{
                device = $target
                port = $targetPort
                status = "planned"
                reason = "not_started"
                required = $required
                planned_model = if ($ModelByDevice.ContainsKey($target)) { [string]$ModelByDevice[$target] } else { $resolvedModel }
                runtime = $runtime
                health_url = _HealthUrl -TargetHost $ModelBackendHost -Port $targetPort
            }
        }
    )
    $missingPortTargets = @()
    foreach ($entry in $normalizedModelBackendDevices) {
        if (-not $ModelBackendDevicePortMap.ContainsKey($entry) -and ($modelBackendRequiredByDevice[$entry])) {
            $missingPortTargets += $entry
        }
    }
    if ($missingPortTargets.Count -gt 0) {
        $missingJoin = ($missingPortTargets | Select-Object -Unique) -join ", "
        $hybridHint = if ($missingJoin -match "hybrid") {
            " For hybrid, add '-ModelBackendDevicePortMap @{`"hybrid`" = 13305}' and ensure a hybrid service is started on that port."
        } else {
            ""
        }
        throw "Model backend launch map is missing required entries for: $missingJoin. Add these entries to -ModelBackendDevicePortMap.$hybridHint"
    }

    if ($VerbosePreference -ne [System.Management.Automation.ActionPreference]::SilentlyContinue) {
        $planTable = @(
            foreach ($entry in $modelBackendPlan) {
                [pscustomobject]@{
                    Device = $entry.device
                    Port = $entry.port
                    Required = $entry.required
                    Runtime = $entry.runtime
                    PlannedModel = $entry.planned_model
                    Health = $entry.health_url
                    Status = $entry.status
                    Reason = $entry.reason
                }
            }
        )
        $planHeader = "Model backend preflight plan for config '$Config' using devices: $($normalizedModelBackendDevices -join ', ')"
        Write-Verbose $planHeader
        Write-Verbose ($planTable | Format-Table | Out-String)
    }

    if ($PlanOnly) {
        $result = _New-HarnessObject -TypeName "Harness.OneShot.Stack" -DefaultDisplayPropertySet @(
            "action",
            "mode",
            "runtime_config",
            "model_backend_devices"
        ) -Property ([ordered]@{
            action = "plan_only"
            mode = "runtime_stack"
            question = $Question
            runtime_config = $Config
            runtime = $null
            keep_stack = $true
            model_backend_started = $false
            model_backend_ready = $false
            model_backend_devices = @($normalizedModelBackendDevices)
            model_backend_plan = $modelBackendPlan
            model_backend_required_by_device = $modelBackendRequiredByDevice
            model_backend = $null
            one_shot = $null
        }) 
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    if (-not $PSCmdlet.ShouldProcess("$ServerHost`:$Port", "Run one-shot stack")) {
        $result = _New-HarnessObject -TypeName "Harness.OneShot.Stack" -DefaultDisplayPropertySet @(
            "action",
            "mode",
            "runtime_started",
            "model_backend_started",
            "keep_stack",
            "runtime_config",
            "model_backend_devices"
        ) -Property ([ordered]@{
            action = "whatif"
            mode = "runtime_stack"
            question = $Question
            runtime_started = $false
            model_backend_started = $false
            keep_stack = [bool]$KeepStack.IsPresent
            runtime_config = $Config
            model_backend_devices = @($normalizedModelBackendDevices)
            one_shot = $null
            runtime = $null
            model_backend = $null
            model_backend_plan = $modelBackendPlan
        })
        return _Apply-HarnessOutputOptions -InputObject $result -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    }

    $runtimeResult = $null
    $modelResult = $null
    $modelBackendPreflight = $null
     $runtimeStarted = $false
     $runtimeStopProfile = ""
     $modelBackendStarted = $false
     $modelBackendAnyStarted = $false
     $modelBackendReady = $false

    try {
        if (-not $SkipModelBackend) {
            $modelResult = Start-HarnessOwnLLMBackendFleet `
                -ModelBackendHost $ModelBackendHost `
                -Devices $normalizedModelBackendDevices `
                -DevicePortMap $ModelBackendDevicePortMap `
                -DeviceBackendProfiles $modelBackendProfiles `
                -Model $resolvedModel `
                -ModelByDevice $ModelByDevice `
                -ModelPath $ModelPath `
                -ModelPathByDevice $ModelPathByDevice `
                -ModelsRoot $ModelsRoot `
                -ModelsRootByDevice $ModelsRootByDevice `
                -ExtraModel $ExtraModel `
                -ExtraModelByDevice $ExtraModelByDevice `
                -Backend $Backend `
                -MaxNewTokens $MaxNewTokens `
                -LlamaCppContext $LlamaCppContext `
                -LlamaCppGpuLayers $LlamaCppGpuLayers `
                -LlamaCppThreads $LlamaCppThreads `
                -LocalOnly:$LocalOnly.IsPresent `
                -AllowFallback:$AllowFallback.IsPresent `
                -WaitSeconds $ModelBackendWaitSeconds `
                -PythonPath $PythonPath `
                -DryRun:$DryRun.IsPresent
            $modelStartedCount = _Get-HarnessNamedProperty -InputObject $modelResult -Names @("started_count")
            $modelFailedCount = _Get-HarnessNamedProperty -InputObject $modelResult -Names @("failed_count")
            $modelBackendsValue = _Get-HarnessNamedProperty -InputObject $modelResult -Names @("backends")
            $requiredStartFailures = @()
            $requiredFailuresValue = _Get-HarnessNamedProperty -InputObject $modelResult -Names @("required_failures")
            if ($null -ne $requiredFailuresValue) {
                $requiredStartFailures = @($requiredFailuresValue)
            }
            $modelBackendCount = if ($null -eq $modelBackendsValue) { 0 } else { @($modelBackendsValue).Count }
            $requiredBackendDevices = @(
                foreach ($entry in $normalizedModelBackendDevices) {
                    if ($modelBackendRequiredByDevice.ContainsKey($entry) -and [bool]$modelBackendRequiredByDevice[$entry]) {
                        $entry
                    }
                }
            )
            $requiredBackendCount = $requiredBackendDevices.Count
            $requiredStartedCount = 0
            if ($null -ne $modelBackendsValue) {
                foreach ($entry in @($modelBackendsValue)) {
                    if ($null -eq $entry -or ($entry.PSObject.Properties.Name -notcontains "device")) {
                        continue
                    }
                    $device = [string]$entry.device
                    $isRequired = if ($modelBackendRequiredByDevice.ContainsKey($device)) { [bool]$modelBackendRequiredByDevice[$device] } else { $true }
                    if (-not $isRequired) {
                        continue
                    }
                    $isStarted = if ($entry.PSObject.Properties.Name -contains "started") { [bool]$entry.started } else { $false }
                    if ($isStarted) {
                        $requiredStartedCount += 1
                    }
                }
            }
            if ($requiredStartFailures.Count -gt 0 -and -not $DryRun) {
                throw "Model backend startup required failures: $($requiredStartFailures -join '; ')"
            }
            $modelBackendStarted = $requiredBackendCount -eq 0 -or $requiredStartedCount -eq $requiredBackendCount
            $modelBackendAnyStarted = [bool]($modelStartedCount -gt 0)

            $modelBackendPreflight = Get-HarnessModelBackendFleetStatus `
                -ModelBackendHost $ModelBackendHost `
                -Devices $normalizedModelBackendDevices `
                -DevicePortMap $ModelBackendDevicePortMap `
                -RequestTimeoutSeconds $RuntimeWaitSeconds `
                -IncludeSession:$true

            $failedRequiredBackends = @()
            if ($modelBackendPreflight -and $modelBackendPreflight.PSObject.Properties.Name -contains "backends") {
                foreach ($entry in $modelBackendPreflight.backends) {
                    if ($null -eq $entry -or ($entry.PSObject.Properties.Name -notcontains "device")) {
                        continue
                    }
                    $device = [string]$entry.device
                    $deviceRequired = if ($modelBackendRequiredByDevice.ContainsKey($device)) { [bool]$modelBackendRequiredByDevice[$device] } else { $true }
                    if ($entry.PSObject.Properties.Name -contains "status") {
                        $statusObj = $entry.status
                    } else {
                        $statusObj = $null
                    }
                    $healthOk = [bool](
                        $statusObj -and
                        $statusObj.PSObject.Properties.Name -contains "health_reachable" -and
                        [bool]$statusObj.health_reachable
                    )
                    $errorText = if ($entry.PSObject.Properties.Name -contains "error") { [string]$entry.error } else { "" }
                    if (-not $healthOk -and $deviceRequired) {
                        $failedRequiredBackends += [ordered]@{
                            device = $device
                            port = if ($entry.PSObject.Properties.Name -contains "port") { [int]$entry.port } else { $null }
                            reason = if (-not [string]::IsNullOrWhiteSpace($errorText)) { $errorText } else { "Backend health not reachable." }
                        }
                    }
                }
            } else {
                $failedRequiredBackends += [ordered]@{
                    device = "unknown"
                    reason = "Unable to read model backend fleet status after startup."
                }
            }
            $modelBackendReady = $requiredBackendCount -eq 0 -or ($failedRequiredBackends.Count -eq 0 -and $requiredStartedCount -ge $requiredBackendCount)

            if ($failedRequiredBackends.Count -gt 0 -and -not $DryRun) {
                $summary = @(
                    $failedRequiredBackends | ForEach-Object {
                        "{0}:{1}" -f $_.device, $_.reason
                    }
                ) -join "; "
                throw "Model backend preflight failed before runtime start. $summary"
            }
            if (-not $modelBackendStarted -and -not $DryRun -and $requiredBackendCount -gt 0) {
                $modelErrorsValue = _Get-HarnessNamedProperty -InputObject $modelResult -Names @("errors")
                $modelErrors = if ($null -eq $modelErrorsValue) { @() } else { @($modelErrorsValue) }
                throw "Required model backend services did not start. errors=$(@($modelErrors) -join '; ')"
            }
        }

        if (-not $SkipRuntimeBackend -and -not $UseExistingServer) {
            $runtimeResult = Start-HarnessBackend `
                -ExecutionMode $ExecutionMode `
                -Config $Config `
                -ServerHost $ServerHost `
                -Port $Port `
                -RuntimeModel $resolvedModel `
                -ContainerProfile $ContainerProfile `
                -ComposeFile $ComposeFile `
                -EnvFile $EnvFile `
                -WaitSeconds $RuntimeWaitSeconds `
                -PythonPath $PythonPath `
                -DryRun:$DryRun.IsPresent
            $runtimeStarted = [bool]$runtimeResult.started
            if ($runtimeResult -and $runtimeResult.PSObject.Properties.Name -contains "profile" -and -not [string]::IsNullOrWhiteSpace([string]$runtimeResult.profile)) {
                $runtimeStopProfile = [string]$runtimeResult.profile
            }
        }

        $callParams = @{
            Mode = "runtime"
            Question = $Question
            Config = $Config
            ServerHost = $ServerHost
            Port = $Port
            Model = $resolvedModel
            StartupTimeoutSeconds = $StartupTimeoutSeconds
            RequestTimeoutSeconds = $RequestTimeoutSeconds
            PythonPath = $PythonPath
            RequireEvidence = $RequireEvidence.IsPresent
            EnableAdvancedRouter = $EnableAdvancedRouter.IsPresent
            NoNetwork = $NoNetwork.IsPresent
            SkipBackendCheck = $SkipBackendCheck.IsPresent
            UseExistingServer = $true
            DryRun = $DryRun.IsPresent
            AnswerOnly = $AnswerOnly.IsPresent
            Property = @()
            ExpandProperty = ""
            AsJson = $false
            JsonDepth = $JsonDepth
        }
        if (-not [string]::IsNullOrWhiteSpace($FeatureLevel)) {
            $callParams["FeatureLevel"] = $FeatureLevel
        }
        if (-not [string]::IsNullOrWhiteSpace($ToolSandbox)) {
            $callParams["ToolSandbox"] = $ToolSandbox
        }
        $result = Invoke-HarnessOneShot @callParams

        $stackResult = _New-HarnessObject -TypeName "Harness.OneShot.Stack" -DefaultDisplayPropertySet @(
            "action",
            "mode",
            "runtime_started",
            "model_backend_started",
            "keep_stack",
            "runtime_config",
            "model_backend_devices"
        ) -Property ([ordered]@{
            action = "completed"
            mode = "runtime_stack"
            question = $Question
            runtime_started = $runtimeStarted
            model_backend_started = $modelBackendStarted
            model_backend_ready = $modelBackendReady
            keep_stack = [bool]$KeepStack.IsPresent
            runtime_config = $Config
            model_backend_devices = @($normalizedModelBackendDevices)
            model_backend_preflight = $modelBackendPreflight
            runtime = $runtimeResult
            model_backend = $modelResult
            one_shot = $result
        })
        return _Apply-HarnessOutputOptions -InputObject $stackResult -Property $Property -ExpandProperty $ExpandProperty -AsJson:$AsJson.IsPresent -JsonDepth $JsonDepth
    } finally {
        if (-not $KeepStack -and -not $DryRun) {
            if ($modelBackendAnyStarted) {
                Stop-HarnessModelBackendFleet `
                    -ModelBackendHost $ModelBackendHost `
                    -Devices $normalizedModelBackendDevices `
                    -DevicePortMap $ModelBackendDevicePortMap `
                    -DryRun:$DryRun.IsPresent
            }
            if ($runtimeStarted -and -not $SkipRuntimeBackend) {
                $resolvedRuntimeStopProfile = if ($ExecutionMode -eq "containerized" -and -not [string]::IsNullOrWhiteSpace($runtimeStopProfile)) {
                    $runtimeStopProfile
                } elseif ($ExecutionMode -eq "containerized" -and [string]::IsNullOrWhiteSpace($runtimeStopProfile) -and -not [string]::IsNullOrWhiteSpace($ContainerProfile)) {
                    $ContainerProfile
                } else {
                    ""
                }
                Stop-HarnessBackend `
                    -ExecutionMode $ExecutionMode `
                    -Config $Config `
                    -ServerHost $ServerHost `
                    -Port $Port `
                    -ContainerProfile $resolvedRuntimeStopProfile `
                    -AutoResolveContainerProfile:$AutoResolveContainerProfile.IsPresent `
                    -ComposeFile $ComposeFile `
                    -EnvFile $EnvFile `
                    -DryRun:$DryRun.IsPresent
            }
        }
    }
}

Set-Alias -Name Invoke-HarnessStackOneShot -Value Invoke-HarnessOneShotStack

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
        [switch]$RunStack,
        [switch]$KeepStack,

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
        [switch]$AutoResolveContainerProfile,
        [Alias("Properties")]
        [string[]]$Property,
        [string]$ExpandProperty = "",
        [switch]$AsJson,
        [int]$JsonDepth = 20
    )

    if ($RunStack) {
        $stackParams = @{
            Mode = $Mode
            Question = $Question
            Config = $Config
            ServerHost = $ServerHost
            Port = $Port
            Model = $Model
            StartupTimeoutSeconds = $StartupTimeoutSeconds
            RequestTimeoutSeconds = $RequestTimeoutSeconds
            PythonPath = $PythonPath
            RequireEvidence = $RequireEvidence.IsPresent
            EnableAdvancedRouter = $EnableAdvancedRouter.IsPresent
            NoNetwork = $NoNetwork.IsPresent
            SkipBackendCheck = $SkipBackendCheck.IsPresent
            UseExistingServer = $UseExistingServer.IsPresent
            KeepStack = $KeepStack.IsPresent
            DryRun = $DryRun.IsPresent
            AnswerOnly = $AnswerOnly.IsPresent
            Property = $Property
            ExpandProperty = $ExpandProperty
            AsJson = $AsJson.IsPresent
            JsonDepth = $JsonDepth
        }
        if (-not [string]::IsNullOrWhiteSpace($FeatureLevel)) {
            $stackParams["FeatureLevel"] = $FeatureLevel
        }
        if (-not [string]::IsNullOrWhiteSpace($ToolSandbox)) {
            $stackParams["ToolSandbox"] = $ToolSandbox
        }
        if ($AutoResolveContainerProfile.IsPresent) {
            $stackParams["AutoResolveContainerProfile"] = $true
        }
        return Invoke-HarnessOneShotStack @stackParams
    }

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

        return $helperOutputText | ConvertFrom-HarnessJson -Depth 20
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
                $response = Invoke-WebRequest -UseBasicParsing -Uri $chatUrl -Method Post -ContentType "application/json" -Body $payload -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
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
            $result = $response.Content | ConvertFrom-HarnessJson -Depth 20
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



