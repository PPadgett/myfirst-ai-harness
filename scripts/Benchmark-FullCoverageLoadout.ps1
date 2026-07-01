<#
.SYNOPSIS
Runs a full coverage hardware loadout benchmark across CPU, GPU, NPU, Hybrid, and agentic_parallel.

.DESCRIPTION
Executes the same complex prompt against each forced execution profile, first with
thinking disabled and then with thinking enabled. The script records latency,
token/context metrics, routing metadata, and agentic controller/worker evidence.

Hybrid NPU+iGPU is required. If a healthy Hybrid backend is not registered in
/health, the script fails before running the benchmark.
#>
param(
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 8080,
    [int]$RequestTimeoutSeconds = 600,
    [ValidateRange(1, 100)]
    [int]$RunsPerCase = 1,
    [string]$Prompt = "Design a production-ready AI harness strategy for a Ryzen AI system that uses CPU control, NPU fast planning, GPU long-form synthesis, Hybrid NPU+iGPU execution, fallback policy, observability, and benchmark acceptance criteria. Include tradeoffs and a final recommendation.",
    [string]$Model = "qwen3:4b",
    [ValidateRange(1, 32768)]
    [int]$MaxTokens = 512,
    [double]$Temperature = 0.2,
    [string]$OutputJsonPath = ".\state\benchmarks\full-coverage-loadout.json",
    [string]$OutputCsvPath = ".\state\benchmarks\full-coverage-loadout.csv"
)

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
$ErrorActionPreference = "Stop"

$healthUrl = "http://$ServerHost`:$ServerPort/health"
$chatUrl = "http://$ServerHost`:$ServerPort/v1/chat/completions"

function Get-OptionalProperty {
    param(
        [object]$InputObject,
        [string]$Name
    )

    if ($null -eq $InputObject -or [string]::IsNullOrWhiteSpace($Name)) {
        return $null
    }
    if ($InputObject -is [System.Collections.IDictionary]) {
        if ($InputObject.Contains($Name)) {
            return $InputObject[$Name]
        }
        return $null
    }
    if ($InputObject.PSObject -and $InputObject.PSObject.Properties.Name -contains $Name) {
        return $InputObject.PSObject.Properties[$Name].Value
    }
    return $null
}

function ConvertTo-CompactJson {
    param([object]$Value)

    if ($null -eq $Value) {
        return ""
    }
    return ($Value | ConvertTo-Json -Depth 20 -Compress)
}

function ConvertTo-SafeString {
    param([object]$Value)

    if ($null -eq $Value) {
        return ""
    }
    return [string]$Value
}

function Estimate-TokenCount {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return 0
    }
    return [int][math]::Ceiling($Text.Length / 4.0)
}

function Add-ThinkingDirective {
    param(
        [string]$Text,
        [ValidateSet("off", "on")]
        [string]$ThinkingMode
    )

    $trimmed = if ($null -eq $Text) { "" } else { $Text.TrimStart() }
    if ($trimmed.StartsWith("/think") -or $trimmed.StartsWith("/no_think")) {
        return $Text
    }
    if ($ThinkingMode -eq "on") {
        return "/think`n$Text"
    }
    return "/no_think`n$Text"
}

function New-ThinkingMessages {
    param(
        [string]$Text,
        [ValidateSet("off", "on")]
        [string]$ThinkingMode
    )

    $trimmed = if ($null -eq $Text) { "" } else { $Text.TrimStart() }
    if ($trimmed.StartsWith("/think") -or $trimmed.StartsWith("/no_think")) {
        return @(@{ role = "user"; content = $Text })
    }

    $directive = if ($ThinkingMode -eq "on") { "/think" } else { "/no_think" }
    return @(
        @{ role = "system"; content = $directive },
        @{ role = "user"; content = $Text }
    )
}

function Ensure-ParentDirectory {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return
    }
    $resolvedPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    $parent = Split-Path -Parent $resolvedPath
    if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Test-BackendHealthy {
    param([object]$Backend)

    $health = (ConvertTo-SafeString (Get-OptionalProperty -InputObject $Backend -Name "health")).ToLowerInvariant()
    $status = (ConvertTo-SafeString (Get-OptionalProperty -InputObject $Backend -Name "status")).ToLowerInvariant()
    $reachable = Get-OptionalProperty -InputObject $Backend -Name "health_reachable"

    if ($health -in @("healthy", "ok", "ready", "true")) {
        return $true
    }
    if ($status -in @("healthy", "ok", "ready", "true")) {
        return $true
    }
    if ($reachable -is [bool] -and $reachable) {
        return $true
    }
    return $false
}

function Test-BackendMatchesDevice {
    param(
        [object]$Backend,
        [ValidateSet("cpu", "gpu", "npu", "hybrid")]
        [string]$Device
    )

    $id = (ConvertTo-SafeString (Get-OptionalProperty -InputObject $Backend -Name "id")).ToLowerInvariant()
    $deviceValue = (ConvertTo-SafeString (Get-OptionalProperty -InputObject $Backend -Name "device")).ToLowerInvariant()
    $deviceMode = (ConvertTo-SafeString (Get-OptionalProperty -InputObject $Backend -Name "device_mode")).ToLowerInvariant()
    $runtime = (ConvertTo-SafeString (Get-OptionalProperty -InputObject $Backend -Name "runtime")).ToLowerInvariant()
    $combined = "$id $deviceValue $deviceMode $runtime"

    switch ($Device) {
        "cpu" { return (($deviceValue -eq "cpu") -or ($id -eq "cpu") -or ($combined -match '\bcpu\b')) }
        "gpu" { return (($deviceValue -eq "gpu") -or ($id -eq "gpu") -or (($combined -match '\bgpu\b') -and ($combined -notmatch 'hybrid'))) }
        "npu" { return (($deviceValue -eq "npu") -or ($id -eq "npu") -or (($combined -match '\bnpu\b') -and ($combined -notmatch 'hybrid|igpu'))) }
        "hybrid" { return ($combined -match 'hybrid|npu\+igpu|npu_igpu|npu-igpu') }
    }
}

function Get-BackendFleet {
    Write-Host "Checking harness health at $healthUrl"
    $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 15
    $ok = Get-OptionalProperty -InputObject $health -Name "ok"
    if ($ok -is [bool] -and -not $ok) {
        throw "Harness health returned ok=false."
    }

    $backendsValue = Get-OptionalProperty -InputObject $health -Name "backends"
    $backends = @($backendsValue)
    if ($backends.Count -eq 0 -or ($backends.Count -eq 1 -and $null -eq $backends[0])) {
        throw "Harness health did not return any backend entries."
    }

    $deviceMap = @{}
    foreach ($backend in $backends) {
        $id = ConvertTo-SafeString (Get-OptionalProperty -InputObject $backend -Name "id")
        if ([string]::IsNullOrWhiteSpace($id)) {
            continue
        }
        $deviceMap[$id] = ConvertTo-SafeString (Get-OptionalProperty -InputObject $backend -Name "device")
    }

    return [ordered]@{
        health = $health
        backends = $backends
        device_map = $deviceMap
    }
}

function Assert-RequiredFleet {
    param([object[]]$Backends)

    foreach ($device in @("cpu", "gpu", "npu", "hybrid")) {
        $matches = @($Backends | Where-Object { (Test-BackendHealthy -Backend $_) -and (Test-BackendMatchesDevice -Backend $_ -Device $device) })
        if ($matches.Count -eq 0) {
            if ($device -eq "hybrid") {
                throw "Required Hybrid NPU+iGPU backend is not registered and healthy. Add a backend with device/id/device_mode indicating hybrid, hybrid_npu_igpu, or npu_igpu before running this benchmark."
            }
            throw "Required $device backend is not registered and healthy."
        }
    }
}

function Get-InferredDevice {
    param(
        [string]$BackendId,
        [System.Collections.IDictionary]$DeviceMap
    )

    if ([string]::IsNullOrWhiteSpace($BackendId)) {
        return ""
    }
    if ($DeviceMap.Contains($BackendId) -and -not [string]::IsNullOrWhiteSpace([string]$DeviceMap[$BackendId])) {
        return [string]$DeviceMap[$BackendId]
    }

    $lower = $BackendId.ToLowerInvariant()
    if ($lower -match 'hybrid|npu\+igpu|npu_igpu|npu-igpu') {
        return "hybrid"
    }
    if ($lower -match '\bcpu\b|cpu') {
        return "cpu"
    }
    if ($lower -match '\bgpu\b|gpu') {
        return "gpu"
    }
    if ($lower -match '\bnpu\b|npu') {
        return "npu"
    }
    return ""
}

function Get-ExecutionMetadata {
    param([object]$Response)

    $empty = [ordered]@{
        backend_id = ""
        fallback_attempted = $false
        selected_via = ""
        wait_ms = $null
        runtime_ms = $null
        ttft_ms = $null
        tokens_per_second = $null
        backend_plan = @()
        fallback_reason = ""
        attempts = @()
        profile = ""
        profile_source = ""
        agentic_parallel = $null
        model_loadout = $null
    }

    if ($null -eq $Response) {
        return $empty
    }

    $execution = Get-OptionalProperty -InputObject $Response -Name "execution"
    $provider = Get-OptionalProperty -InputObject $Response -Name "provider"
    if (-not $execution -and $provider) {
        $execution = Get-OptionalProperty -InputObject $provider -Name "execution"
    }
    if (-not $execution) {
        return $empty
    }

    $attempts = Get-OptionalProperty -InputObject $execution -Name "attempts"
    $backendPlan = Get-OptionalProperty -InputObject $execution -Name "backend_plan"

    return [ordered]@{
        backend_id = ConvertTo-SafeString (Get-OptionalProperty -InputObject $execution -Name "backend_id")
        fallback_attempted = [bool](Get-OptionalProperty -InputObject $execution -Name "fallback_attempted")
        selected_via = ConvertTo-SafeString (Get-OptionalProperty -InputObject $execution -Name "selected_via")
        profile_source = ConvertTo-SafeString (Get-OptionalProperty -InputObject $execution -Name "profile_source")
        wait_ms = Get-OptionalProperty -InputObject $execution -Name "wait_ms"
        runtime_ms = Get-OptionalProperty -InputObject $execution -Name "runtime_ms"
        ttft_ms = Get-OptionalProperty -InputObject $execution -Name "ttft_ms"
        tokens_per_second = Get-OptionalProperty -InputObject $execution -Name "tokens_per_second"
        backend_plan = if ($null -eq $backendPlan) { @() } else { @($backendPlan) }
        fallback_reason = ConvertTo-SafeString (Get-OptionalProperty -InputObject $execution -Name "fallback_reason")
        attempts = if ($null -eq $attempts) { @() } else { @($attempts) }
        profile = ConvertTo-SafeString (Get-OptionalProperty -InputObject $execution -Name "profile")
        agentic_parallel = Get-OptionalProperty -InputObject $execution -Name "agentic_parallel"
        model_loadout = Get-OptionalProperty -InputObject $execution -Name "model_loadout"
    }
}

function Get-UsageMetrics {
    param(
        [object]$Response,
        [string]$PromptText,
        [string]$ResponseText
    )

    $usage = Get-OptionalProperty -InputObject $Response -Name "usage"
    $promptTokens = Get-OptionalProperty -InputObject $usage -Name "prompt_tokens"
    $completionTokens = Get-OptionalProperty -InputObject $usage -Name "completion_tokens"
    $totalTokens = Get-OptionalProperty -InputObject $usage -Name "total_tokens"

    if ($null -eq $promptTokens) {
        $promptTokens = Estimate-TokenCount -Text $PromptText
    }
    if ($null -eq $completionTokens) {
        $completionTokens = Estimate-TokenCount -Text $ResponseText
    }
    if ($null -eq $totalTokens) {
        $totalTokens = [int]$promptTokens + [int]$completionTokens
    }

    return [ordered]@{
        prompt_tokens = [int]$promptTokens
        completion_tokens = [int]$completionTokens
        total_tokens = [int]$totalTokens
        estimated_context_tokens = Estimate-TokenCount -Text $PromptText
    }
}

function Get-ResponseText {
    param([object]$Response)

    $choices = Get-OptionalProperty -InputObject $Response -Name "choices"
    foreach ($choice in @($choices)) {
        $message = Get-OptionalProperty -InputObject $choice -Name "message"
        $content = Get-OptionalProperty -InputObject $message -Name "content"
        if ($null -ne $content) {
            return [string]$content
        }
        $text = Get-OptionalProperty -InputObject $choice -Name "text"
        if ($null -ne $text) {
            return [string]$text
        }
    }
    $answer = Get-OptionalProperty -InputObject $Response -Name "answer"
    if ($null -ne $answer) {
        return [string]$answer
    }
    return ""
}

function Get-AgenticSummary {
    param([object]$AgenticMetadata)

    if ($null -eq $AgenticMetadata) {
        return [ordered]@{
            enabled = $false
            worker_count = 0
            controller_backend = ""
            accepted_results = 0
            rejected_results = 0
            workers_json = ""
        }
    }

    $workers = Get-OptionalProperty -InputObject $AgenticMetadata -Name "workers"
    $workerRows = if ($null -eq $workers) { @() } else { @($workers) }
    $accepted = Get-OptionalProperty -InputObject $AgenticMetadata -Name "accepted_results"
    $rejected = Get-OptionalProperty -InputObject $AgenticMetadata -Name "rejected_results"

    if ($null -eq $accepted) {
        $accepted = 0
        foreach ($worker in $workerRows) {
            $acceptedValue = Get-OptionalProperty -InputObject $worker -Name "accepted"
            if ($acceptedValue -is [bool] -and $acceptedValue) {
                $accepted++
            }
        }
    }
    if ($null -eq $rejected) {
        $rejected = 0
        foreach ($worker in $workerRows) {
            $acceptedValue = Get-OptionalProperty -InputObject $worker -Name "accepted"
            if ($acceptedValue -is [bool] -and -not $acceptedValue) {
                $rejected++
            }
        }
    }

        return [ordered]@{
            enabled = [bool](Get-OptionalProperty -InputObject $AgenticMetadata -Name "enabled")
            worker_count = @($workerRows).Count
            controller_backend = ConvertTo-SafeString (Get-OptionalProperty -InputObject $AgenticMetadata -Name "controller_backend")
            accepted_results = [int]$accepted
            rejected_results = [int]$rejected
            workers_json = ConvertTo-CompactJson -Value $workerRows
        }
}

function Invoke-LoadoutRequest {
    param(
        [hashtable]$CaseDef,
        [object[]]$Messages
    )

    $payload = [ordered]@{
        messages = $Messages
        max_tokens = [int]$MaxTokens
        temperature = $Temperature
        route = "direct"
        execution_profile = [string]$CaseDef.execution_profile
        request_id = "full-coverage-" + [guid]::NewGuid().ToString("N")
    }
    if (-not [string]::IsNullOrWhiteSpace($Model)) {
        $payload.model = $Model
    }

    $start = Get-Date
    try {
        $raw = Invoke-WebRequest -UseBasicParsing -Uri $chatUrl -Method Post -TimeoutSec $RequestTimeoutSeconds -ContentType "application/json" -Body ($payload | ConvertTo-Json -Depth 30) -ErrorAction Stop
        $elapsedMs = [int][math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        $response = $raw.Content | ConvertFrom-HarnessJson -Depth 30
        $status = ConvertTo-SafeString (Get-OptionalProperty -InputObject $response -Name "status")
        if ([string]::IsNullOrWhiteSpace($status)) {
            $status = if ($raw.StatusCode -ge 200 -and $raw.StatusCode -lt 300) { "ok" } else { "http_error" }
        }

        return [ordered]@{
            ok = ($raw.StatusCode -ge 200 -and $raw.StatusCode -lt 300 -and $status -eq "ok")
            status = $status
            status_code = $raw.StatusCode
            elapsed_ms = $elapsedMs
            response = $response
            error = ""
        }
    } catch {
        $elapsedMs = [int][math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        return [ordered]@{
            ok = $false
            status = "exception"
            status_code = $null
            elapsed_ms = $elapsedMs
            response = $null
            error = $_.Exception.Message
        }
    }
}

function Test-ExpectedSelection {
    param(
        [hashtable]$CaseDef,
        [string]$SelectedBackend,
        [string]$SelectedDevice,
        [object]$AgenticMetadata,
        [string]$ExpectedProfile = "",
        [string]$ReportedProfile = "",
        [string]$ProfileSource = ""
    )

    $caseId = [string]$CaseDef.id
    $backendLower = if ($null -eq $SelectedBackend) { "" } else { $SelectedBackend.ToLowerInvariant() }
    $deviceLower = if ($null -eq $SelectedDevice) { "" } else { $SelectedDevice.ToLowerInvariant() }
    if (-not [string]::IsNullOrWhiteSpace($ExpectedProfile) -and -not [string]::IsNullOrWhiteSpace($ReportedProfile)) {
        if ($ExpectedProfile.ToLowerInvariant() -ne $ReportedProfile.ToLowerInvariant()) {
            return "failed: requested execution_profile='$ExpectedProfile' but runtime reported '$ReportedProfile' (source=$ProfileSource)"
        }
    }

    if ($caseId -eq "agentic_parallel") {
        $agentic = Get-AgenticSummary -AgenticMetadata $AgenticMetadata
        $controller = $agentic.controller_backend.ToLowerInvariant()
        if (-not $agentic.enabled) {
            return "failed: agentic_parallel metadata did not report enabled=true"
        }
        if ($agentic.worker_count -lt 2) {
            return "failed: agentic_parallel returned fewer than 2 workers"
        }
        if (($controller -notmatch 'cpu') -and ($backendLower -notmatch 'cpu') -and ($deviceLower -notmatch 'cpu')) {
            return "failed: agentic_parallel did not show a CPU controller"
        }
        return "passed"
    }

    $expectedDevice = [string]$CaseDef.expected_device
    switch ($expectedDevice) {
        "cpu" {
            if ($deviceLower -eq "cpu" -or $backendLower -match 'cpu') { return "passed" }
        }
        "gpu" {
            if (($deviceLower -eq "gpu" -or $backendLower -match 'gpu') -and ($backendLower -notmatch 'hybrid')) { return "passed" }
        }
        "npu" {
            if (($deviceLower -eq "npu" -or $backendLower -match 'npu') -and ($backendLower -notmatch 'hybrid|igpu')) { return "passed" }
        }
        "hybrid" {
            if ($deviceLower -match 'hybrid|npu\+igpu|npu_igpu|npu-igpu' -or $backendLower -match 'hybrid|npu\+igpu|npu_igpu|npu-igpu') { return "passed" }
        }
    }

    return "failed: expected $expectedDevice but selected backend='$SelectedBackend' device='$SelectedDevice'"
}

function Get-Percentile {
    param(
        [double[]]$Samples,
        [double]$Percentile
    )

    if (-not $Samples -or $Samples.Count -eq 0) {
        return $null
    }
    $sorted = @($Samples | Sort-Object)
    $index = [int][math]::Max(0, [math]::Min($sorted.Count - 1, [math]::Floor(($sorted.Count - 1) * ($Percentile / 100.0))))
    return [math]::Round([double]$sorted[$index], 3)
}

function New-Summary {
    param([object[]]$Rows)

    $groups = $Rows | Group-Object -Property thinking_mode, execution_profile
    $summaryRows = @()
    foreach ($group in $groups) {
        $items = @($group.Group)
        $successful = @($items | Where-Object { $_.status -eq "ok" -and $_.validation_status -eq "passed" })
        $latencies = @($successful | ForEach-Object { [double]$_.latency_ms })
        $tps = @($successful | Where-Object { $_.tokens_per_second -gt 0 } | ForEach-Object { [double]$_.tokens_per_second })
        $summaryRows += [ordered]@{
            thinking_mode = ConvertTo-SafeString $items[0].thinking_mode
            execution_profile = ConvertTo-SafeString $items[0].execution_profile
            selected_backends = ((@($items | ForEach-Object { $_.selected_backend }) | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | Select-Object -Unique) -join "|")
            total = @($items).Count
            succeeded = @($successful).Count
            failed = @($items).Count - @($successful).Count
            fallback_count = @($items | Where-Object { $_.fallback_attempted }).Count
            latency_p50_ms = Get-Percentile -Samples $latencies -Percentile 50
            latency_p95_ms = Get-Percentile -Samples $latencies -Percentile 95
            tokens_per_second_p50 = Get-Percentile -Samples $tps -Percentile 50
            tokens_per_second_p95 = Get-Percentile -Samples $tps -Percentile 95
            max_total_tokens = if (@($successful).Count -gt 0) { (@($successful | ForEach-Object { [int]$_.total_tokens }) | Measure-Object -Maximum).Maximum } else { 0 }
        }
    }
    return $summaryRows
}

function Select-Winner {
    param(
        [object[]]$Rows,
        [ValidateSet("latency", "tokens_per_second", "context")]
        [string]$Metric
    )

    $successful = @($Rows | Where-Object { $_.status -eq "ok" -and $_.validation_status -eq "passed" })
    if (@($successful).Count -eq 0) {
        return $null
    }

    if ($Metric -eq "latency") {
        return $successful | Sort-Object -Property latency_ms | Select-Object -First 1
    }
    if ($Metric -eq "tokens_per_second") {
        return $successful | Sort-Object -Property tokens_per_second -Descending | Select-Object -First 1
    }
    return $successful | Sort-Object -Property total_tokens -Descending | Select-Object -First 1
}

$fleet = Get-BackendFleet
Assert-RequiredFleet -Backends $fleet.backends

$loadoutMatrix = @(
    @{ id = "cpu"; label = "CPU-only"; execution_profile = "cpu"; expected_device = "cpu" },
    @{ id = "gpu"; label = "GPU-only"; execution_profile = "gpu"; expected_device = "gpu" },
    @{ id = "npu"; label = "NPU-only"; execution_profile = "npu_only"; expected_device = "npu" },
    @{ id = "hybrid"; label = "Hybrid NPU+iGPU"; execution_profile = "hybrid_npu_igpu"; expected_device = "hybrid" },
    @{ id = "agentic_parallel"; label = "Agentic parallel"; execution_profile = "agentic_parallel"; expected_device = "cpu" }
)
$thinkingModes = @("off", "on")

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss")
$runId = "full-coverage-loadout-$stamp"
$rows = @()

foreach ($thinkingMode in $thinkingModes) {
    foreach ($case in $loadoutMatrix) {
        for ($run = 1; $run -le $RunsPerCase; $run++) {
            $messages = @(New-ThinkingMessages -Text $Prompt -ThinkingMode $thinkingMode)
            $effectivePrompt = (($messages | ForEach-Object { ConvertTo-SafeString (Get-OptionalProperty -InputObject $_ -Name "content") }) -join "`n")
            Write-Host ("Running {0}, thinking={1}, run={2}/{3}" -f $case.id, $thinkingMode, $run, $RunsPerCase)

            $result = Invoke-LoadoutRequest -CaseDef $case -Messages $messages
            $responseText = Get-ResponseText -Response $result.response
            $execution = Get-ExecutionMetadata -Response $result.response
            $usage = Get-UsageMetrics -Response $result.response -PromptText $effectivePrompt -ResponseText $responseText
            $agentic = Get-AgenticSummary -AgenticMetadata $execution.agentic_parallel

            $selectedBackend = $execution.backend_id
            if ([string]::IsNullOrWhiteSpace($selectedBackend)) {
                $selectedBackend = "unknown"
            }
            $selectedDevice = Get-InferredDevice -BackendId $selectedBackend -DeviceMap $fleet.device_map
            $ttftMs = if ($null -ne $execution.ttft_ms) { [int]$execution.ttft_ms } else { [int]$result.elapsed_ms }
            $runtimeMs = if ($null -ne $execution.runtime_ms) { [int]$execution.runtime_ms } else { [int]$result.elapsed_ms }
            $tokensPerSecond = 0.0
            if ($null -ne $execution.tokens_per_second) {
                $tokensPerSecond = [double]$execution.tokens_per_second
            } elseif ($result.elapsed_ms -gt 0 -and $usage.completion_tokens -gt 0) {
                $tokensPerSecond = [math]::Round(($usage.completion_tokens / ([math]::Max(0.001, $result.elapsed_ms / 1000.0))), 3)
            }

            $validationStatus = if ($result.ok) {
                Test-ExpectedSelection `
                    -CaseDef $case `
                    -SelectedBackend $selectedBackend `
                    -SelectedDevice $selectedDevice `
                    -AgenticMetadata $execution.agentic_parallel `
                    -ExpectedProfile $case.execution_profile `
                    -ReportedProfile $execution.profile `
                    -ProfileSource $execution.profile_source
            } else {
                "failed: request did not complete successfully"
            }

            $rows += [pscustomobject][ordered]@{
                benchmark_run_id = $runId
                timestamp = (Get-Date).ToUniversalTime().ToString("o")
                run = $run
                runs_per_case = $RunsPerCase
                loadout_id = [string]$case.id
                loadout_label = [string]$case.label
                thinking_mode = $thinkingMode
                execution_profile = [string]$case.execution_profile
                reported_execution_profile = [string]$execution.profile
                reported_profile_source = [string]$execution.profile_source
                expected_device = [string]$case.expected_device
                selected_backend = $selectedBackend
                selected_device = $selectedDevice
                status = [string]$result.status
                validation_status = $validationStatus
                latency_ms = [int]$result.elapsed_ms
                runtime_ms = $runtimeMs
                ttft_ms = $ttftMs
                wait_ms = $execution.wait_ms
                tokens_per_second = $tokensPerSecond
                prompt_tokens = [int]$usage.prompt_tokens
                completion_tokens = [int]$usage.completion_tokens
                total_tokens = [int]$usage.total_tokens
                estimated_context_tokens = [int]$usage.estimated_context_tokens
                context_tier = if ($usage.estimated_context_tokens -ge 3000) { "near_context_limit" } elseif ($usage.estimated_context_tokens -ge 700) { "long" } else { "complex" }
                max_output_tokens = $MaxTokens
                fallback_attempted = [bool]$execution.fallback_attempted
                fallback_reason = $execution.fallback_reason
                selected_via = $execution.selected_via
                backend_plan = ConvertTo-CompactJson -Value $execution.backend_plan
                attempts = ConvertTo-CompactJson -Value $execution.attempts
                agentic_enabled = [bool]$agentic.enabled
                agentic_workers = [int]$agentic.worker_count
                agentic_controller_backend = $agentic.controller_backend
                accepted_results = [int]$agentic.accepted_results
                rejected_results = [int]$agentic.rejected_results
                agentic_workers_json = $agentic.workers_json
                response_chars = if ($null -eq $responseText) { 0 } else { $responseText.Length }
                response_id = ConvertTo-SafeString (Get-OptionalProperty -InputObject $result.response -Name "id")
                status_code = $result.status_code
                error = $result.error
                model = $Model
                temperature = $Temperature
            }
        }
    }
}

$summary = New-Summary -Rows $rows
$fastest = Select-Winner -Rows $rows -Metric latency
$bestTokensPerSecond = Select-Winner -Rows $rows -Metric tokens_per_second
$largestContext = Select-Winner -Rows $rows -Metric context
$failedRows = @($rows | Where-Object { $_.status -ne "ok" -or $_.validation_status -ne "passed" })
$fallbackRows = @($rows | Where-Object { $_.fallback_attempted })

$final = [ordered]@{
    metadata = [ordered]@{
        benchmark_run_id = $runId
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        server = "$ServerHost`:$ServerPort"
        model = $Model
        request_timeout_seconds = $RequestTimeoutSeconds
        runs_per_case = $RunsPerCase
        max_tokens = $MaxTokens
        temperature = $Temperature
        thinking_control = "prompt directives: /no_think and /think"
        memory_metric = "token/context only"
        ttft_definition = "client_roundtrip_ms unless execution.ttft_ms is provided"
    }
    preflight = [ordered]@{
        health_url = $healthUrl
        backends = $fleet.backends
    }
    summary = $summary
    winners = [ordered]@{
        fastest_latency = $fastest
        best_tokens_per_second = $bestTokensPerSecond
        largest_completed_context = $largestContext
    }
    failures = $failedRows
    fallbacks = $fallbackRows
    cases = $rows
}

Ensure-ParentDirectory -Path $OutputJsonPath
Ensure-ParentDirectory -Path $OutputCsvPath

$final | ConvertTo-Json -Depth 30 | Set-Content -Path $OutputJsonPath -Encoding UTF8
$rows | Export-Csv -Path $OutputCsvPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Full coverage benchmark summary"
($summary | Format-Table thinking_mode, execution_profile, selected_backends, total, succeeded, failed, fallback_count, latency_p50_ms, tokens_per_second_p50, max_total_tokens -AutoSize | Out-String).TrimEnd() | Write-Host
Write-Host ""
Write-Host "Winners"
if ($null -ne $fastest) {
    Write-Host ("Fastest latency: {0}/{1} on {2} at {3} ms" -f $fastest.thinking_mode, $fastest.execution_profile, $fastest.selected_backend, $fastest.latency_ms)
}
if ($null -ne $bestTokensPerSecond) {
    Write-Host ("Best tokens/sec: {0}/{1} on {2} at {3}" -f $bestTokensPerSecond.thinking_mode, $bestTokensPerSecond.execution_profile, $bestTokensPerSecond.selected_backend, $bestTokensPerSecond.tokens_per_second)
}
if ($null -ne $largestContext) {
    Write-Host ("Largest completed context: {0}/{1} on {2} with {3} total tokens" -f $largestContext.thinking_mode, $largestContext.execution_profile, $largestContext.selected_backend, $largestContext.total_tokens)
}
Write-Host ""
Write-Host "Wrote JSON: $OutputJsonPath"
Write-Host "Wrote CSV:  $OutputCsvPath"

if (@($failedRows).Count -gt 0) {
    $failureSummary = ($failedRows | Select-Object thinking_mode, execution_profile, selected_backend, status, validation_status | Format-Table -AutoSize | Out-String).TrimEnd()
    Write-Host ""
    Write-Host "Failures"
    Write-Host $failureSummary
    throw "Full coverage benchmark completed with $(@($failedRows).Count) failed or invalid row(s). Artifacts were written before failure."
}

$final | ConvertTo-Json -Depth 30 | Write-Output

