<#
.SYNOPSIS
Benchmark and latency profiling for multi-device harness routing.

.DESCRIPTION
Executes a fixed benchmark matrix across selected routes to gather routing and
latency distribution metrics. Produces request-level evidence plus aggregate
statistics (success/error rates, fallback counts, tokens/sec, and latency
percentiles).
#>
param(
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 8080,
    [int]$RequestTimeoutSeconds = 300,
    [int]$WarmupIterations = 2,
    [int]$Iterations = 10,
    [double]$Temperature = 0.2,
    [int]$MaxTokens = 256,
    [string]$Model = "",
    [string]$Prompt = "",
    [string]$ExecutionProfileOverride = "",
    [string]$OutputDir = "",
    [switch]$SkipHealthCheck,
    [switch]$WriteCsv,
    [switch]$FailOnBenchmarkAssertion,
    [switch]$RequireBackendEvidence
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

function Estimate-TokenCount {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return 0
    }
    return [int][math]::Ceiling($Text.Length / 4.0)
}

$testMatrix = @(
    @{
        id = "short_direct"
        route = "direct"
        label = "short_prompt"
        messages = @(@{role = "user"; content = "Summarize the benefits of staged multi-device model routing in one sentence."})
        context_tier = "low"
        expected_context_tokens = 32
        expected_max_output_tokens = 256
        note = "Short prompt, low context."
    },
    @{
        id = "tool_required_short"
        route = "tool_required"
        label = "tool_augmented"
        messages = @(@{role = "user"; content = "This request intentionally requires a calculator tool call. Use calculator with arguments exactly containing expression = '14 + 29', then explain one short inference tradeoff."})
        toolset = @("calculator")
        context_tier = "low"
        expected_context_tokens = 44
        expected_max_output_tokens = 256
        note = "Tool-augmented route, same max tokens."
    },
    @{
        id = "direct_long"
        route = "direct"
        label = "long_prompt"
        messages = @(@{role = "user"; content = ("The following prompt probes sustained reasoning over longer context. " * 180)})
        context_tier = "long_700+"
        expected_context_tokens = 700
        expected_max_output_tokens = 256
        note = "Long prompt baseline (700+ token target)."
    },
    @{
        id = "near_context_limit"
        route = "grounded_qa"
        label = "near_context_limit"
        messages = @(@{role = "user"; content = ("Context continuity and routing policy should preserve confidence while handling long retrieval windows. " * 420)})
        context_tier = "near_context_limit"
        expected_context_tokens = 3800
        expected_max_output_tokens = 256
        note = "Near-context pressure probe."
    },
    @{
        id = "structured_extraction"
        route = "structured_extraction"
        label = "structured"
        messages = @(@{role = "user"; content = "Extract two risks and two mitigations from: queue backpressure, routing confidence, fallback policy, and scheduling load balancing."})
        context_tier = "low"
        expected_context_tokens = 36
        expected_max_output_tokens = 256
        note = "Structured route baseline."
    },
    @{
        id = "agentic_parallel_complex"
        route = "direct"
        label = "agentic_parallel"
        messages = @(@{role = "user"; content = "Design a multi-step short story with worldbuilding, a character arc, a twist, and a revision checklist. Return the final story in a concise form."})
        execution_profile = "agentic_parallel"
        context_tier = "complex_multi_step"
        expected_context_tokens = 120
        expected_max_output_tokens = 512
        note = "Complex prompt should trigger CPU controller plus NPU/GPU workers."
    }
)

$customPromptEnabled = -not [string]::IsNullOrWhiteSpace($Prompt)
if ($customPromptEnabled) {
    $estimatedPromptTokens = [int][math]::Ceiling($Prompt.Length / 4.0)
    $testMatrix = @(
        @{
            id = "custom_prompt"
            route = "direct"
            label = "custom_prompt"
            messages = @(@{role = "user"; content = $Prompt})
            context_tier = "custom_prompt"
            expected_context_tokens = $estimatedPromptTokens
            expected_max_output_tokens = $MaxTokens
            note = "Custom benchmark prompt supplied via -Prompt; direct route used so the prompt is not transformed by route-specific tool/schema behavior."
        }
    )
}

if (-not [string]::IsNullOrWhiteSpace($ExecutionProfileOverride)) {
    foreach ($case in $testMatrix) {
        $case.execution_profile = $ExecutionProfileOverride
    }
}

function Get-ExecutionMetadata {
    param([object]$Response)

    if ($null -eq $Response) {
        return [ordered]@{
            backend_id = $null
            fallback_attempted = $false
            attempts = @()
            selected_via = $null
            profile_source = $null
            wait_ms = $null
            runtime_ms = $null
            tokens_per_second = $null
            backend_plan = @()
            fallback_reason = $null
            error_codes = @()
            profile = $null
            agentic_parallel = $null
            provider_configured_backend = $null
            provider_generation_backend = $null
            provider_fallback_active = $false
            provider_model = $null
            provider_warning = $null
        }
    }

    $execution = Get-OptionalProperty -InputObject $Response -Name "execution"
    $provider = Get-OptionalProperty -InputObject $Response -Name "provider"
    if (-not $execution -and $provider) {
        $execution = Get-OptionalProperty -InputObject $provider -Name "execution"
    }
    if (-not $execution) {
        return [ordered]@{
            backend_id = $null
            fallback_attempted = $false
            attempts = @()
            selected_via = $null
            profile_source = $null
            wait_ms = $null
            runtime_ms = $null
            tokens_per_second = $null
            backend_plan = @()
            fallback_reason = $null
            error_codes = @()
            profile = $null
            agentic_parallel = $null
            provider_configured_backend = $null
            provider_generation_backend = $null
            provider_fallback_active = $false
            provider_model = $null
            provider_warning = $null
        }
    }

    $errorCodes = @()
    $attempts = Get-OptionalProperty -InputObject $execution -Name "attempts"
    foreach ($attempt in @($attempts)) {
        $attemptErrorCode = Get-OptionalProperty -InputObject $attempt -Name "error_code"
        if ($attemptErrorCode) {
            $errorCodes += [string]$attemptErrorCode
        }
    }
    $tokensPerSecond = Get-OptionalProperty -InputObject $execution -Name "tokens_per_second"
    $backendPlan = Get-OptionalProperty -InputObject $execution -Name "backend_plan"
    $agenticParallel = Get-OptionalProperty -InputObject $execution -Name "agentic_parallel"

    return [ordered]@{
        backend_id = Get-OptionalProperty -InputObject $execution -Name "backend_id"
        profile = Get-OptionalProperty -InputObject $execution -Name "profile"
        profile_source = Get-OptionalProperty -InputObject $execution -Name "profile_source"
        fallback_attempted = [bool](Get-OptionalProperty -InputObject $execution -Name "fallback_attempted")
        attempts = if ($null -eq $attempts) { @() } else { @($attempts) }
        selected_via = Get-OptionalProperty -InputObject $execution -Name "selected_via"
        wait_ms = Get-OptionalProperty -InputObject $execution -Name "wait_ms"
        runtime_ms = Get-OptionalProperty -InputObject $execution -Name "runtime_ms"
        tokens_per_second = if ($tokensPerSecond) { [double]$tokensPerSecond } else { $null }
        backend_plan = if ($null -eq $backendPlan) { @() } else { @($backendPlan) }
        fallback_reason = Get-OptionalProperty -InputObject $execution -Name "fallback_reason"
        error_codes = @($errorCodes)
        agentic_parallel = $agenticParallel
        provider_configured_backend = Get-OptionalProperty -InputObject $provider -Name "configured_backend"
        provider_generation_backend = Get-OptionalProperty -InputObject $provider -Name "generation_backend"
        provider_fallback_active = [bool](Get-OptionalProperty -InputObject $provider -Name "fallback_active")
        provider_model = Get-OptionalProperty -InputObject $provider -Name "model"
        provider_warning = Get-OptionalProperty -InputObject $provider -Name "provider_warning"
    }
}

function Get-BenchmarkResponseText {
    param([object]$Response)

    if ($null -eq $Response) {
        return ""
    }

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

function Get-BenchmarkUsageMetrics {
    param(
        [object]$Response,
        [string]$PromptText,
        [string]$ResponseText
    )

    $usage = Get-OptionalProperty -InputObject $Response -Name "usage"
    $promptTokens = Get-OptionalProperty -InputObject $usage -Name "prompt_tokens"
    if ($null -eq $promptTokens) {
        $promptTokens = Get-OptionalProperty -InputObject $usage -Name "input_tokens"
    }
    $completionTokens = Get-OptionalProperty -InputObject $usage -Name "completion_tokens"
    if ($null -eq $completionTokens) {
        $completionTokens = Get-OptionalProperty -InputObject $usage -Name "output_tokens"
    }
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
    }
}

function New-OrderedAttempts {
    param([object[]]$Attempts)
    $ordered = @()
    foreach ($attempt in $Attempts) {
        if (-not $attempt) {
            continue
        }
        $backendId = Get-OptionalProperty -InputObject $attempt -Name "backend_id"
        if ($null -ne $backendId -and -not [string]::IsNullOrWhiteSpace([string]$backendId)) {
            $ordered += [string]$backendId
        }
    }
    return $ordered
}

function Invoke-BenchmarkRequest {
    param(
        [hashtable]$CaseDef,
        [string]$ModelOverride
    )

    $payload = [ordered]@{
        messages = $CaseDef.messages
        max_tokens = [int]$MaxTokens
        temperature = $Temperature
        route = $CaseDef.route
        request_id = "md-bench-" + [guid]::NewGuid().ToString("N")
    }
    if (-not [string]::IsNullOrWhiteSpace($ModelOverride)) {
        $payload.model = $ModelOverride
    }
    if ($CaseDef.ContainsKey("toolset") -and $CaseDef["toolset"]) {
        $payload.toolset = $CaseDef["toolset"]
    }
    if ($CaseDef.ContainsKey("execution_profile") -and $CaseDef["execution_profile"]) {
        $payload.execution_profile = $CaseDef["execution_profile"]
    }

    $start = Get-Date
    try {
        $raw = Invoke-WebRequest -UseBasicParsing -Uri $chatUrl -Method Post -TimeoutSec $RequestTimeoutSeconds -ContentType "application/json" -Body ($payload | ConvertTo-Json -Depth 30) -ErrorAction Stop
        $latencyMs = [math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        if ($raw.StatusCode -lt 200 -or $raw.StatusCode -ge 300) {
            return [ordered]@{
                ok = $false
                status = "http_error"
                status_code = $raw.StatusCode
                response = $null
                elapsed_ms = $latencyMs
                failure_stage = "http_status"
                error_code = "http_$($raw.StatusCode)"
                error_message = "HTTP $($raw.StatusCode)"
                meta = Get-ExecutionMetadata -Response $null
            }
        }
        $response = $raw.Content | ConvertFrom-HarnessJson -Depth 20
        $meta = Get-ExecutionMetadata -Response $response
        $errorCode = if ($meta.error_codes.Count -gt 0) { [string]$meta.error_codes[0] } else { "" }
        if ([string]::IsNullOrWhiteSpace($errorCode)) {
            $responseErrorCode = Get-OptionalProperty -InputObject $response -Name "error_code"
            if ($responseErrorCode) {
                $errorCode = [string]$responseErrorCode
            }
        }
        if ([string]::IsNullOrWhiteSpace($errorCode)) {
            $validation = Get-OptionalProperty -InputObject $response -Name "validation"
            $validationErrorCodes = Get-OptionalProperty -InputObject $validation -Name "error_codes"
            $validationErrorCodeRows = @($validationErrorCodes | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
            if ($validationErrorCodeRows.Count -gt 0) {
                $errorCode = [string]$validationErrorCodeRows[0]
            }
        }
        $errorMessage = ""
        if ($response.status -ne "ok") {
            $responseError = Get-OptionalProperty -InputObject $response -Name "error"
            if ($responseError) {
                $errorMessage = [string]$responseError
            }
        }
        return [ordered]@{
            ok = ($response.status -eq "ok")
            status = [string]$response.status
            status_code = $raw.StatusCode
            response = $response
            elapsed_ms = $latencyMs
            failure_stage = $null
            error_code = $errorCode
            error_message = $errorMessage
            meta = $meta
            ttft_ms = $latencyMs
        }
    } catch {
        $latencyMs = [math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        $msg = $_.Exception.Message
        return [ordered]@{
            ok = $false
            status = "exception"
            status_code = $null
            response = $null
            elapsed_ms = $latencyMs
            ttft_ms = $latencyMs
            failure_stage = "exception"
            error_code = "request_exception"
            error_message = $msg
            meta = Get-ExecutionMetadata -Response $null
        }
    }
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

function Write-CsvOutput {
    param(
        [string]$Path,
        [object[]]$Rows
    )

    if (-not $Rows -or $Rows.Count -eq 0) {
        return
    }

    $headers = $Rows[0].Keys
    $lines = @()
    $lines += ($headers -join ",")
    foreach ($row in $Rows) {
        $values = foreach ($header in $headers) {
            $value = $row[$header]
            if ($null -eq $value) {
                $value = ""
            }
            $escaped = [string]$value -replace '"', '""'
            if ($escaped -match '[,"\r\n]') {
                '"' + $escaped + '"'
            } else {
                $escaped
            }
        }
        $lines += ($values -join ",")
    }
    $lines | Set-Content -Path $Path -Encoding UTF8
}

function New-Accumulator {
    return [ordered]@{
        total = 0
        ok = 0
        error = 0
        fallback = 0
        latencies = @()
        ttft = @()
        tps = @()
        errors = @{}
        status_counts = @{}
        backend_plan_misses = 0
    }
}

function Record-Sample {
    param(
        [System.Collections.IDictionary]$Accumulator,
        [int]$LatencyMs,
        [int]$TtftMs,
        [double]$TokensPerSecond,
        [bool]$Success,
        [bool]$Fallback,
        [string]$ErrorCode,
        [string]$Status
    )
    $Accumulator.total++
    if ($Success) {
        $Accumulator.ok++
    } else {
        $Accumulator.error++
    }
    if ($Fallback) {
        $Accumulator.fallback++
    }

    if ($LatencyMs -gt 0) {
        $Accumulator.latencies += [double]$LatencyMs
    }
    if ($TtftMs -ge 0) {
        $Accumulator.ttft += [double]$TtftMs
    }
    if ($TokensPerSecond -ge 0) {
        $Accumulator.tps += [double]$TokensPerSecond
    }
    if ($Status) {
        if (-not $Accumulator.status_counts.ContainsKey($Status)) {
            $Accumulator.status_counts[$Status] = 0
        }
        $Accumulator.status_counts[$Status]++
    }
    if (-not [string]::IsNullOrWhiteSpace($ErrorCode)) {
        if (-not $Accumulator.errors.ContainsKey($ErrorCode)) {
            $Accumulator.errors[$ErrorCode] = 0
        }
        $Accumulator.errors[$ErrorCode]++
    }
}

function Build-PercentileReport {
    param([System.Collections.IDictionary]$Bucket)
    return [ordered]@{
        p50_ms = Get-Percentile -Samples $Bucket.latencies -Percentile 50
        p95_ms = Get-Percentile -Samples $Bucket.latencies -Percentile 95
        p99_ms = Get-Percentile -Samples $Bucket.latencies -Percentile 99
        ttft_p50_ms = Get-Percentile -Samples $Bucket.ttft -Percentile 50
        ttft_p95_ms = Get-Percentile -Samples $Bucket.ttft -Percentile 95
        ttft_p99_ms = Get-Percentile -Samples $Bucket.ttft -Percentile 99
        tokens_per_second_p50 = Get-Percentile -Samples $Bucket.tps -Percentile 50
        tokens_per_second_p95 = Get-Percentile -Samples $Bucket.tps -Percentile 95
        tokens_per_second_p99 = Get-Percentile -Samples $Bucket.tps -Percentile 99
    }
}

function New-BenchmarkPlanRecord {
    param(
        [string]$Scenario,
        [int]$Iteration,
        [hashtable]$CaseDef,
        [string]$Backend,
        [string]$Status,
        [string]$FailureStage,
        [bool]$Fallback,
        [int]$LatencyMs,
        [int]$TtftMs,
        [double]$TokensPerSecond,
        [string]$AttemptBackends,
        [string]$SelectedVia,
        [string]$ExecutionProfile,
        [int]$AgenticWorkers,
        [string]$AgenticControllerBackend,
        [string]$ErrorCode,
        [string]$ErrorMessage,
        [string]$ResponseId,
        [string]$ResponseText,
        [int]$PromptTokens,
        [int]$CompletionTokens,
        [int]$TotalTokens,
        [string]$ProviderConfiguredBackend,
        [string]$ProviderGenerationBackend,
        [bool]$ProviderFallbackActive,
        [string]$ProviderModel,
        [string]$ProviderWarning
    )

    return [ordered]@{
        scenario = $Scenario
        run_id = "md-bench-" + [guid]::NewGuid().ToString("N")
        iteration = $Iteration
        case_id = [string]$CaseDef.id
        route = [string]$CaseDef.route
        context_tier = [string]$CaseDef.context_tier
        label = [string]$CaseDef.label
        selected_backend = $Backend
        execution_profile = $ExecutionProfile
        fallback_attempted = [bool]$Fallback
        attempt_backends = $AttemptBackends
        agentic_workers = $AgenticWorkers
        agentic_controller_backend = $AgenticControllerBackend
        selected_via = $SelectedVia
        status = [string]$Status
        failure_stage = $FailureStage
        latency_ms = $LatencyMs
        ttft_ms = $TtftMs
        tokens_per_second = $TokensPerSecond
        error_code = $ErrorCode
        error_message = $ErrorMessage
        response_id = $ResponseId
        response_chars = if ($null -eq $ResponseText) { 0 } else { $ResponseText.Length }
        response_text = if ($null -eq $ResponseText) { "" } else { $ResponseText }
        prompt_tokens = $PromptTokens
        completion_tokens = $CompletionTokens
        total_tokens = $TotalTokens
        provider_configured_backend = $ProviderConfiguredBackend
        provider_generation_backend = $ProviderGenerationBackend
        provider_fallback_active = [bool]$ProviderFallbackActive
        provider_model = $ProviderModel
        provider_warning = $ProviderWarning
        expected_context_tokens = [int]$CaseDef.expected_context_tokens
        expected_max_output_tokens = [int]$CaseDef.expected_max_output_tokens
        temp = $Temperature
        max_tokens = $MaxTokens
        model = $Model
        custom_prompt = $customPromptEnabled
        note = [string]$CaseDef.note
    }
}

if (-not $SkipHealthCheck) {
    Write-Host "Checking harness readiness at $healthUrl"
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 12
        if (-not $health.ok) {
            throw "Health check returned ok=$($health.ok)"
        }
    } catch {
        throw "Health check failed: $($_.Exception.Message)"
    }
}

if (-not (Test-Path -LiteralPath $OutputDir -PathType Container) -and (-not [string]::IsNullOrWhiteSpace($OutputDir))) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss")
$runId = "md-bench-run-$stamp"
$rows = @()

$backendBuckets = @{}
$routeBuckets = @{}
$overall = New-Accumulator

foreach ($case in $testMatrix) {
    Write-Host ("Running case: " + $case.id + " (" + $case.route + ")")

    for ($i = 1; $i -le $WarmupIterations; $i++) {
        $null = Invoke-BenchmarkRequest -CaseDef $case -ModelOverride $Model
    }

    for ($i = 1; $i -le $Iterations; $i++) {
        $result = Invoke-BenchmarkRequest -CaseDef $case -ModelOverride $Model
        $meta = $result.meta
        $attemptBackends = New-OrderedAttempts -Attempts $meta.attempts
        $selected = if ($meta.backend_id) { [string]$meta.backend_id } else { "unknown" }
        $selectedVia = [string]$meta.selected_via
        if ([string]::IsNullOrWhiteSpace($selectedVia)) {
            $selectedVia = "unknown"
        }
        $executionProfile = if ($meta.profile) { [string]$meta.profile } else { "unknown" }
        $agenticWorkers = 0
        $agenticControllerBackend = ""
        if ($meta.agentic_parallel) {
            $agenticWorkerRows = Get-OptionalProperty -InputObject $meta.agentic_parallel -Name "workers"
            $agenticWorkers = if ($agenticWorkerRows) { @($agenticWorkerRows).Count } else { 0 }
            $agenticController = Get-OptionalProperty -InputObject $meta.agentic_parallel -Name "controller_backend"
            if ($agenticController) {
                $agenticControllerBackend = [string]$agenticController
            }
        }
        $latency = [int]$result.elapsed_ms
        $ttft = [int]$result.ttft_ms

        $responseId = if ($result.response -and $result.response.id) { [string]$result.response.id } else { "" }
        $responseText = Get-BenchmarkResponseText -Response $result.response
        $effectivePrompt = (($case.messages | ForEach-Object { [string](Get-OptionalProperty -InputObject $_ -Name "content") }) -join "`n")
        $usage = Get-BenchmarkUsageMetrics -Response $result.response -PromptText $effectivePrompt -ResponseText $responseText
        $completionTokens = [int]$usage.completion_tokens

        $tokensPerSecond = if ($meta.tokens_per_second) { [double]$meta.tokens_per_second } else {
            if ($ttft -gt 0 -and $completionTokens -gt 0) {
                [math]::Round(($completionTokens / ([math]::Max(0.001, $ttft / 1000))), 3)
            } else {
                0
            }
        }

        $status = if ([string]::IsNullOrWhiteSpace($result.status)) { "unknown" } else { [string]$result.status }
        $failStage = if ([string]::IsNullOrWhiteSpace($result.failure_stage)) { "" } else { [string]$result.failure_stage }
        $errorCode = [string]$result.error_code
        $errorMessage = [string]$result.error_message
        $providerConfiguredBackend = [string]$meta.provider_configured_backend
        $providerGenerationBackend = [string]$meta.provider_generation_backend
        $providerFallbackActive = [bool]$meta.provider_fallback_active
        $providerModel = [string]$meta.provider_model
        $providerWarning = [string]$meta.provider_warning
        $fallback = ([bool]$meta.fallback_attempted) -or $providerFallbackActive

        if (-not $backendBuckets.ContainsKey($selected)) {
            $backendBuckets[$selected] = New-Accumulator
        }
        if (-not $routeBuckets.ContainsKey([string]$case.route)) {
            $routeBuckets[[string]$case.route] = New-Accumulator
        }

        Record-Sample -Accumulator $backendBuckets[$selected] -LatencyMs $latency -TtftMs $ttft -TokensPerSecond $tokensPerSecond -Success $result.ok -Fallback $fallback -ErrorCode $errorCode -Status $status
        Record-Sample -Accumulator $routeBuckets[$case.route] -LatencyMs $latency -TtftMs $ttft -TokensPerSecond $tokensPerSecond -Success $result.ok -Fallback $fallback -ErrorCode $errorCode -Status $status
        Record-Sample -Accumulator $overall -LatencyMs $latency -TtftMs $ttft -TokensPerSecond $tokensPerSecond -Success $result.ok -Fallback $fallback -ErrorCode $errorCode -Status $status

        $rows += New-BenchmarkPlanRecord -Scenario $runId -Iteration $i -CaseDef $case -Backend $selected -Status $status -FailureStage $failStage `
            -Fallback $fallback -LatencyMs $latency -TtftMs $ttft -TokensPerSecond $tokensPerSecond -AttemptBackends ($attemptBackends -join "|") -SelectedVia $selectedVia `
            -ExecutionProfile $executionProfile -AgenticWorkers $agenticWorkers -AgenticControllerBackend $agenticControllerBackend `
            -ErrorCode $errorCode -ErrorMessage $errorMessage -ResponseId $responseId -ResponseText $responseText `
            -PromptTokens ([int]$usage.prompt_tokens) -CompletionTokens ([int]$usage.completion_tokens) -TotalTokens ([int]$usage.total_tokens) `
            -ProviderConfiguredBackend $providerConfiguredBackend -ProviderGenerationBackend $providerGenerationBackend `
            -ProviderFallbackActive $providerFallbackActive -ProviderModel $providerModel -ProviderWarning $providerWarning

        if ($RequireBackendEvidence.IsPresent -and $selected -eq "unknown" -and $result.ok) {
            throw "Could not determine backend selection for benchmark sample."
        }
        if ($RequireBackendEvidence.IsPresent -and $providerFallbackActive) {
            throw "Provider fallback was active for benchmark sample case=$($case.id), backend=$selected."
        }
    }
}

$backendSummary = @{}
foreach ($backend in $backendBuckets.Keys) {
    $bucket = $backendBuckets[$backend]
    $backendSummary[$backend] = [ordered]@{
        total = $bucket.total
        succeeded = $bucket.ok
        failed = $bucket.error
        success_rate = if ($bucket.total -gt 0) { [math]::Round(($bucket.ok / [double]$bucket.total) * 100, 2) } else { 0.0 }
        fallback_count = $bucket.fallback
        fallback_rate = if ($bucket.total -gt 0) { [math]::Round(($bucket.fallback / [double]$bucket.total) * 100, 2) } else { 0.0 }
        status_counts = $bucket.status_counts
        errors = $bucket.errors
        percentile_ms = Build-PercentileReport -Bucket $bucket
        tokens_per_second_avg = if ($bucket.tps.Count -gt 0) { [math]::Round(($bucket.tps | Measure-Object -Average).Average, 3) } else { 0 }
    }
}

$routeSummary = @{}
foreach ($route in $routeBuckets.Keys) {
    $bucket = $routeBuckets[$route]
    $routeSummary[$route] = [ordered]@{
        total = $bucket.total
        succeeded = $bucket.ok
        failed = $bucket.error
        success_rate = if ($bucket.total -gt 0) { [math]::Round(($bucket.ok / [double]$bucket.total) * 100, 2) } else { 0.0 }
        fallback_count = $bucket.fallback
        fallback_rate = if ($bucket.total -gt 0) { [math]::Round(($bucket.fallback / [double]$bucket.total) * 100, 2) } else { 0.0 }
        status_counts = $bucket.status_counts
        errors = $bucket.errors
        percentile_ms = Build-PercentileReport -Bucket $bucket
        tokens_per_second_avg = if ($bucket.tps.Count -gt 0) { [math]::Round(($bucket.tps | Measure-Object -Average).Average, 3) } else { 0 }
    }
}

$overallSummary = [ordered]@{
    total = $overall.total
    succeeded = $overall.ok
    failed = $overall.error
    success_rate = if ($overall.total -gt 0) { [math]::Round(($overall.ok / [double]$overall.total) * 100, 2) } else { 0.0 }
    fallback_count = $overall.fallback
    fallback_rate = if ($overall.total -gt 0) { [math]::Round(($overall.fallback / [double]$overall.total) * 100, 2) } else { 0.0 }
    status_counts = $overall.status_counts
    errors = $overall.errors
    percentiles_ms = Build-PercentileReport -Bucket $overall
    tokens_per_second_avg = if ($overall.tps.Count -gt 0) { [math]::Round(($overall.tps | Measure-Object -Average).Average, 3) } else { 0 }
}

$final = [ordered]@{
    metadata = [ordered]@{
        benchmark_run_id = $runId
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        server = "$ServerHost`:$ServerPort"
        request_timeout_seconds = $RequestTimeoutSeconds
        iterations = $Iterations
        warmup_iterations = $WarmupIterations
        temperature = $Temperature
        max_tokens = $MaxTokens
        model = $Model
        custom_prompt = $customPromptEnabled
        prompt_chars = if ($customPromptEnabled) { $Prompt.Length } else { 0 }
        prompt_estimated_tokens = if ($customPromptEnabled) { [int][math]::Ceiling($Prompt.Length / 4.0) } else { 0 }
        execution_profile_override = $ExecutionProfileOverride
        require_backend_evidence = $RequireBackendEvidence.IsPresent
        ttft_definition = "client_roundtrip_ms (non-stream)"
    }
    overall = $overallSummary
    backend_summary = $backendSummary
    route_summary = $routeSummary
    cases = $rows
}

$assertionsFailed = $false
if ($overall.error -gt 0 -and ($overall.total -gt 0)) {
    $errorRate = ($overall.error / [double]$overall.total) * 100
    $assertionsFailed = $errorRate -gt 5
}

if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $jsonPath = Join-Path $OutputDir "multidevice-benchmark-$stamp.json"
    $final | ConvertTo-Json -Depth 20 | Set-Content -Path $jsonPath -Encoding UTF8
    Write-Host "Wrote benchmark JSON: $jsonPath"
    if ($WriteCsv.IsPresent) {
        $csvPath = Join-Path $OutputDir "multidevice-benchmark-$stamp.csv"
        Write-CsvOutput -Path $csvPath -Rows $rows
        Write-Host "Wrote benchmark CSV: $csvPath"
    }
}

$final | ConvertTo-Json -Depth 20 | Write-Output

if ($FailOnBenchmarkAssertion.IsPresent -and $assertionsFailed) {
    throw "Benchmark assertion failed: error rate > 5%."
}

