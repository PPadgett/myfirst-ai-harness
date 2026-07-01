<#
.SYNOPSIS
Smoke test the multi-device harness dispatcher (NPU/GPU/CPU).

.DESCRIPTION
Runs a small fixed request matrix against a running harness service and prints
per-request backend selection plus an aggregate summary.
#>
param(
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 8080,
    [int]$RequestTimeoutSeconds = 180,
    [string]$Model = "",
    [string]$OutputPath = ""
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

$cases = @(
    @{
        id = "direct_short"
        name = "short direct route"
        route = "direct"
        messages = @(@{role = "user"; content = "Hello—give one sentence status on Ryzen AI today."})
        max_tokens = 64
        expected_device = "npu"
    },
    @{
        id = "grounded_qa_medium"
        name = "grounded qa route"
        route = "grounded_qa"
        messages = @(@{role = "user"; content = "Compare two practical ways to choose model size for local deployment."})
        max_tokens = 128
        expected_device = "gpu"
    },
    @{
        id = "structured_extraction"
        name = "structured extraction route"
        route = "structured_extraction"
        messages = @(@{role = "user"; content = "Extract two bullet themes from this: AI offload planning, routing strategy, queue pressure tuning, and fallback policy."})
        max_tokens = 160
        expected_device = "gpu"
    },
    @{
        id = "tool_required"
        name = "tool required route"
        route = "tool_required"
        messages = @(@{role = "user"; content = "This request intentionally requires a calculator tool call. Use calculator with arguments exactly containing expression = '18 + 22', then describe the result in one sentence."})
        max_tokens = 128
        expected_device = "cpu"
        toolset = @("calculator")
    },
    @{
        id = "direct_long_context"
        name = "direct long context"
        route = "direct"
        messages = @(@{role = "user"; content = ("This is a context load probe. " * 500)})
        max_tokens = 128
        expected_device = "gpu"
    },
    @{
        id = "agentic_parallel_complex"
        name = "agentic parallel complex route"
        route = "direct"
        messages = @(@{role = "user"; content = "Design a multi-step short story with worldbuilding, a character arc, a twist, and a revision checklist. Return the final story in a concise form."})
        max_tokens = 512
        expected_device = "cpu"
        execution_profile = "agentic_parallel"
    }
)

function Get-ExecutionMetadata {
    param([object]$Response)

    if ($null -eq $Response) {
        return [ordered]@{
            backend_id = $null
            fallback_attempted = $false
            selected_via = $null
            wait_ms = $null
            runtime_ms = $null
            concurrency_bucket = $null
            backend_plan = @()
            attempts = @()
            profile = $null
            agentic_parallel = $null
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
            selected_via = $null
            wait_ms = $null
            runtime_ms = $null
            concurrency_bucket = $null
            backend_plan = @()
            attempts = @()
            profile = $null
            agentic_parallel = $null
        }
    }

    $attempts = Get-OptionalProperty -InputObject $execution -Name "attempts"
    $backendPlan = Get-OptionalProperty -InputObject $execution -Name "backend_plan"
    $agenticParallel = Get-OptionalProperty -InputObject $execution -Name "agentic_parallel"

    return [ordered]@{
        backend_id = Get-OptionalProperty -InputObject $execution -Name "backend_id"
        profile = Get-OptionalProperty -InputObject $execution -Name "profile"
        fallback_attempted = [bool](Get-OptionalProperty -InputObject $execution -Name "fallback_attempted")
        selected_via = Get-OptionalProperty -InputObject $execution -Name "selected_via"
        wait_ms = Get-OptionalProperty -InputObject $execution -Name "wait_ms"
        runtime_ms = Get-OptionalProperty -InputObject $execution -Name "runtime_ms"
        concurrency_bucket = Get-OptionalProperty -InputObject $execution -Name "concurrency_bucket"
        backend_plan = if ($null -eq $backendPlan) { @() } else { @($backendPlan) }
        attempts = if ($null -eq $attempts) { @() } else { @($attempts) }
        agentic_parallel = $agenticParallel
    }
}

function New-ErrorRecord {
    param(
        [string]$CaseId,
        [string]$ErrorCode,
        [string]$ErrorText
    )

    return [ordered]@{
        case_id = $CaseId
        error = $ErrorText
        error_code = if ([string]::IsNullOrWhiteSpace($ErrorCode)) { "request_error" } else { $ErrorCode }
    }
}

try {
    Write-Host "Checking runtime health at $healthUrl"
    $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 10
    if (-not $health.ok) {
        throw "Runtime health reports ok=$($health.ok)."
    }
    if ($null -eq $health.backends -or @($health.backends).Count -eq 0) {
        throw "Runtime health returned no backend metadata."
    }
} catch {
    throw "Health check failed. Start multi-device stack first: $_"
}

$results = @()
$start = Get-Date

foreach ($case in $cases) {
    $payload = [ordered]@{
        messages = $case.messages
        max_tokens = $case.max_tokens
        route = $case.route
        request_id = ("smoke-" + [guid]::NewGuid().ToString("N"))
    }
    if (-not [string]::IsNullOrWhiteSpace($Model)) {
        $payload.model = $Model
    }

    if ($case.ContainsKey("toolset") -and $case["toolset"]) {
        $payload.toolset = $case["toolset"]
    }
    if ($case.ContainsKey("execution_profile") -and $case["execution_profile"]) {
        $payload.execution_profile = $case["execution_profile"]
    }

    try {
        $raw = Invoke-WebRequest -Uri $chatUrl -Method Post -TimeoutSec $RequestTimeoutSeconds -ContentType "application/json" -Body ($payload | ConvertTo-Json -Depth 30)
        if ($raw.StatusCode -lt 200 -or $raw.StatusCode -ge 300) {
            $results += New-ErrorRecord -CaseId $case.id -ErrorCode "http_$($raw.StatusCode)" -ErrorText "HTTP $($raw.StatusCode)"
            continue
        }
        $response = $raw.Content | ConvertFrom-HarnessJson -Depth 20
        $meta = Get-ExecutionMetadata -Response $response
        $responseStatus = Get-OptionalProperty -InputObject $response -Name "status"
        $responseRoute = Get-OptionalProperty -InputObject $response -Name "route"
        $responseMs = Get-OptionalProperty -InputObject $response -Name "response_ms"
        $responseEvidence = Get-OptionalProperty -InputObject $response -Name "evidence"
        $provider = Get-OptionalProperty -InputObject $response -Name "provider"
        $providerErrorCodes = Get-OptionalProperty -InputObject $provider -Name "error_codes"
        $guard = Get-OptionalProperty -InputObject $response -Name "guard"
        $guardOutput = Get-OptionalProperty -InputObject $guard -Name "output"
        $guardAllow = Get-OptionalProperty -InputObject $guardOutput -Name "allow"
        $agenticWorkers = if ($meta.agentic_parallel) { Get-OptionalProperty -InputObject $meta.agentic_parallel -Name "workers" } else { $null }

        $result = [ordered]@{
            case_id = $case.id
            case_name = $case.name
            configured_route = $case.route
            expected_device = $case.expected_device
            status = $responseStatus
            route = $responseRoute
            selected_backend = $meta.backend_id
            execution_profile = $meta.profile
            execution_selected_via = $meta.selected_via
            fallback_attempted = $meta.fallback_attempted
            wait_ms = $meta.wait_ms
            runtime_ms = $meta.runtime_ms
            concurrency_bucket = $meta.concurrency_bucket
            attempts = $meta.attempts.Count
            agentic_workers = if ($agenticWorkers) { @($agenticWorkers).Count } else { 0 }
            agentic_controller_backend = if ($meta.agentic_parallel) { Get-OptionalProperty -InputObject $meta.agentic_parallel -Name "controller_backend" } else { $null }
            response_ms = $responseMs
            evidence = if ($null -eq $responseEvidence) { @() } else { @($responseEvidence) }
            error_codes = if ($null -eq $providerErrorCodes) { @() } else { @($providerErrorCodes) }
            guard_allow = if ($null -ne $guardAllow) { [bool]$guardAllow } else { $true }
        }

        if ([string]$responseStatus -ne "ok") {
            $result.validation_failed = $true
        } else {
            $result.validation_failed = $false
        }
        $results += $result
    } catch {
        $results += New-ErrorRecord -CaseId $case.id -ErrorCode "" -ErrorText $_.Exception.Message
    }
}

$elapsed = [math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
$backendsByCase = @{}
$fallbackCount = 0
$failCount = 0
$statusCounts = @{}
foreach ($result in $results) {
    if ($result.Contains("selected_backend")) {
        $backendId = if ([string]::IsNullOrWhiteSpace([string]$result["selected_backend"])) { "unknown" } else { [string]$result["selected_backend"] }
        if (-not $backendsByCase.ContainsKey($backendId)) {
            $backendsByCase[$backendId] = 0
        }
        $backendsByCase[$backendId]++
    }
    $hasStatus = $result.Contains("status")
    $status = if ($hasStatus) { [string]$result["status"] } else { "" }
    if ($hasStatus -and $status -ne "ok" -and -not $result.Contains("error")) {
        $result.validation_failed = $true
    }
    if ($result.Contains("error")) {
        $failCount++
    } elseif ($hasStatus -and $status -ne "ok") {
        $failCount++
    } elseif ($result.Contains("fallback_attempted") -and [bool]$result["fallback_attempted"]) {
        $fallbackCount++
    }
    if (-not [string]::IsNullOrWhiteSpace($status)) {
        if (-not $statusCounts.ContainsKey($status)) {
            $statusCounts[$status] = 0
        }
        $statusCounts[$status]++
    }
}

$summary = [ordered]@{
    total = $results.Count
    succeeded = ($results.Count - $failCount)
    failed = $failCount
    fallback_count = $fallbackCount
    backend_selection_counts = $backendsByCase
    status_counts = $statusCounts
    elapsed_ms = $elapsed
}

$output = [ordered]@{
    summary = $summary
    cases = $results
}

if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
    $output | ConvertTo-Json -Depth 20 | Set-Content -Path $OutputPath -Encoding UTF8
    Write-Host "Wrote output: $OutputPath"
}

$output | ConvertTo-Json -Depth 20 | Write-Output

