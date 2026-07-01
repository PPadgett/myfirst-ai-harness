<#
.SYNOPSIS
Phase 2 validation for multi-device harness routing with backend failover.

.DESCRIPTION
Runs repeated request rounds across a fixed case matrix while optionally stopping
one or more backend services to force fallback behavior. Asserts that route selection
stays within expected fallback candidates and exports JSON + CSV diagnostics.
#>
param(
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 8080,
    [int]$RequestTimeoutSeconds = 180,
    [int]$Iterations = 8,
    [string]$Model = "",
    [string]$ComposeFile = "docker-compose.multi-device.yaml",
    [string]$ComposeProfile = "multi-device",
    [string]$EnvFile = "",
    [string]$OutputDir = "",
    [switch]$DryRun,
    [switch]$CheckServiceHealth,
    [switch]$SkipServiceControl,
    [switch]$FailOnAssertion,
    [switch]$WriteCsv
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

$backendServiceMap = @{
    npu = "asus-npu-backend"
    gpu = "asus-gpu-backend"
    cpu = "asus-cpu-backend"
}
$routeExpectations = @{
    direct = @("npu", "gpu", "cpu")
    grounded_qa = @("gpu", "npu", "cpu")
    structured_extraction = @("gpu", "cpu")
    tool_required = @("cpu", "gpu", "npu")
    code_or_data = @("gpu", "cpu", "npu")
}

$testCases = @(
    @{
        id = "direct_short"
        route = "direct"
        messages = @(@{role = "user"; content = "Hello—one concise status update with the next step only."})
        max_tokens = 64
        expected_backends = $routeExpectations.direct
        note = "NPU-first interactive path"
    },
    @{
        id = "grounded_qa_medium"
        route = "grounded_qa"
        messages = @(@{role = "user"; content = "Compare two practical approaches for local model deployment on hybrid hardware."})
        max_tokens = 128
        expected_backends = $routeExpectations.grounded_qa
        note = "GPU-first analysis path"
    },
    @{
        id = "structured_extraction"
        route = "structured_extraction"
        messages = @(@{role = "user"; content = "Extract two key themes from this: queue pressure, latency targets, fallback policy, and routing tiers."})
        max_tokens = 180
        expected_backends = $routeExpectations.structured_extraction
        note = "Batch-oriented path"
    },
    @{
        id = "tool_required"
        route = "tool_required"
        messages = @(@{role = "user"; content = "Use calculator: sum 19 + 23, then provide a one-sentence summary."})
        max_tokens = 160
        expected_backends = $routeExpectations.tool_required
        toolset = @("calculator")
        note = "Tool-heavy path prefers CPU"
    },
    @{
        id = "code_or_data"
        route = "code_or_data"
        messages = @(@{role = "user"; content = "Summarize this snippet with one concise action item and one risk: \"Edge cache misses cause latency spikes in GPU queue, while CPU remains stable under intermittent load.\""})
        max_tokens = 160
        expected_backends = $routeExpectations.code_or_data
        note = "Code/data route with structured response behavior"
    }
)

function Invoke-ComposeCommand {
    param(
        [string]$ComposeAction,
        [string[]]$Services
    )

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker is required for service-control scenarios. Install Docker Desktop or use -SkipServiceControl."
    }

    $args = @("-f", $ComposeFile, "--profile", $ComposeProfile)
    if ($EnvFile) {
        $args += @("--env-file", $EnvFile)
    }
    $args += @($ComposeAction)
    if ($Services -and $Services.Count -gt 0) {
        $args += $Services
    }

    $proc = Start-Process -FilePath "docker" -ArgumentList @("compose") + $args -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        throw "docker compose $ComposeAction $($Services -join ',') failed with exit code $($proc.ExitCode)."
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
            wait_ms = $null
            runtime_ms = $null
            tokens_per_second = $null
            concurrency_bucket = $null
            backend_plan = @()
            fallback_reason = $null
            backend_error_codes = @()
        }
    }

    $execution = $Response.execution
    if (-not $execution -and $Response.provider) {
        $execution = $Response.provider.execution
    }
    if (-not $execution) {
        return [ordered]@{
            backend_id = $null
            fallback_attempted = $false
            attempts = @()
            selected_via = $null
            wait_ms = $null
            runtime_ms = $null
            tokens_per_second = $null
            concurrency_bucket = $null
            backend_plan = @()
            fallback_reason = $null
            backend_error_codes = @()
        }
    }

    $errorCodes = @()
    foreach ($attempt in @($execution.attempts)) {
        if ($attempt.error_code) {
            $errorCodes += [string]$attempt.error_code
        }
    }
    return [ordered]@{
        backend_id = $execution.backend_id
        fallback_attempted = [bool]$execution.fallback_attempted
        attempts = @($execution.attempts)
        selected_via = $execution.selected_via
        wait_ms = $execution.wait_ms
        runtime_ms = $execution.runtime_ms
        tokens_per_second = if ($execution.tokens_per_second) { [double]$execution.tokens_per_second } else { $null }
        concurrency_bucket = $execution.concurrency_bucket
        backend_plan = @($execution.backend_plan)
        fallback_reason = $execution.fallback_reason
        backend_error_codes = @($errorCodes)
    }
}

function Invoke-HarnessRequest {
    param(
        [hashtable]$CaseDef,
        [string]$ModelOverride = ""
    )

    $payload = [ordered]@{
        messages = $CaseDef.messages
        max_tokens = [int]$CaseDef.max_tokens
        route = $CaseDef.route
        request_id = "md-phase2-" + [guid]::NewGuid().ToString("N")
    }
    if (-not [string]::IsNullOrWhiteSpace($ModelOverride)) {
        $payload.model = $ModelOverride
    }
    if ($CaseDef.toolset) {
        $payload.toolset = $CaseDef.toolset
    }

    $start = Get-Date
    try {
        $resp = Invoke-WebRequest -Uri $chatUrl -Method Post -TimeoutSec $RequestTimeoutSeconds -ContentType "application/json" -Body ($payload | ConvertTo-Json -Depth 30) -ErrorAction Stop
        $elapsedMs = [math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        if ($resp.StatusCode -lt 200 -or $resp.StatusCode -ge 300) {
            return [ordered]@{
                ok = $false
                failure_stage = "http_status"
                error_code = "http_$($resp.StatusCode)"
                status = "http_error"
                status_code = $resp.StatusCode
                elapsed_ms = $elapsedMs
                raw = $null
            }
        }
        $response = $resp.Content | ConvertFrom-HarnessJson -Depth 20
        $meta = Get-ExecutionMetadata -Response $response
        return [ordered]@{
            ok = ($response.status -eq "ok")
            failure_stage = $null
            error_code = $null
            status = [string]$response.status
            response = $response
            meta = $meta
            elapsed_ms = $elapsedMs
        }
    } catch {
        $elapsedMs = [math]::Round((New-TimeSpan -Start $start -End (Get-Date)).TotalMilliseconds)
        return [ordered]@{
            ok = $false
            failure_stage = "exception"
            error_code = "request_exception"
            status = "exception"
            status_code = $null
            elapsed_ms = $elapsedMs
            raw_error = $_.Exception.Message
            response = $null
            meta = Get-ExecutionMetadata -Response $null
        }
    }
}

function Get-ExpectedBackendsForCase {
    param([object[]]$Expected, [string[]]$Disabled)

    $disabledSet = @{}
    foreach ($item in $Disabled) {
        if ([string]::IsNullOrWhiteSpace($item)) {
            continue
        }
        $disabledSet[$item] = $true
    }
    $allowed = @()
    foreach ($candidate in $Expected) {
        if (-not $disabledSet.ContainsKey([string]$candidate)) {
            $allowed += [string]$candidate
        }
    }
    if ($allowed.Count -eq 0) {
        return @()
    }
    return $allowed
}

function Get-OrderedAttemptBackends {
    param([object[]]$Attempts)

    $ordered = @()
    foreach ($attempt in $Attempts) {
        if (-not $attempt) {
            continue
        }
        $backend = $attempt.backend_id
        if ($null -eq $backend -or [string]::IsNullOrWhiteSpace([string]$backend)) {
            continue
        }
        $ordered += [string]$backend
    }
    return $ordered
}

function Run-Scenario {
    param(
        [string]$ScenarioName,
        [string[]]$DisableBackends
    )

    $servicesToStop = @()
    foreach ($backend in $DisableBackends) {
        if ($backendServiceMap.ContainsKey($backend)) {
            $servicesToStop += $backendServiceMap[$backend]
        }
    }

    if ($DryRun.IsPresent) {
        Write-Host "Scenario '$ScenarioName': dry-run mode enabled; skipping service control and request execution."
        $rows = @()
        $caseIndex = 0
        $assertionFailures = 0
        $scenarioSucceeded = 0
        $scenarioFailed = 0
        $fallbackCount = 0
        $backendCounts = @{}

        foreach ($case in $testCases) {
            for ($i = 1; $i -le $Iterations; $i++) {
                $caseIndex++
                $expected = Get-ExpectedBackendsForCase -Expected $case.expected_backends -Disabled $DisableBackends
                $selected = $null
                $runResult = [ordered]@{
                    ok = $false
                    status = "dry_run_skipped"
                    meta = [ordered]@{
                        fallback_attempted = $false
                        attempts = @()
                        selected_via = "dry_run"
                        wait_ms = 0
                        runtime_ms = 0
                        tokens_per_second = 0
                        concurrency_bucket = "dry_run"
                        backend_plan = $case.expected_backends
                        fallback_reason = "dry_run"
                        backend_error_codes = @()
                    }
                }
                $attemptBackends = @()
                $assertionOk = $true
                $failureReason = ""

                if ($expected.Count -gt 0 -and $expected.Count -gt 0) {
                    $expectedText = $expected -join ", "
                    $failureReason = "dry_run_plan_only (expected_backends=$expectedText)"
                    $assertionOk = $true
                }

                if ($assertionOk) {
                    $scenarioSucceeded++
                } else {
                    $scenarioFailed++
                    $assertionFailures++
                }

                $rows += [ordered]@{
                    scenario = $ScenarioName
                    backend_disabled = ($DisableBackends -join "|")
                    case_id = [string]$case.id
                    route = [string]$case.route
                    expected_backends = $expected -join "|"
                    iteration = $i
                    status = $runResult.status
                    selected_backend = "dry-run"
                    attempt_backends = ""
                    fallback_attempted = [bool]$runResult.meta.fallback_attempted
                    fallback_reason = $runResult.meta.fallback_reason
                    attempts = $attemptBackends.Count
                    wait_ms = $runResult.meta.wait_ms
                    runtime_ms = $runResult.meta.runtime_ms
                    latency_ms = 0
                    guard_allow = $true
                    tokens_per_second = 0
                    concurrency_bucket = $runResult.meta.concurrency_bucket
                    selected_via = $runResult.meta.selected_via
                    assertion_ok = $assertionOk
                    assertion_failure = $failureReason
                    error_code = "dry_run"
                    response_id = "dry-run"
                    failure_stage = "dry_run"
                    request_error = ""
                    backend_plan = ($case.expected_backends -join "|")
                }
            }
        }

        $summary = [ordered]@{
            scenario = $ScenarioName
            disabled_backends = ($DisableBackends -join "|")
            total_requests = $rows.Count
            succeeded = $scenarioSucceeded
            failed = $scenarioFailed
            fallback_count = $fallbackCount
            assertion_failures = $assertionFailures
            backend_counts = @{ "dry-run" = $rows.Count }
        }
        return [ordered]@{
            scenario = $ScenarioName
            summary = $summary
            rows = $rows
            assertion_failed = $false
        }
    }

    if (-not $SkipServiceControl -and $servicesToStop.Count -gt 0) {
        Write-Host "Pausing scenario '$ScenarioName': disabling $($servicesToStop -join ', ')"
        Invoke-ComposeCommand -ComposeAction "stop" -Services $servicesToStop | Out-Null
    } elseif ($SkipServiceControl -and $servicesToStop.Count -gt 0) {
        Write-Host "Scenario '$ScenarioName': service control skipped by -SkipServiceControl (expected backend IDs: $($DisableBackends -join ', '))."
    } elseif ($servicesToStop.Count -eq 0) {
        Write-Host "Scenario '$ScenarioName': no backend disabled."
    }

    try {
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 8
            if (-not $health.ok) {
                throw "Health endpoint returned ok=$($health.ok)"
            }
            if ($CheckServiceHealth.IsPresent -and $health.backends) {
                foreach ($item in $health.backends) {
                        if ($item.enabled -eq $false) {
                            throw "Backend '$($item.id)' reported enabled=$($item.enabled)."
                        }
                    }
                }
            } catch {
                throw "Runtime not ready before scenario '$ScenarioName'. $($_.Exception.Message)"
            }

        $rows = @()
        $caseIndex = 0
        $assertionFailures = 0
        $scenarioSucceeded = 0
        $scenarioFailed = 0
        $fallbackCount = 0
        $backendCounts = @{}

        foreach ($case in $testCases) {
            for ($i = 1; $i -le $Iterations; $i++) {
                $caseIndex++
                $runResult = Invoke-HarnessRequest -CaseDef $case -ModelOverride $Model
                $selected = $runResult.meta.backend_id
                $expected = Get-ExpectedBackendsForCase -Expected $case.expected_backends -Disabled $DisableBackends
                $assertionOk = $true
                $failureReason = ""

                if ($null -ne $selected -and $selected -isnot [string]) {
                    $selected = [string]$selected
                }

                $status = [string]$runResult.status
                $attempts = if ($runResult.meta.attempts) { @($runResult.meta.attempts) } else { @() }
                $attemptBackends = Get-OrderedAttemptBackends -Attempts $attempts

                if ($runResult.ok -and $status -eq "ok") {
                    $scenarioSucceeded++
                } else {
                    $scenarioFailed++
                }

                if ($expected.Count -gt 0 -and [string]::IsNullOrWhiteSpace($selected) -and $attempts.Count -gt 0) {
                    $assertionOk = $false
                    $failureReason = "selected_backend is null"
                } elseif ($expected.Count -gt 0 -and [string]::IsNullOrWhiteSpace($selected) -and $attempts.Count -eq 0) {
                    $assertionOk = $false
                    $failureReason = "selected_backend is null and no execution attempts were returned"
                } elseif ($expected.Count -gt 0 -and ($expected -notcontains $selected)) {
                    $assertionOk = $false
                    $failureReason = "selected backend '$selected' outside allowed set [$($expected -join ', ')]"
                }

                if ($expected.Count -gt 0 -and $attempts.Count -gt 0) {
                    $lastAttemptBackend = [string]$attemptBackends[-1]
                    if ([string]::IsNullOrWhiteSpace($lastAttemptBackend) -or ($expected -notcontains $lastAttemptBackend)) {
                        $assertionOk = $false
                        $failureReason = "attempt sequence ended with unexpected backend '$lastAttemptBackend'"
                    }
                    $expectedIndex = @{}
                    for ($idx = 0; $idx -lt $expected.Count; $idx++) {
                        $expectedIndex[[string]$expected[$idx]] = $idx
                    }
                    $lastIndex = -1
                    foreach ($attemptBackend in $attemptBackends) {
                        if ([string]::IsNullOrWhiteSpace($attemptBackend)) {
                            continue
                        }
                        if (-not $expectedIndex.ContainsKey([string]$attemptBackend)) {
                            $assertionOk = $false
                            $failureReason = "attempt backend '$attemptBackend' outside expected route order [$($expected -join ', ')]"
                            break
                        }
                        if ($expectedIndex[$attemptBackend] -lt $lastIndex) {
                            $assertionOk = $false
                            $failureReason = "attempt order violates route order: $($attemptBackends -join ' -> ')"
                            break
                        }
                        $lastIndex = $expectedIndex[$attemptBackend]
                    }
                    if ($runResult.meta.fallback_attempted -ne $true -and $selected -and $expected.Count -gt 0 -and $expected[0] -ne $selected) {
                        $assertionOk = $false
                        $failureReason = "selected backend '$selected' does not match expected primary '$($expected[0])'"
                    }
                }

                if ($runResult.meta.fallback_attempted) {
                    $fallbackCount++
                }

                if ($runResult.meta.fallback_reason -and $failureReason -eq "") {
                    $failureReason = "fallback_reason=$($runResult.meta.fallback_reason)"
                }

                if (-not $assertionOk) {
                    $assertionFailures++
                }

                if ($selected) {
                    if (-not $backendCounts.ContainsKey($selected)) {
                        $backendCounts[$selected] = 0
                    }
                    $backendCounts[$selected]++
                } else {
                    if (-not $backendCounts.ContainsKey("unknown")) {
                        $backendCounts["unknown"] = 0
                    }
                    $backendCounts["unknown"]++
                }

                $tokensPerSecond = $runResult.meta.tokens_per_second
                if (-not $tokensPerSecond -and $runResult.response -and $runResult.response.usage -and $runResult.response.usage.completion_tokens -gt 0) {
                    $completionTokens = [double]$runResult.response.usage.completion_tokens
                    if ($runResult.elapsed_ms -gt 0) {
                        $tokensPerSecond = [math]::Round(($completionTokens / ($runResult.elapsed_ms / 1000.0)), 3)
                    }
                }

                $errorCode = if ($runResult.error_code) { [string]$runResult.error_code } else { "" }
                if ($runResult.meta.backend_error_codes.Count -gt 0) {
                    $errorCode = [string]($runResult.meta.backend_error_codes[0])
                }

                $rows += [ordered]@{
                    scenario = $ScenarioName
                    backend_disabled = ($DisableBackends -join "|")
                    case_id = [string]$case.id
                    route = [string]$case.route
                    expected_backends = $expected -join "|"
                    iteration = $i
                    status = $status
                    selected_backend = if ([string]::IsNullOrWhiteSpace($selected)) { "unknown" } else { $selected }
                    fallback_attempted = [bool]$runResult.meta.fallback_attempted
                    attempt_backends = if ($attemptBackends.Count -gt 0) { ($attemptBackends -join "|") } else { "" }
                    fallback_reason = $runResult.meta.fallback_reason
                    attempts = $attempts.Count
                    wait_ms = $runResult.meta.wait_ms
                    runtime_ms = $runResult.meta.runtime_ms
                    latency_ms = $runResult.elapsed_ms
                    guard_allow = if ($runResult.response -and $runResult.response.guard -and $runResult.response.guard.output -and $runResult.response.guard.output.allow -ne $null) { [bool]$runResult.response.guard.output.allow } else { $true }
                    tokens_per_second = $tokensPerSecond
                    concurrency_bucket = $runResult.meta.concurrency_bucket
                    selected_via = $runResult.meta.selected_via
                    assertion_ok = $assertionOk
                    assertion_failure = $failureReason
                    error_code = $errorCode
                    response_id = if ($runResult.response -and $runResult.response.id) { [string]$runResult.response.id } else { "" }
                    failure_stage = $runResult.failure_stage
                    request_error = if ($runResult.raw_error) { [string]$runResult.raw_error } else { "" }
                    backend_plan = if ($runResult.meta.backend_plan) { ($runResult.meta.backend_plan -join "|") } else { "" }
                }
            }
        }

        $summary = [ordered]@{
            scenario = $ScenarioName
            disabled_backends = ($DisableBackends -join "|")
            total_requests = $rows.Count
            succeeded = $scenarioSucceeded
            failed = $scenarioFailed
            fallback_count = $fallbackCount
            assertion_failures = $assertionFailures
            backend_counts = $backendCounts
        }

        return [ordered]@{
            scenario = $ScenarioName
            summary = $summary
            rows = $rows
            assertion_failed = ($assertionFailures -gt 0)
        }
    } finally {
        if (-not $SkipServiceControl -and $servicesToStop.Count -gt 0) {
            Write-Host "Scenario '$ScenarioName': restoring services $($servicesToStop -join ', ')"
            Invoke-ComposeCommand -ComposeAction "start" -Services $servicesToStop | Out-Null
            Start-Sleep -Milliseconds 800
        }
    }
}

function Write-CsvOutput {
    param(
        [string]$Path,
        [object[]]$Rows
    )

    if (-not $Rows -or $Rows.Count -eq 0) {
        return
    }

    $csvHeaders = $Rows[0].Keys
    $lines = @()
    $lines += ($csvHeaders -join ",")
    foreach ($row in $Rows) {
        $values = foreach ($key in $csvHeaders) {
            $value = $row[$key]
            if ($value -eq $null) {
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

if (-not (Test-Path -LiteralPath $OutputDir -PathType Container) -and (-not [string]::IsNullOrWhiteSpace($OutputDir))) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmss")
$allResults = @()
$allSummaries = @()
$assertionFailed = $false

if ($SkipServiceControl) {
    Write-Host "Service control is disabled. Ensure the harness stack is already running with your target state."
}

foreach ($scenario in @(
    @{name = "baseline"; disable = @()},
    @{name = "npu_down"; disable = @("npu")},
    @{name = "gpu_down"; disable = @("gpu")},
    @{name = "cpu_down"; disable = @("cpu")},
    @{name = "npu_gpu_down"; disable = @("npu", "gpu")}
)) {
    Write-Host "==================================================="
    Write-Host ("Running scenario: " + $scenario.name + " | disable=" + (($scenario.disable | ForEach-Object { $_ }) -join ","))
    $result = Run-Scenario -ScenarioName $scenario.name -DisableBackends $scenario.disable
    $allSummaries += $result.summary
    $allResults += $result.rows
    if ($result.assertion_failed) {
        $assertionFailed = $true
    }
}

$final = [ordered]@{
    metadata = [ordered]@{
        started = (Get-Date).ToUniversalTime().ToString("o")
        host = $ServerHost
        port = $ServerPort
        iterations = $Iterations
        model = $Model
        dry_run = $DryRun.IsPresent
        request_timeout_seconds = $RequestTimeoutSeconds
        compose_file = $ComposeFile
        compose_profile = $ComposeProfile
        env_file = $EnvFile
        check_service_health = $CheckServiceHealth.IsPresent
        skip_service_control = $SkipServiceControl.IsPresent
    }
    summary = $allSummaries
    assertions_failed = if ($assertionFailed) { $true } else { $false }
    total_requests = $allResults.Count
    total_cases = $testCases.Count
    cases = $allResults
}

if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $jsonPath = Join-Path $OutputDir "multidevice-phase2-$stamp.json"
    $final | ConvertTo-Json -Depth 20 | Set-Content -Path $jsonPath -Encoding UTF8
    Write-Host "Wrote JSON report: $jsonPath"
    if ($WriteCsv.IsPresent) {
        $csvPath = Join-Path $OutputDir "multidevice-phase2-$stamp.csv"
        Write-CsvOutput -Path $csvPath -Rows $allResults
        Write-Host "Wrote CSV report: $csvPath"
    }
}

$final | ConvertTo-Json -Depth 20 | Write-Output

if ($FailOnAssertion.IsPresent -and $assertionFailed) {
    throw "Phase2 assertions failed."
}

