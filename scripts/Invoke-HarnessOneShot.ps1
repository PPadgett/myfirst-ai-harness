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
        [switch]$NoNetwork
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
            [int]$RequestTimeout
        )

        $env:HARNESS_ONESHOT_MODE = $ModeArg
        $env:HARNESS_ONESHOT_QUESTION = $QuestionText
        $env:HARNESS_ONESHOT_CONFIG = $ConfigPath
        $env:HARNESS_ONESHOT_HOST = $QuestionHost
        $env:HARNESS_ONESHOT_PORT = $QuestionPort.ToString()
        $env:HARNESS_ONESHOT_EXPLICIT_MODEL = $ExplicitModel
        $env:HARNESS_ONESHOT_STARTUP_TIMEOUT_SECONDS = $StartupTimeout.ToString()
        $env:HARNESS_ONESHOT_REQUEST_TIMEOUT_SECONDS = $RequestTimeout.ToString()

        $pythonScript = @'
import json
import os

from harness.oneshot import (
    build_chat_payload,
    build_health_url,
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

validate_oneshot_args(
    mode=mode,
    question=question,
    host=host,
    port=port,
    startup_timeout_seconds=startup_timeout,
    request_timeout_seconds=request_timeout,
)

resolved_model = resolve_model(config_path, explicit_model)
payload = build_chat_payload(question, explicit_model and resolved_model or None)

print(
    json.dumps(
        {
            "payload": payload,
            "health_url": build_health_url(host, port),
            "resolved_model": resolved_model,
            "explicit_model": explicit_model or "",
        },
        ensure_ascii=False,
    )
)
'@

        $result = & python -c $pythonScript
        if ($LASTEXITCODE -ne 0) {
            throw "Invoke-HarnessOneShot: failed while resolving runtime request shape."
        }

        return $result | ConvertFrom-Json -Depth 20
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
            -RequestTimeout $RequestTimeoutSeconds

        $payload = $helper.payload | ConvertTo-Json -Depth 20 -Compress
        $resolvedModel = $helper.explicit_model
        if ($resolvedModel -and $resolvedModel -ne "") {
            Write-Verbose ("Using explicit model: " + $resolvedModel)
        }

            $chatUrl = ("http://{0}:{1}/v1/chat/completions" -f $ServerHost.TrimEnd("/"), $Port)
        $serverProcess = $null
        try {
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

            $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
            $ready = $false
            while ([DateTime]::UtcNow -lt $deadline) {
                try {
                    $health = Invoke-WebRequest -Uri $helper.health_url -Method Get -TimeoutSec 3 -ErrorAction Stop
                    if ($health.StatusCode -eq 200) {
                        $ready = $true
                        break
                    }
                } catch {
                    Start-Sleep -Milliseconds 300
                }
            }

            if (-not $ready) {
                throw "Runtime one-shot startup timed out waiting for $($helper.health_url)."
            }

            try {
                $response = Invoke-WebRequest -Uri $chatUrl -Method Post -ContentType "application/json" -Body $payload -TimeoutSec $RequestTimeoutSeconds -ErrorAction Stop
            } catch {
                throw "Runtime one-shot request failed: $($_.Exception.Message)"
            }

            if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) {
                throw "Runtime one-shot request returned HTTP $($response.StatusCode): $($response.Content)"
            }
            return $response.Content | ConvertFrom-Json -Depth 20
        } finally {
            if ($null -ne $serverProcess -and -not $serverProcess.HasExited) {
                Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
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
