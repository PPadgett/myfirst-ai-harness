[CmdletBinding()]
param(
    [string]$RemoteName = "origin",
    [string]$RemoteUrl = "",
    [double]$MaxFileSizeMB = 50,
    [double]$MaxTotalSizeMB = 200,
    [string[]]$RefUpdateLines = @()
)

Set-StrictMode -Version Latest

function Test-ZeroSha {
    param([string]$Sha)
    return ($Sha -match '^[0]{40}$')
}

function Invoke-GitCapture {
    param([string[]]$GitArgs)

    $output = & git @GitArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ("git {0} failed: {1}" -f ($GitArgs -join " "), (($output | Out-String).Trim()))
    }

    if ($null -eq $output) {
        return @()
    }

    if ($output -is [System.Array]) {
        return @($output | ForEach-Object { [string]$_ })
    }

    return @([string]$output)
}

if ($env:GIT_PUSH_MAX_FILE_MB) {
    $MaxFileSizeMB = [double]$env:GIT_PUSH_MAX_FILE_MB
}
if ($env:GIT_PUSH_MAX_TOTAL_MB) {
    $MaxTotalSizeMB = [double]$env:GIT_PUSH_MAX_TOTAL_MB
}

$maxFileBytes = [int64]($MaxFileSizeMB * 1MB)
$maxTotalBytes = [int64]($MaxTotalSizeMB * 1MB)
$refUpdateLinesBuffer = @()
foreach ($line in $RefUpdateLines) {
    if (-not [string]::IsNullOrWhiteSpace($line)) {
        $refUpdateLinesBuffer += $line.Trim()
    }
}

try {
    $rangeSpecs = @()
    $refSummaries = @()

    if ($refUpdateLinesBuffer.Count -eq 0) {
        $upstream = ""
        try {
            $upstream = (Invoke-GitCapture @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}") | Select-Object -First 1).Trim()
        } catch {
            $upstream = ""
        }

        if ([string]::IsNullOrWhiteSpace($upstream)) {
            Write-Host "[push-size-guard] No pre-push ref updates or upstream detected. Skipping check."
            exit 0
        }

        $rangeSpecs += "{0}..HEAD" -f $upstream
        $refSummaries += "HEAD against {0}" -f $upstream
    } else {
        foreach ($line in $refUpdateLinesBuffer) {
            $parts = $line -split '\s+'
            if ($parts.Count -lt 4) {
                continue
            }

            $localRef = $parts[0]
            $localSha = $parts[1]
            $remoteRef = $parts[2]
            $remoteSha = $parts[3]

            if (Test-ZeroSha $localSha) {
                # Deletion push, no new objects are introduced.
                continue
            }

            if (Test-ZeroSha $remoteSha) {
                $rangeSpecs += $localSha
                $refSummaries += "{0} -> {1} (new remote ref)" -f $localRef, $remoteRef
            } else {
                $rangeSpecs += "{0}..{1}" -f $remoteSha, $localSha
                $refSummaries += "{0} -> {1}" -f $localRef, $remoteRef
            }
        }
    }

    if ($rangeSpecs.Count -eq 0) {
        Write-Host "[push-size-guard] Push contains no new objects."
        exit 0
    }

    $objectPathById = @{}
    foreach ($spec in $rangeSpecs) {
        $lines = Invoke-GitCapture @("rev-list", "--objects", $spec)
        foreach ($entry in $lines) {
            if ([string]::IsNullOrWhiteSpace($entry)) {
                continue
            }

            $pieces = $entry -split ' ', 2
            $oid = $pieces[0].Trim()
            if ($oid.Length -eq 0) {
                continue
            }

            if (-not $objectPathById.ContainsKey($oid)) {
                $path = if ($pieces.Count -gt 1) { $pieces[1].Trim() } else { "" }
                $objectPathById[$oid] = $path
            }
        }
    }

    if ($objectPathById.Count -eq 0) {
        Write-Host "[push-size-guard] Push contains no new objects after range resolution."
        exit 0
    }

    $batchInput = $objectPathById.Keys
    $batchOutput = $batchInput | & git cat-file --batch-check="%(objectname) %(objecttype) %(objectsize)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ("git cat-file --batch-check failed: {0}" -f (($batchOutput | Out-String).Trim()))
    }

    $blobEntries = @()
    foreach ($line in $batchOutput) {
        $text = [string]$line
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }

        $parts = $text -split '\s+'
        if ($parts.Count -lt 3) {
            continue
        }

        $oid = $parts[0]
        $type = $parts[1]
        $sizeText = $parts[2]

        if ($type -ne "blob") {
            continue
        }

        $sizeBytes = [int64]$sizeText
        $path = [string]$objectPathById[$oid]
        if ([string]::IsNullOrWhiteSpace($path)) {
            $path = "(path unavailable for {0})" -f $oid.Substring(0, 12)
        }

        $blobEntries += [pscustomobject]@{
            Oid = $oid
            Path = $path
            SizeBytes = $sizeBytes
            SizeMB = [math]::Round($sizeBytes / 1MB, 2)
        }
    }

    $totalBytes = ($blobEntries | Measure-Object -Property SizeBytes -Sum).Sum
    if ($null -eq $totalBytes) {
        $totalBytes = 0
    }

    $offenders = @($blobEntries | Where-Object { $_.SizeBytes -gt $maxFileBytes } | Sort-Object SizeBytes -Descending)
    $largest = @($blobEntries | Sort-Object SizeBytes -Descending | Select-Object -First 10)

    Write-Host "[push-size-guard] Remote: $RemoteName $RemoteUrl"
    foreach ($summary in $refSummaries) {
        Write-Host "[push-size-guard] Ref update: $summary"
    }
    Write-Host ("[push-size-guard] New blob bytes in push: {0} MB" -f [math]::Round($totalBytes / 1MB, 2))
    Write-Host ("[push-size-guard] Limits: max file {0} MB, max total {1} MB" -f $MaxFileSizeMB, $MaxTotalSizeMB)

    if ($largest.Count -gt 0) {
        Write-Host "[push-size-guard] Largest new blobs:"
        $largest | ForEach-Object {
            Write-Host ("  {0,8} MB  {1}" -f $_.SizeMB, $_.Path)
        }
    }

    if ($offenders.Count -gt 0 -or $totalBytes -gt $maxTotalBytes) {
        Write-Host "[push-size-guard] BLOCKED: push exceeds size policy."
        if ($offenders.Count -gt 0) {
            Write-Host "[push-size-guard] Files above per-file threshold:"
            $offenders | ForEach-Object {
                Write-Host ("  {0,8} MB  {1}" -f $_.SizeMB, $_.Path)
            }
        }
        Write-Host "[push-size-guard] Tip: move artifacts/models to ignored paths or external storage before pushing."
        exit 1
    }

    Write-Host "[push-size-guard] PASS: push is within size policy."
    exit 0
} catch {
    Write-Host "[push-size-guard] ERROR: $($_.Exception.Message)"
    exit 2
}