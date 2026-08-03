<#
.SYNOPSIS
    Inventory locally-downloaded LLM / AI models across every known provider on Windows.

.DESCRIPTION
    Scans the well-known cache locations for Ollama, Hugging Face, Lemonade / AMD
    (Ryzen AI / GAIA), LM Studio, GPT4All, Jan, Microsoft Foundry Local and more,
    honours the relevant environment-variable overrides, and (optionally) sweeps
    common drives for stray model files (.gguf / .safetensors / .onnx / .bin / .pt).

    Read-only. Nothing is moved or deleted. Use the output to decide what to relocate
    to D:\LLM-Projects.

.PARAMETER Csv
    Also write the full inventory to a CSV file (default: .\model-inventory.csv).

.PARAMETER Deep
    Additionally run the generic file sweep across the drives in -SweepDrives.
    Slower, but catches models saved in arbitrary folders.

.PARAMETER SweepDrives
    Drive roots to scan when -Deep is used. Default: every fixed drive.

.PARAMETER MinSweepGB
    Ignore stray model files smaller than this during -Deep sweep. Default 0.5 GB.

.EXAMPLE
    .\Get-AllLocalModels.ps1

.EXAMPLE
    .\Get-AllLocalModels.ps1 -Deep -Csv

.NOTES
    No admin rights required. Runs against the CURRENT user's profile.
    Run in a second user's context too if models were pulled under another account.
#>

[CmdletBinding()]
param(
    [switch]$Csv,
    [switch]$Deep,
    [string[]]$SweepDrives,
    [double]$MinSweepGB = 0.5
)

$ErrorActionPreference = 'SilentlyContinue'
$results = New-Object System.Collections.Generic.List[object]

# ---------- helpers ----------------------------------------------------------

function Get-DirSizeBytes {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    ($([System.IO.DirectoryInfo]$Path).EnumerateFiles('*', 'AllDirectories') |
        Measure-Object -Property Length -Sum).Sum
}

# Size of a folder counting only REAL files, skipping symlink/junction reparse
# points. Needed for the Hugging Face hub, where /snapshots either symlinks into
# /blobs (measuring both double-counts) OR - for some AMD Ryzen AI repos - holds
# the real files directly while /blobs is empty (measuring blobs alone = 0).
# Counting real files once handles both layouts correctly.
function Get-RealFileBytes {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    $sum = [int64]0
    Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object { -not ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) } |
        ForEach-Object { $sum += $_.Length }
    return $sum
}

function Add-Model {
    param(
        [string]$Provider,
        [string]$Model,
        [double]$Bytes,
        [string]$Location,
        [string]$Note = ''
    )
    $results.Add([PSCustomObject]@{
        Provider = $Provider
        Model    = $Model
        SizeGB   = [math]::Round($Bytes / 1GB, 2)
        Location = $Location
        Note     = $Note
    })
}

# First existing path from a candidate list, honouring env-var overrides.
function Resolve-FirstPath {
    param([string[]]$Candidates)
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path -LiteralPath $c)) { return $c }
    }
    return $null
}

Write-Host "Scanning for local models..." -ForegroundColor Cyan

# ---------- 1. OLLAMA --------------------------------------------------------
# Default store is $env:USERPROFILE\.ollama\models ; overridable via OLLAMA_MODELS.
# Sizes come from the manifests (config + layer sizes). The AMD hybrid build also
# keeps npu/gpu/cpu backend copies alongside blobs.
$ollamaRoots = @(
    $env:OLLAMA_MODELS,
    (Join-Path $env:USERPROFILE '.ollama\models'),
    'C:\ProgramData\ollama\models'
) | Where-Object { $_ } | Select-Object -Unique

foreach ($root in $ollamaRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    $man = Join-Path $root 'manifests'
    if (Test-Path -LiteralPath $man) {
        Get-ChildItem $man -Recurse -File | ForEach-Object {
            $name = ($_.FullName.Substring($man.Length + 1)) -replace '\\', '/'
            try {
                $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
                $bytes = [int64]$j.config.size
                foreach ($l in $j.layers) { $bytes += [int64]$l.size }
                Add-Model 'Ollama' $name $bytes $root
            } catch {
                Add-Model 'Ollama' $name 0 $root 'manifest unreadable'
            }
        }
    }
    # AMD hybrid backend copies (npu / gpu / cpu) that live next to blobs
    foreach ($be in 'npu', 'gpu', 'cpu') {
        $bePath = Join-Path $root $be
        if (Test-Path -LiteralPath $bePath) {
            Get-ChildItem $bePath -Directory | ForEach-Object {
                Add-Model 'Ollama (AMD hybrid)' "$be/$($_.Name)" (Get-DirSizeBytes $_.FullName) $_.FullName "$be backend copy"
            }
        }
    }
}

# ---------- 2. HUGGING FACE HUB ----------------------------------------------
# Used by transformers / huggingface_hub / diffusers and many downstream tools
# (incl. Lemonade). Env overrides: HF_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE,
# TRANSFORMERS_CACHE. Snapshots are symlinks into /blobs, so measure /blobs to
# avoid double counting.
$hfCandidates = @(
    $env:HF_HUB_CACHE,
    $env:HUGGINGFACE_HUB_CACHE,
    $env:TRANSFORMERS_CACHE,
    ($(if ($env:HF_HOME) { Join-Path $env:HF_HOME 'hub' })),
    (Join-Path $env:USERPROFILE '.cache\huggingface\hub'),
    (Join-Path $env:LOCALAPPDATA 'huggingface\hub')
) | Where-Object { $_ } | Select-Object -Unique

foreach ($hub in $hfCandidates) {
    if (-not (Test-Path -LiteralPath $hub)) { continue }
    Get-ChildItem $hub -Directory -Filter 'models--*' | ForEach-Object {
        $name  = ($_.Name -replace '^models--', '') -replace '--', '/'
        # Count real files across the whole repo (blobs + any non-symlinked
        # snapshot files), skipping symlinks so nothing is double counted.
        $bytes = Get-RealFileBytes $_.FullName
        Add-Model 'HuggingFace' $name $bytes $hub
    }
    # datasets stored in the same hub layout are ignored (models only)
}

# ---------- 3. LEMONADE (AMD Lemonade SDK / Server) --------------------------
# Env override: LEMONADE_CACHE_DIR. Default cache ~/.cache/lemonade. Lemonade also
# reuses the HF hub above; here we count its own model store.
$lemonadeRoots = @(
    $env:LEMONADE_CACHE_DIR,
    (Join-Path $env:USERPROFILE '.cache\lemonade'),
    (Join-Path $env:LOCALAPPDATA 'lemonade')
) | Where-Object { $_ } | Select-Object -Unique

$lemonadeSkip = 'bin', 'builds', 'logs', 'cache', 'tmp', 'temp', '.locks'
foreach ($root in $lemonadeRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    $modelsDir = Resolve-FirstPath @((Join-Path $root 'models'), $root)
    Get-ChildItem $modelsDir -Directory |
        Where-Object { $lemonadeSkip -notcontains $_.Name.ToLower() } |
        ForEach-Object {
            Add-Model 'Lemonade/AMD' $_.Name (Get-DirSizeBytes $_.FullName) $root
        }
    # Loose GGUFs sitting directly in the cache root
    Get-ChildItem $modelsDir -File -Filter '*.gguf' | ForEach-Object {
        Add-Model 'Lemonade/AMD' $_.Name $_.Length $root
    }
}

# ---------- 4. AMD GAIA / RYZEN AI -------------------------------------------
$gaiaRoots = @(
    (Join-Path $env:USERPROFILE '.cache\gaia'),
    (Join-Path $env:LOCALAPPDATA 'GAIA'),
    (Join-Path $env:LOCALAPPDATA 'AMD\RyzenAI'),
    (Join-Path $env:USERPROFILE '.cache\ryzenai')
) | Where-Object { $_ } | Select-Object -Unique

foreach ($root in $gaiaRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    Get-ChildItem $root -Directory | ForEach-Object {
        Add-Model 'AMD GAIA/RyzenAI' $_.Name (Get-DirSizeBytes $_.FullName) $root
    }
}

# ---------- 5. LM STUDIO -----------------------------------------------------
# Newer builds: ~/.lmstudio/models ; older: ~/.cache/lm-studio/models .
# Layout is publisher/model/quant.gguf .
$lmRoots = @(
    (Join-Path $env:USERPROFILE '.lmstudio\models'),
    (Join-Path $env:USERPROFILE '.cache\lm-studio\models')
) | Where-Object { $_ } | Select-Object -Unique

foreach ($root in $lmRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    Get-ChildItem $root -Directory | ForEach-Object {
        $publisher = $_.Name
        Get-ChildItem $_.FullName -Directory | ForEach-Object {
            Add-Model 'LM Studio' "$publisher/$($_.Name)" (Get-DirSizeBytes $_.FullName) $root
        }
    }
}

# ---------- 6. GPT4All -------------------------------------------------------
$gpt4allRoots = @(
    (Join-Path $env:LOCALAPPDATA 'nomic.ai\GPT4All'),
    (Join-Path $env:USERPROFILE '.cache\gpt4all')
) | Where-Object { $_ } | Select-Object -Unique

foreach ($root in $gpt4allRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    Get-ChildItem $root -File -Include '*.gguf', '*.bin' -Recurse | ForEach-Object {
        Add-Model 'GPT4All' $_.Name $_.Length $root
    }
}

# ---------- 7. JAN -----------------------------------------------------------
$janRoots = @(
    (Join-Path $env:USERPROFILE 'jan\models'),
    (Join-Path $env:APPDATA 'Jan\data\models'),
    (Join-Path $env:LOCALAPPDATA 'Jan\data\models')
) | Where-Object { $_ } | Select-Object -Unique

foreach ($root in $janRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    Get-ChildItem $root -Directory | ForEach-Object {
        Add-Model 'Jan' $_.Name (Get-DirSizeBytes $_.FullName) $root
    }
}

# ---------- 8. MICROSOFT FOUNDRY LOCAL ---------------------------------------
$foundryRoots = @(
    (Join-Path $env:USERPROFILE '.foundry\cache\models'),
    (Join-Path $env:LOCALAPPDATA 'Microsoft\Foundry\cache\models')
) | Where-Object { $_ } | Select-Object -Unique

foreach ($root in $foundryRoots) {
    if (-not (Test-Path -LiteralPath $root)) { continue }
    Get-ChildItem $root -Directory | ForEach-Object {
        Add-Model 'Foundry Local' $_.Name (Get-DirSizeBytes $_.FullName) $root
    }
}

# ---------- 9. GENERIC / ASUS / etc. FILE SWEEP (-Deep) ----------------------
# Catch-all for anything the named providers miss: ASUS AI bundles, oobabooga,
# ComfyUI, koboldcpp, hand-placed GGUFs, etc. Reports large model files by size.
if ($Deep) {
    if (-not $SweepDrives) {
        $SweepDrives = (Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3').DeviceID |
            ForEach-Object { "$_\" }
    }
    Write-Host ("Deep sweep across: {0}" -f ($SweepDrives -join ', ')) -ForegroundColor Cyan
    $exts = '*.gguf', '*.safetensors', '*.onnx', '*.onnx_data', '*.pt', '*.pth', '*.bin', '*.gguf.part'
    $minBytes = $MinSweepGB * 1GB
    foreach ($drive in $SweepDrives) {
        if (-not (Test-Path -LiteralPath $drive)) { continue }
        Get-ChildItem $drive -File -Include $exts -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.Length -ge $minBytes } |
            ForEach-Object {
                # skip files already inside a provider cache we counted
                $inKnown = $false
                foreach ($k in '.ollama\models', 'huggingface\hub', '.cache\lemonade',
                                '.lmstudio', 'lm-studio', 'nomic.ai\GPT4All', 'jan\models', '.foundry') {
                    if ($_.FullName -like "*$k*") { $inKnown = $true; break }
                }
                if (-not $inKnown) {
                    Add-Model 'Loose file (sweep)' $_.Name $_.Length $_.DirectoryName
                }
            }
    }
}

# ---------- OUTPUT -----------------------------------------------------------
if ($results.Count -eq 0) {
    Write-Host "No local models found in any known location." -ForegroundColor Yellow
    return
}

$sorted = $results | Sort-Object SizeGB -Descending

Write-Host "`n===== ALL LOCAL MODELS =====" -ForegroundColor Green
$sorted | Format-Table Provider, Model, SizeGB, Location -Auto

Write-Host "`n===== TOTALS BY PROVIDER =====" -ForegroundColor Green
$results | Group-Object Provider | ForEach-Object {
    [PSCustomObject]@{
        Provider = $_.Name
        Models   = $_.Count
        TotalGB  = [math]::Round((($_.Group | Measure-Object SizeGB -Sum).Sum), 2)
    }
} | Sort-Object TotalGB -Descending | Format-Table -Auto

$grand = [math]::Round((($results | Measure-Object SizeGB -Sum).Sum), 2)
Write-Host ("GRAND TOTAL: {0} GB across {1} model entries" -f $grand, $results.Count) -ForegroundColor Green
Write-Host "(Ollama sizes are logical/manifest sizes; on-disk blobs may differ slightly.)" -ForegroundColor DarkGray

if ($Csv) {
    $out = Join-Path (Get-Location) 'model-inventory.csv'
    $sorted | Export-Csv -NoTypeInformation -Path $out
    Write-Host ("`nCSV written: {0}" -f $out) -ForegroundColor Cyan
}
