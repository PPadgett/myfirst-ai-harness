<#
.SYNOPSIS
    Cold-archive unused local AI models to another drive (e.g. external/USB),
    fully reversibly and WITHOUT re-downloading. Nothing here talks to the
    internet. Every move is preview-able first and records how to restore it.

.DESCRIPTION
    Three model stores live on this machine:
      * Ollama            %USERPROFILE%\.ollama\models   (content-addressed blobs)
      * HuggingFace cache %USERPROFILE%\.cache\huggingface\hub (self-contained repo dirs)
      * stray dumps       .ollama\models\{cpu,gpu,npu}\<model> (plain folders)

    Plain folders (HF repos, the stray dumps) archive trivially.
    Ollama library models need blob-aware handling so shared layers used by a
    model you KEEP are never moved. Use Archive-OllamaModel for those.

    ALWAYS run with -Preview first. Real moves use robocopy /MOVE, which copies
    then deletes the source only after a verified per-file copy.

.NOTES
    Run elevated is not required for USB moves, but STOP the services first so
    files aren't locked:  Stop-ModelServices
    USB drive should be NTFS or exFAT (FAT32 can't hold the >4 GB shards).
#>

# ---- keep-set: models the benchmark actively uses. Edit if your config changes.
$Script:KeepOllamaModels = @('qwen3:4b')                       # $Model in the gauntlet
$Script:OllamaRoot = Join-Path $env:USERPROFILE '.ollama\models'
$Script:HfRoot     = Join-Path $env:USERPROFILE '.cache\huggingface\hub'

function Stop-ModelServices {
    # Release file locks before moving anything.
    Get-Process -Name 'ollama*','lemonade*','LemonadeServer','llama*' -EA SilentlyContinue |
        Stop-Process -Force -EA SilentlyContinue
    Start-Sleep -Seconds 2
    'Stopped Ollama / Lemonade / llama processes (if any were running).'
}

function Show-RemovableDrives {
    Get-Volume | Where-Object { $_.DriveType -eq 'Removable' -or $_.DriveType -eq 'Fixed' } |
        Where-Object DriveLetter |
        Select-Object DriveLetter, FileSystemLabel, FileSystem, DriveType,
            @{n='FreeGB';e={[math]::Round($_.SizeRemaining/1GB,1)}},
            @{n='SizeGB';e={[math]::Round($_.Size/1GB,1)}} |
        Sort-Object DriveType, DriveLetter | Format-Table -Auto
}

function Get-FolderGB { param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    [math]::Round((Get-ChildItem $Path -Recurse -File -EA SilentlyContinue |
        Measure-Object Length -Sum).Sum/1GB, 2)
}

# ---- Inventory ------------------------------------------------------------

function Get-OllamaModels {
    # Works even though `ollama list` is broken: reads manifests directly.
    $man = Join-Path $Script:OllamaRoot 'manifests'
    if (-not (Test-Path $man)) { Write-Warning "No manifests at $man"; return }
    Get-ChildItem $man -Recurse -File -EA SilentlyContinue | ForEach-Object {
        $rel  = $_.FullName.Substring($man.Length + 1) -replace '\\','/'
        # rel = registry.ollama.ai/library/<model>/<tag>  ->  <model>:<tag>
        $parts = $rel -split '/'
        $name = if ($parts.Count -ge 2) { "$($parts[-2]):$($parts[-1])" } else { $rel }
        try {
            $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
            $bytes = [int64]$j.config.size
            foreach ($l in $j.layers) { $bytes += [int64]$l.size }
            [PSCustomObject]@{ Model=$name; SizeGB=[math]::Round($bytes/1GB,2); Manifest=$_.FullName }
        } catch { [PSCustomObject]@{ Model=$name; SizeGB='unreadable'; Manifest=$_.FullName } }
    } | Sort-Object SizeGB -Descending
}

function Get-HFRepos {
    Get-ChildItem $Script:HfRoot -Directory -EA SilentlyContinue |
        Where-Object Name -like 'models--*' | ForEach-Object {
            [PSCustomObject]@{ Repo=$_.Name; SizeGB=(Get-FolderGB $_.FullName); Path=$_.FullName }
        } | Sort-Object SizeGB -Descending
}

# ---- Generic plain-folder archive / restore (HF repos, stray dumps) --------

function Archive-Folder {
    param(
        [Parameter(Mandatory)][string]$Source,       # folder to move off C:
        [Parameter(Mandatory)][string]$ArchiveRoot,  # e.g. E:\model-archive
        [switch]$Preview
    )
    if (-not (Test-Path $Source)) { Write-Warning "Missing: $Source"; return }
    $Source = (Resolve-Path $Source).Path
    $needGB = Get-FolderGB $Source
    $qual   = (Split-Path $ArchiveRoot -Qualifier)              # e.g. 'E:'
    $freeGB = [math]::Round((Get-Volume -DriveLetter $qual.TrimEnd(':')).SizeRemaining/1GB,2)
    if (-not $Preview -and $needGB -gt $freeGB) {
        Write-Warning "Not enough space on $qual ($freeGB GB free, need $needGB GB)."; return
    }
    $tag  = ($Source -replace '^[A-Za-z]:\\','') -replace '[\\/:]','__'
    $dest = Join-Path $ArchiveRoot $tag
    Write-Host ("{0}  {1,7:N2} GB  {2}  ->  {3}" -f ($(if($Preview){'PREVIEW'}else{'MOVE   '})), $needGB, $Source, $dest)
    if ($Preview) { robocopy $Source $dest /E /L /NFL /NDL /NJH /NJS /NP | Out-Null; return }

    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    @{ OriginalPath=$Source; ArchivedUtc=(Get-Date).ToUniversalTime().ToString('o'); SizeGB=$needGB } |
        ConvertTo-Json | Set-Content (Join-Path $dest '_restore.json')
    robocopy $Source $dest /E /MOVE /R:1 /W:1 /NFL /NDL /NJH /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Write-Warning "robocopy reported errors (code $LASTEXITCODE) - source left intact." ; return }
    "Archived -> $dest"
}

function Restore-Folder {
    param([Parameter(Mandatory)][string]$ArchivedDir)   # the ...\model-archive\<tag> folder
    $meta = Get-Content (Join-Path $ArchivedDir '_restore.json') -Raw | ConvertFrom-Json
    $target = $meta.OriginalPath
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    robocopy $ArchivedDir $target /E /MOVE /R:1 /W:1 /XF _restore.json /NFL /NDL /NJH /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { Write-Warning "robocopy errors (code $LASTEXITCODE)."; return }
    "Restored -> $target"
}

# ---- Blob-aware Ollama model archive --------------------------------------

function Get-OllamaModelBlobs {
    param([Parameter(Mandatory)][string]$Model)   # e.g. 'llama3:70b'
    $name,$tag = $Model -split ':',2
    if (-not $tag) { $tag = 'latest' }
    $mf = Get-ChildItem (Join-Path $Script:OllamaRoot 'manifests') -Recurse -File -EA SilentlyContinue |
          Where-Object { $_.FullName -match "\\$([regex]::Escape($name))\\$([regex]::Escape($tag))$" } |
          Select-Object -First 1
    if (-not $mf) { Write-Warning "Manifest not found for $Model"; return $null }
    $j = Get-Content $mf.FullName -Raw | ConvertFrom-Json
    $digests = @($j.config.digest) + @($j.layers.digest)
    [PSCustomObject]@{ Model=$Model; Manifest=$mf.FullName; Blobs=($digests | Where-Object {$_}) }
}

function Archive-OllamaModel {
    param(
        [Parameter(Mandatory)][string]$Model,        # model:tag to archive
        [Parameter(Mandatory)][string]$ArchiveRoot,
        [switch]$Preview
    )
    $target = Get-OllamaModelBlobs -Model $Model
    if (-not $target) { return }

    # Blobs referenced by any model we KEEP -> never move these.
    $keepBlobs = @{}
    foreach ($k in $Script:KeepOllamaModels) {
        $ki = Get-OllamaModelBlobs -Model $k
        if ($ki) { foreach ($b in $ki.Blobs) { $keepBlobs[$b] = $true } }
    }
    $moveBlobs = $target.Blobs | Where-Object { -not $keepBlobs.ContainsKey($_) }
    $blobDir   = Join-Path $Script:OllamaRoot 'blobs'
    $destBlobs = Join-Path $ArchiveRoot 'ollama\blobs'
    $destMan   = Join-Path $ArchiveRoot ('ollama\manifests\' + ($target.Manifest.Substring((Join-Path $Script:OllamaRoot 'manifests').Length+1)))

    $files = $moveBlobs | ForEach-Object { Join-Path $blobDir ($_ -replace ':','-') } | Where-Object { Test-Path $_ }
    $gb = [math]::Round((($files | ForEach-Object { (Get-Item $_).Length }) | Measure-Object -Sum).Sum/1GB, 2)
    Write-Host ("{0} Ollama {1}: {2} unshared blob(s), {3:N2} GB" -f ($(if($Preview){'PREVIEW'}else{'MOVE'})), $Model, $files.Count, $gb)
    if ($Preview) { $files | ForEach-Object { "  $_" }; "  + manifest $($target.Manifest)"; return }

    New-Item -ItemType Directory -Force -Path $destBlobs | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $destMan) | Out-Null
    foreach ($f in $files) { robocopy (Split-Path $f) $destBlobs (Split-Path $f -Leaf) /MOV /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null }
    Copy-Item $target.Manifest $destMan -Force
    @{ Model=$Model; OriginalManifest=$target.Manifest; MovedBlobs=$files.Count; SizeGB=$gb; ArchivedUtc=(Get-Date).ToUniversalTime().ToString('o') } |
        ConvertTo-Json | Set-Content (Join-Path $ArchiveRoot ("ollama\_restore-$($Model -replace '[:/]','_').json"))
    Remove-Item $target.Manifest -Force -EA SilentlyContinue
    "Archived Ollama $Model ($gb GB) -> $ArchiveRoot\ollama"
}
