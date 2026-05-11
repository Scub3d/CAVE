$ErrorActionPreference = 'Stop'

$outDir = 'data\videos\potential_v2_4k\'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$logDir = "$env:USERPROFILE\.claude\projects\C--Users-solys-Documents-GitHub-CAVE\bench_archive\Release\data"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$batchLog = Join-Path $logDir 'potential_v2_4k_batch.log'

# Decoded from filenames in data/videos/potential_v2/.
# Each entry: name (for log/output subfolder), neighborhood, birth, survival, maxCS.
$jobs = @(
    @{ Name='1152921504606846995-2305843009213693971'; Nbhd='F';  Birth='1,2,5';      Survival='1,2,5';     MaxCs=3 },
    @{ Name='3458764513820540952-2305843009213759512'; Nbhd='FE'; Birth='4,5,17';     Survival='4,5';       MaxCs=3 },
    @{ Name='3458764513820540964-1152921504606857220'; Nbhd='FE'; Birth='3,12,14';    Survival='3,6';       MaxCs=2 },
    @{ Name='3458764513820540972-1152921504606916616'; Nbhd='FE'; Birth='4,13,17';    Survival='3,4,6';     MaxCs=2 },
    @{ Name='3458764513820541188-2305843009213703172'; Nbhd='FE'; Birth='3,11,14';    Survival='3,9';       MaxCs=3 },
    @{ Name='3458764513820541446-1152921504606848260'; Nbhd='FE'; Birth='3,9,11';     Survival='2,3,10';    MaxCs=2 },
    @{ Name='3458764513820541958-1152921504606848040'; Nbhd='FE'; Birth='4,6,11';     Survival='2,3,11';    MaxCs=2 },
    @{ Name='3458764513820545036-1152921504606879816'; Nbhd='FE'; Birth='4,7,16';     Survival='3,4,13';    MaxCs=2 },
    @{ Name='3458764513820545296-1152921504606849284'; Nbhd='FE'; Birth='3,9,12';     Survival='5,9,13';    MaxCs=2 },
    @{ Name='3458764513820607489-2305843009213694979'; Nbhd='FE'; Birth='1,2,11';     Survival='1,11,17';   MaxCs=3 }
)

"=== potential_v2 4K batch start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $batchLog
"Total jobs: $($jobs.Count) | grid 4201, spawn 25, 4K 20fps, 60s, dual-GPU" | Tee-Object -FilePath $batchLog -Append

$jobIndex = 0
foreach ($job in $jobs) {
    $jobIndex++
    $jobLabel = "[$jobIndex/$($jobs.Count)] $($job.Name)"
    "----- $jobLabel start: $(Get-Date -Format 'HH:mm:ss') -----" | Tee-Object -FilePath $batchLog -Append
    "Nbhd=$($job.Nbhd) B={$($job.Birth)} S={$($job.Survival)} maxCS=$($job.MaxCs)" | Tee-Object -FilePath $batchLog -Append

    Get-Process Cave -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 200

    $perJobLog = Join-Path $logDir "potential_v2_4k_$($job.Name).log"

    & .\build\Release\Cave.exe `
        --mode video `
        --grid 4201 `
        --spawn 25 `
        --spawn-mode filled `
        --neighborhood $job.Nbhd `
        --birth $job.Birth `
        --survival $job.Survival `
        --max-cs $job.MaxCs `
        --width 3840 `
        --height 2160 `
        --fps 20 `
        --duration 1200 `
        --output $outDir `
        --dual-gpu `
        --headless 2>&1 | Tee-Object -FilePath $perJobLog

    "----- $jobLabel done: $(Get-Date -Format 'HH:mm:ss') -----" | Tee-Object -FilePath $batchLog -Append
}

"=== potential_v2 4K batch complete: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Tee-Object -FilePath $batchLog -Append
