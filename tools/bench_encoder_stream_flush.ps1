# A/B benchmark: streaming bitstream flush vs legacy accumulate-in-RAM.
#
# For each variant (Short / Long), runs the same encode twice -- once with
# --encoder-stream-flush=true and once with =false -- sampling working set + private
# bytes every 5 s. Captures wall time, peak memory, output file size, and frame count.
#
# Usage:
#   ./tools/bench_encoder_stream_flush.ps1                # both variants
#   ./tools/bench_encoder_stream_flush.ps1 -Variant Short # quick smoke
#   ./tools/bench_encoder_stream_flush.ps1 -Variant Long  # 4K dual-GPU

param(
    [ValidateSet('Short', 'Medium', 'Stress', 'Long', 'Both')]
    [string]$Variant = 'Both',
    [double]$SampleIntervalSeconds = 1.0,
    # Optional GPU index passed to Cave as --gpu N. Use to pin a single-GPU variant
    # to the non-primary card so the bench doesn't compete with the display.
    [int]$Gpu = -1,
    # When set, do NOT pass --keep-h264-only so the post-encode ffmpeg step muxes a
    # .mp4 alongside the .h264. Use when you want to play the bench outputs back to
    # compare streaming vs legacy visually.
    [switch]$KeepVideo
)

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'

# Per user rule (2026-05-11): all bench artifacts live inside the CAVE repo.
# Logs / CSVs / sampling traces -> data/bench/<name>/<timestamp>/...
# Video output (.h264 / .mp4)   -> videos/<name>/<timestamp>/...
$benchLogRoot = Join-Path (Get-Location) "data\bench\encoder_stream_flush\$timestamp"
New-Item -ItemType Directory -Force -Path $benchLogRoot | Out-Null

$videoRoot = Join-Path (Get-Location) "videos\bench_encoder_stream_flush\$timestamp"
New-Item -ItemType Directory -Force -Path $videoRoot | Out-Null

$csvPath = Join-Path $benchLogRoot 'results.csv'
"variant,streaming,wall_seconds,peak_working_set_mb,peak_private_bytes_mb,bitstream_bytes,frame_count,exit_code,output_path" | Out-File -FilePath $csvPath -Encoding utf8

$caveExe = Join-Path (Get-Location) 'build\Release\Cave.exe'
if (-not (Test-Path $caveExe)) {
    Write-Host "ERROR: Cave.exe not found at $caveExe"
    exit 1
}

# Build the run lists. Each entry is a hashtable; we iterate over the variants the
# user asked for and run streaming=false, then streaming=true for each.
$shortArgs  = @('--mode','video','--grid','129','--spawn','25','--spawn-mode','filled','--neighborhood','CW','--birth','1,2','--max-cs','12','--width','640','--height','360','--fps','20','--duration','600')
# Medium: single-GPU, 4K, 2 min of video at grid 1001. Bitstream grows to ~500-800 MB
# during the encode -- enough to make the RAM win visible without a 26-min commitment.
# Honors --gpu N if -Gpu was passed (so the user can pin to the non-primary card).
$mediumArgs = @('--mode','video','--grid','1001','--spawn','25','--spawn-mode','filled','--neighborhood','CW','--birth','1,2','--max-cs','12','--width','3840','--height','2160','--fps','20','--duration','2400')
# Stress: dual-GPU, 4K, 1 min of video at grid 3901. Final integration check at near-
# max grid scale. Mirrors the proven 4201^3 baseline shape but slightly smaller so
# VRAM headroom stays comfortable on dual RTX 8000.
$stressArgs = @('--mode','video','--grid','3901','--spawn','25','--spawn-mode','filled','--neighborhood','CW','--birth','1,2','--max-cs','12','--width','3840','--height','2160','--fps','20','--duration','1200','--dual-gpu')
$longArgs   = @('--mode','video','--grid','4201','--spawn','25','--spawn-mode','filled','--neighborhood','CW','--birth','1,2','--max-cs','12','--width','3840','--height','2160','--fps','20','--duration','1200','--dual-gpu')

if ($Gpu -ge 0) {
    $shortArgs  += @('--gpu', "$Gpu")
    $mediumArgs += @('--gpu', "$Gpu")
    # Long uses --dual-gpu so --gpu is intentionally not appended there.
}

$runs = @()
if ($Variant -eq 'Short' -or $Variant -eq 'Both') {
    $runs += @{ VariantName='Short'; Streaming=$false; CaveArgs=$shortArgs }
    $runs += @{ VariantName='Short'; Streaming=$true;  CaveArgs=$shortArgs }
}
if ($Variant -eq 'Medium') {
    $runs += @{ VariantName='Medium'; Streaming=$false; CaveArgs=$mediumArgs }
    $runs += @{ VariantName='Medium'; Streaming=$true;  CaveArgs=$mediumArgs }
}
if ($Variant -eq 'Stress') {
    $runs += @{ VariantName='Stress'; Streaming=$false; CaveArgs=$stressArgs }
    $runs += @{ VariantName='Stress'; Streaming=$true;  CaveArgs=$stressArgs }
}
if ($Variant -eq 'Long' -or $Variant -eq 'Both') {
    $runs += @{ VariantName='Long'; Streaming=$false; CaveArgs=$longArgs }
    $runs += @{ VariantName='Long'; Streaming=$true;  CaveArgs=$longArgs }
}

foreach ($run in $runs) {
    $variantName = $run.VariantName
    $streaming = $run.Streaming
    $caveArgs = $run.CaveArgs

    $streamingLabel = if ($streaming) { 'streaming' } else { 'legacy' }
    $runLabel = "${variantName}_${streamingLabel}"

    # Per-run log dir (CSV + stdout/stderr) outside the repo.
    $logDir = Join-Path $benchLogRoot $runLabel
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null

    # Per-run video output dir (.h264 / .mp4) inside the repo at videos/...
    $videoDir = Join-Path $videoRoot $runLabel
    New-Item -ItemType Directory -Force -Path $videoDir | Out-Null

    $stdoutPath = Join-Path $logDir 'stdout.log'
    $stderrPath = Join-Path $logDir 'stderr.log'
    $samplesPath = Join-Path $logDir 'memory_samples.csv'
    "elapsed_s,working_set_mb,private_bytes_mb" | Out-File -FilePath $samplesPath -Encoding utf8

    $flagValue = if ($streaming) { 'true' } else { 'false' }
    $fullArgs = $caveArgs + @('--output', "$videoDir\", '--headless', "--encoder-stream-flush=$flagValue")
    if (-not $KeepVideo) { $fullArgs += '--keep-h264-only' }

    Write-Host "[$runLabel] starting"

    Get-Process Cave -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 200

    $startTime = Get-Date
    $proc = Start-Process -FilePath $caveExe -ArgumentList $fullArgs -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    $procId = $proc.Id

    # Sample via Get-Process -Id because Start-Process's returned handle does not
    # always refresh its memory counters via .Refresh() (handle is detached when the
    # process redirects stdout/stderr). Get-Process opens a fresh handle each poll.
    $peakWs = 0L
    $peakPm = 0L
    while (-not $proc.HasExited) {
        $current = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($current) {
            $ws = $current.WorkingSet64
            $pm = $current.PrivateMemorySize64
            if ($ws -gt $peakWs) { $peakWs = $ws }
            if ($pm -gt $peakPm) { $peakPm = $pm }
            $elapsed = ((Get-Date) - $startTime).TotalSeconds
            ("{0:F1},{1:F1},{2:F1}" -f $elapsed, ($ws / 1MB), ($pm / 1MB)) | Out-File -FilePath $samplesPath -Encoding utf8 -Append
        }
        Start-Sleep -Seconds $SampleIntervalSeconds
    }
    $proc.WaitForExit()
    $wallSeconds = ((Get-Date) - $startTime).TotalSeconds

    # Start-Process + redirect can leave .ExitCode unpopulated even after WaitForExit.
    # If it's null, infer success from output: nonzero bitstream + matching frame count
    # at the end is good enough for the bench's pass criteria.
    $exitCode = $proc.ExitCode
    if ($null -eq $exitCode) { $exitCode = 'n/a' }

    # Look for .h264 first (legacy / --keep-h264-only path); fall back to .mp4 when
    # -KeepVideo was set and ffmpeg removed the .h264 after muxing.
    $videoFiles = @(Get-ChildItem -Path $videoDir -Filter '*.h264' -File)
    if (-not $videoFiles) { $videoFiles = @(Get-ChildItem -Path $videoDir -Filter '*.mp4' -File) }
    $bitstreamBytes = if ($videoFiles) { ($videoFiles | Measure-Object -Property Length -Sum).Sum } else { 0 }
    $outputFile = if ($videoFiles) { $videoFiles[0].FullName } else { '' }

    $frameCount = 0
    if ($outputFile -ne '') {
        $ffprobeOutput = & ffprobe -v error -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 $outputFile
        if ($ffprobeOutput) { $frameCount = [int]$ffprobeOutput }
    }

    $peakWsMb = [Math]::Round($peakWs / 1MB, 1)
    $peakPmMb = [Math]::Round($peakPm / 1MB, 1)
    $row = "$variantName,$streamingLabel,$([Math]::Round($wallSeconds,2)),$peakWsMb,$peakPmMb,$bitstreamBytes,$frameCount,$exitCode,$outputFile"
    $row | Out-File -FilePath $csvPath -Encoding utf8 -Append

    Write-Host "[$runLabel] done: wall=$([Math]::Round($wallSeconds,1))s peakWS=${peakWsMb}MB peakPM=${peakPmMb}MB bytes=$bitstreamBytes frames=$frameCount exit=$exitCode"
}

# Compare each variant's two rows
$allRows = Import-Csv -Path $csvPath
foreach ($v in @('Short','Medium','Stress','Long')) {
    $streamRow = $allRows | Where-Object { $_.variant -eq $v -and $_.streaming -eq 'streaming' }
    $legacyRow = $allRows | Where-Object { $_.variant -eq $v -and $_.streaming -eq 'legacy' }
    if (-not $streamRow -or -not $legacyRow) { continue }

    $wallDeltaPct = (([double]$streamRow.wall_seconds - [double]$legacyRow.wall_seconds) / [double]$legacyRow.wall_seconds) * 100
    $peakWsDeltaMb = [double]$legacyRow.peak_working_set_mb - [double]$streamRow.peak_working_set_mb
    $sizeDelta = [Math]::Abs([long]$streamRow.bitstream_bytes - [long]$legacyRow.bitstream_bytes)
    $sizeDeltaPct = ($sizeDelta / [double]$legacyRow.bitstream_bytes) * 100
    $framesMatch = ($streamRow.frame_count -eq $legacyRow.frame_count)

    Write-Host ""
    Write-Host "=== [$v] comparison ==="
    Write-Host ("  wall delta:        {0:F2}% (streaming vs legacy)" -f $wallDeltaPct)
    Write-Host ("  peak WS savings:   {0:F0} MB (legacy - streaming)" -f $peakWsDeltaMb)
    Write-Host ("  bitstream delta:   $sizeDelta bytes ({0:F4}%)" -f $sizeDeltaPct)
    Write-Host "  frame count match: $framesMatch ($($streamRow.frame_count) vs $($legacyRow.frame_count))"
}

Write-Host ""
Write-Host "Results CSV: $csvPath"
