$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$exe = Join-Path $repoRoot "build\Release\Cave.exe"
$searchDir = Join-Path $repoRoot "build\Release\data\search"
$progressFile = Join-Path $searchDir "ssbo_full_progress.txt"
$resultsFile = Join-Path $searchDir "ssbo_full_results.csv"

"" | Out-File -FilePath $progressFile -Encoding utf8
"=== SSBO comprehensive sweep p=1,4,8,16,32,64,128,256 (cube 49^3 FE K=2 wg=256 ticks=20 dual-GPU, NO image mode) - mirrors variant C range for direct row-by-row comparison ===" | Out-File -FilePath $progressFile -Append -Encoding utf8

$partitionCounts = @(1, 4, 8, 16, 32, 64, 128, 256)
$results = @()
foreach ($p in $partitionCounts) {
    $output = "build/Release/data/search/ssbo_full_p${p}"
    $absOutput = Join-Path $repoRoot $output
    Remove-Item -Recurse -Force $absOutput -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "${absOutput}cube" -ErrorAction SilentlyContinue

    $stdoutPath = Join-Path $searchDir "ssbo_full_p${p}.log"
    "[$([DateTime]::Now.ToString('HH:mm:ss'))] starting p=$p" | Out-File -FilePath $progressFile -Append -Encoding utf8

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $exe -ArgumentList @(
        "--mode", "search",
        "--shape", "cube",
        "--grid", "49",
        "--neighborhood", "FE",
        "--chunks-per-config", "$p",
        "--search-workgroup-size", "256",
        "--max-rule-bits", "2",
        "--min-cs", "2",
        "--max-cs-range", "3",
        "--ticks", "20",
        "--dual-gpu",
        "--headless",
        "--output", $output
    ) -NoNewWindow -PassThru -Wait -RedirectStandardOutput $stdoutPath -RedirectStandardError "$stdoutPath.err" -WorkingDirectory $repoRoot
    $sw.Stop()
    $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 2)

    $aggCandidate = Get-ChildItem -Path $searchDir -Recurse -Filter "*aggregate.jsonl" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*ssbo_full_p${p}cube*" } | Select-Object -First 1
    if (-not $aggCandidate) {
        "[$([DateTime]::Now.ToString('HH:mm:ss'))] FAILED p=$p" | Out-File -FilePath $progressFile -Append -Encoding utf8
        continue
    }
    $rows = Get-Content $aggCandidate.FullName | ForEach-Object { ConvertFrom-Json $_ }
    $rowCount = $rows.Count
    $totalRunsSum = ($rows | Measure-Object totalRuns -Sum).Sum
    $dups = ($rows | Where-Object { $_.totalRuns -gt 1 }).Count

    $results += [PSCustomObject]@{
        Partitions = $p
        WallSeconds = $elapsed
        Rows = $rowCount
        TotalRunsSum = $totalRunsSum
        Dups = $dups
    }
    "[$([DateTime]::Now.ToString('HH:mm:ss'))] done p=$p time=${elapsed}s rows=$rowCount totalRuns=$totalRunsSum dups=$dups" | Out-File -FilePath $progressFile -Append -Encoding utf8
    $results | Export-Csv -Path $resultsFile -NoTypeInformation -Encoding utf8
    Get-Process -Name Cave -ErrorAction SilentlyContinue | Stop-Process -Force
}

"" | Out-File -FilePath $progressFile -Append -Encoding utf8
"=== ALL DONE ===" | Out-File -FilePath $progressFile -Append -Encoding utf8
$results | Format-Table -AutoSize | Out-String | Out-File -FilePath $progressFile -Append -Encoding utf8
