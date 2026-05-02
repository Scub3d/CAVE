$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$exe = Join-Path $repoRoot "build\Debug\Cave.exe"
$progressFile = Join-Path $repoRoot "build\Debug\data\search\bench_grand_progress.txt"
$resultsFile = Join-Path $repoRoot "build\Debug\data\search\bench_grand_results.csv"

"" | Out-File -FilePath $progressFile -Encoding utf8
"=== Grand cross-product sweep: workgroup size × chunks-per-config ===" | Out-File -FilePath $progressFile -Append -Encoding utf8
"Grid 49^3 FE K=2 cs 2-3 ticks 20 dual-GPU. Per-run timeout 900s." | Out-File -FilePath $progressFile -Append -Encoding utf8
"" | Out-File -FilePath $progressFile -Append -Encoding utf8

$workgroupSizes = @(32, 64, 128, 256, 512)
$partitionCounts = @(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
$totalRuns = $workgroupSizes.Count * $partitionCounts.Count
$runIndex = 0

$results = @()
foreach ($wg in $workgroupSizes) {
    foreach ($p in $partitionCounts) {
        $runIndex++
        $output = "build/Debug/data/search/grand_wg${wg}_p${p}"
        $absOutput = Join-Path $repoRoot $output
        Remove-Item -Recurse -Force $absOutput -ErrorAction SilentlyContinue

        $stdoutPath = Join-Path $repoRoot "build\Debug\data\search\grand_wg${wg}_p${p}.log"
        "[$([DateTime]::Now.ToString('HH:mm:ss'))] [$runIndex/$totalRuns] starting wg=$wg p=$p" | Out-File -FilePath $progressFile -Append -Encoding utf8

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath $exe -ArgumentList @(
            "--mode", "search",
            "--shape", "cube",
            "--grid", "49",
            "--neighborhood", "FE",
            "--chunks-per-config", "$p",
            "--search-workgroup-size", "$wg",
            "--max-rule-bits", "2",
            "--min-cs", "2",
            "--max-cs-range", "3",
            "--ticks", "20",
            "--dual-gpu",
            "--headless",
            "--output", $output
        ) -NoNewWindow -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError "$stdoutPath.err" -WorkingDirectory $repoRoot
        $timedOut = $false
        if (-not $proc.WaitForExit(900000)) {
            $proc.Kill()
            $timedOut = $true
        }
        $sw.Stop()
        $elapsed = [math]::Round($sw.Elapsed.TotalSeconds, 2)

        $totalViable = 0; $totalPossible = 0; $totalUnviable = 0
        if (-not $timedOut) {
            $chunkLines = Get-Content $stdoutPath | Select-String "viable: (\d+), possible: (\d+), unviable: (\d+).+_c(\d+)_.+_gpu(\d+)_viable"
            $seen = @{}
            foreach ($line in $chunkLines) {
                $key = "$($line.Matches[0].Groups[4].Value)_$($line.Matches[0].Groups[5].Value)"
                if (-not $seen.ContainsKey($key)) {
                    $seen[$key] = $true
                    $totalViable += [int]$line.Matches[0].Groups[1].Value
                    $totalPossible += [int]$line.Matches[0].Groups[2].Value
                    $totalUnviable += [int]$line.Matches[0].Groups[3].Value
                }
            }
        }

        $row = [PSCustomObject]@{
            WorkgroupSize = $wg
            Partitions = $p
            WallSeconds = $elapsed
            Viable = $totalViable
            Possible = $totalPossible
            Unviable = $totalUnviable
            Total = $totalViable + $totalPossible + $totalUnviable
            TimedOut = $timedOut
        }
        $results += $row
        $tag = if ($timedOut) { "TIMEOUT" } else { "done" }
        "[$([DateTime]::Now.ToString('HH:mm:ss'))] [$runIndex/$totalRuns] $tag wg=$wg p=$p time=${elapsed}s viable=$totalViable possible=$totalPossible unviable=$totalUnviable" | Out-File -FilePath $progressFile -Append -Encoding utf8

        # Persist after each run so partial data is recoverable on interrupt
        $results | Export-Csv -Path $resultsFile -NoTypeInformation -Encoding utf8

        # Safety: kill any leftover Cave.exe between runs
        Get-Process -Name Cave -ErrorAction SilentlyContinue | Stop-Process -Force
    }
}

"" | Out-File -FilePath $progressFile -Append -Encoding utf8
"=== ALL DONE ===" | Out-File -FilePath $progressFile -Append -Encoding utf8

# Pivoted heatmap: rows = workgroup, cols = partitions, cell = WallSeconds
$pivotPath = Join-Path $repoRoot "build\Debug\data\search\bench_grand_pivot.txt"
$header = ("wg \\ p`t" + ($partitionCounts -join "`t"))
$lines = @($header)
foreach ($wg in $workgroupSizes) {
    $row = @($wg)
    foreach ($p in $partitionCounts) {
        $cell = $results | Where-Object { $_.WorkgroupSize -eq $wg -and $_.Partitions -eq $p } | Select-Object -First 1
        $row += if ($cell -and -not $cell.TimedOut) { "$($cell.WallSeconds)" } else { "TO" }
    }
    $lines += ($row -join "`t")
}
$lines -join "`n" | Out-File -FilePath $pivotPath -Encoding utf8
$lines -join "`n" | Out-File -FilePath $progressFile -Append -Encoding utf8
