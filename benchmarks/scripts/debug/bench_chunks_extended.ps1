$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$exe = Join-Path $repoRoot "build\Debug\Cave.exe"
$progressFile = Join-Path $repoRoot "build\Debug\data\search\bench_extended_progress.txt"
$resultsFile = Join-Path $repoRoot "build\Debug\data\search\bench_extended_results.csv"

"" | Out-File -FilePath $progressFile -Encoding utf8

$results = @()
foreach ($grid in @(25, 49)) {
    foreach ($p in @(16, 32)) {
        $output = "build/Debug/data/search/bench_g${grid}_p${p}"
        $absOutput = Join-Path $repoRoot $output
        Remove-Item -Recurse -Force $absOutput -ErrorAction SilentlyContinue

        $stdoutPath = Join-Path $repoRoot "build\Debug\data\search\bench_g${grid}_p${p}.log"
        "[$([DateTime]::Now.ToString('HH:mm:ss'))] starting grid=$grid p=$p" | Out-File -FilePath $progressFile -Append -Encoding utf8

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $proc = Start-Process -FilePath $exe -ArgumentList @(
            "--mode", "search",
            "--shape", "cube",
            "--grid", "$grid",
            "--neighborhood", "FE",
            "--chunks-per-config", "$p",
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

        $chunkLines = Get-Content $stdoutPath | Select-String "viable: (\d+), possible: (\d+), unviable: (\d+).+_c(\d+)_.+_gpu(\d+)_viable"
        $totalViable = 0; $totalPossible = 0; $totalUnviable = 0
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

        $row = [PSCustomObject]@{
            Grid = $grid
            Partitions = $p
            WallSeconds = $elapsed
            Viable = $totalViable
            Possible = $totalPossible
            Unviable = $totalUnviable
            Total = $totalViable + $totalPossible + $totalUnviable
        }
        $results += $row
        "[$([DateTime]::Now.ToString('HH:mm:ss'))] done   grid=$grid p=$p time=${elapsed}s viable=$totalViable possible=$totalPossible unviable=$totalUnviable" | Out-File -FilePath $progressFile -Append -Encoding utf8
    }
}

$results | Export-Csv -Path $resultsFile -NoTypeInformation -Encoding utf8
"" | Out-File -FilePath $progressFile -Append -Encoding utf8
"=== ALL DONE ===" | Out-File -FilePath $progressFile -Append -Encoding utf8
$results | Format-Table -AutoSize | Out-String | Out-File -FilePath $progressFile -Append -Encoding utf8
