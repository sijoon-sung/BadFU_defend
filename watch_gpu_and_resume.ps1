# watch_gpu_and_resume.ps1
# Monitor GPU idle status every 10 minutes and launch FUBA\resume_local.bat when idle.

$TargetDir = Join-Path $PSScriptRoot "FUBA"
$TargetBat = Join-Path $TargetDir "resume_local.bat"
$LogsDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir | Out-Null }
$MonitorLog = Join-Path $LogsDir "gpu_monitor.log"
$RunLog = Join-Path $LogsDir "resume_local_run.log"

$IntervalSeconds = 600   # 10 minutes
$WatchedPids = @(13096, 33500)
$GpuUtilThreshold = 20   # %
$GpuMemThreshold = 2000  # MB

function Log-Message([string]$msg) {
    $time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$time] $msg"
    Write-Host $line
    Add-Content -Path $MonitorLog -Value $line -Encoding utf8
}

Log-Message "=== GPU Monitoring Daemon Started ==="
Log-Message "Target script: $TargetBat"
Log-Message "Interval: $IntervalSeconds seconds (10 min)"
Log-Message "Watched PIDs: $($WatchedPids -join ', ')"

function Test-GpuIdle {
    # Query nvidia-smi GPU utilization & memory first for status logging
    $util = $null
    $mem = $null
    try {
        $smiOutput = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>&1
        $parts = $smiOutput.Trim().Split(',')
        $util = [int]($parts[0].Trim())
        $mem = [int]($parts[1].Trim())
    } catch {
        Log-Message "Warning: nvidia-smi query failed: $_"
    }

    # Check 1: Watched PIDs
    $runningPids = @()
    foreach ($pidNum in $WatchedPids) {
        $p = Get-Process -Id $pidNum -ErrorAction SilentlyContinue
        if ($p) { $runningPids += $pidNum }
    }
    if ($runningPids.Count -gt 0) {
        return @{
            IsIdle = $false
            Reason = "Active watched PIDs: $($runningPids -join ', ')"
            Util = $util
            Mem = $mem
        }
    }

    # Check 3: compute apps
    $computeApps = & nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1
    $pythonCompute = @()
    foreach ($line in $computeApps) {
        if ($line -match "python") {
            $pythonCompute += $line.Trim()
        }
    }

    if ($pythonCompute.Count -gt 0) {
        return @{
            IsIdle = $false
            Reason = "Active python compute apps: $($pythonCompute -join '; ')"
            Util = $util
            Mem = $mem
        }
    }

    if ($util -ge $GpuUtilThreshold) {
        return @{
            IsIdle = $false
            Reason = "GPU util ($util%) >= threshold ($GpuUtilThreshold%)"
            Util = $util
            Mem = $mem
        }
    }

    if ($mem -ge $GpuMemThreshold) {
        return @{
            IsIdle = $false
            Reason = "GPU mem ($mem MB) >= threshold ($GpuMemThreshold MB)"
            Util = $util
            Mem = $mem
        }
    }

    return @{
        IsIdle = $true
        Reason = "GPU is idle (Util: $util%, Mem: $mem MB, No python compute apps)"
        Util = $util
        Mem = $mem
    }
}

$iteration = 0
while ($true) {
    $iteration++
    $check = Test-GpuIdle
    $utilStr = if ($check.Util -ne $null) { "$($check.Util)%" } else { "N/A" }
    $memStr = if ($check.Mem -ne $null) { "$($check.Mem) MB" } else { "N/A" }

    if (-not $check.IsIdle) {
        Log-Message "Check #$iteration - BUSY [Util: $utilStr, Mem: $memStr] - $($check.Reason). Next check in $IntervalSeconds sec."
        Start-Sleep -Seconds $IntervalSeconds
    } else {
        Log-Message "Check #$iteration - IDLE DETECTED [Util: $utilStr, Mem: $memStr]! Verifying in 30 seconds..."
        Start-Sleep -Seconds 30
        $secondCheck = Test-GpuIdle
        if (-not $secondCheck.IsIdle) {
            Log-Message "Temporary idle spike detected, still BUSY - $($secondCheck.Reason). Resuming wait."
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        Log-Message "Confirmed IDLE! Launching resume_local.bat..."
        Log-Message "Execution log will be streamed to: $RunLog"
        
        $env:NO_PAUSE = "1"
        $RunErrLog = Join-Path $LogsDir "resume_local_err.log"
        $process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c cd /d `"$TargetDir`" && `"$TargetBat`"" -RedirectStandardOutput $RunLog -RedirectStandardError $RunErrLog -PassThru -NoNewWindow
        
        Log-Message "Process started with PID: $($process.Id). Waiting for completion..."
        $process.WaitForExit()
        $exitCode = $process.ExitCode
        
        Log-Message "resume_local.bat finished with exit code: $exitCode."
        Log-Message "=== GPU Monitoring Daemon Complete ==="
        break
    }
}
