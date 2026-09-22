# -*- coding: utf-8 -*-
# Запускает NeuroVox.exe --selftest, показывает отчёт в логе шага и в Job Summary
# (Job Summary виден через GitHub API как output.summary — без скачивания артефакта),
# а если .exe упал ДО того, как успел создать файл отчёта, достаёт причину из
# журнала событий Windows, чтобы ошибка не осталась «невидимой».
#
# Параметры:
#   -ReportName  имя отчёта без расширения (используется и для имени файла, и в summary)
#   -ExtraArgs   дополнительные аргументы для --selftest (например, @("--models"))
#   -TimeoutMs   сколько миллисекунд ждать завершения процесса

param(
    [Parameter(Mandatory = $true)][string]$ReportName,
    [string[]]$ExtraArgs = @(),
    [Parameter(Mandatory = $true)][int]$TimeoutMs
)

$PSNativeCommandUseErrorActionPreference = $true
$ErrorActionPreference = "Stop"

$exe = Join-Path $PWD "dist\NeuroVox\NeuroVox.exe"
$out = Join-Path $PWD "${ReportName}.txt"
$startedAt = Get-Date

if (-not (Test-Path $exe)) {
    throw "Файл $exe не найден — шаг сборки PyInstaller не создал программу."
}

$exeArgs = @("--selftest", "--strict", "--out", $out) + $ExtraArgs
Write-Host "Запуск: $exe $($exeArgs -join ' ')"

$process = Start-Process -FilePath $exe -ArgumentList $exeArgs -PassThru -RedirectStandardOutput "${ReportName}.stdout.log" -RedirectStandardError "${ReportName}.stderr.log"
$null = $process.Handle   # без этого PowerShell может не запомнить код завершения
$finished = $process.WaitForExit($TimeoutMs)

if (-not $finished) {
    Write-Host "::error::Процесс не завершился за отведённое время — принудительно останавливаю."
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
}

$reportExists = Test-Path $out
$reportText = if ($reportExists) { Get-Content $out -Encoding UTF8 -Raw } else { $null }

Write-Host "::group::Отчёт самопроверки ($ReportName)"
if ($reportExists) {
    Get-Content $out -Encoding UTF8
} else {
    Write-Host "Файл отчёта не создан — программа завершилась раньше, чем успела его записать."
}
Write-Host "::endgroup::"

foreach ($stream in @("stdout", "stderr")) {
    $streamFile = "${ReportName}.${stream}.log"
    if ((Test-Path $streamFile) -and (Get-Item $streamFile).Length -gt 0) {
        Write-Host "::group::$stream процесса ($ReportName)"
        Get-Content $streamFile -Encoding UTF8
        Write-Host "::endgroup::"
    }
}

# Job Summary виден в интерфейсе GitHub и через API (output.summary) — без скачивания артефактов.
$summary = New-Object System.Text.StringBuilder
[void]$summary.AppendLine("## Самопроверка: $ReportName")
[void]$summary.AppendLine("")
[void]$summary.AppendLine("Код завершения: **$($process.ExitCode)** | уложился в срок: **$finished**")
[void]$summary.AppendLine("")
if ($reportExists) {
    [void]$summary.AppendLine('```')
    [void]$summary.AppendLine($reportText)
    [void]$summary.AppendLine('```')
} else {
    [void]$summary.AppendLine("Файл отчёта не создан. Программа завершилась раньше, чем успела его записать —")
    [void]$summary.AppendLine("смотрите раздел «Диагностика сбоя без отчёта» ниже.")
}
Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value $summary.ToString() -Encoding UTF8

if (-not $reportExists -or -not $finished -or $process.ExitCode -ne 0) {
    if (-not $reportExists) {
        Write-Host "::group::Диагностика сбоя без отчёта"
        Write-Host "Ищу записи об аварийном завершении NeuroVox.exe в журнале событий Windows..."
        try {
            $sinceTime = $startedAt.AddSeconds(-2)
            $events = Get-WinEvent -FilterHashtable @{ LogName = "Application"; StartTime = $sinceTime } -ErrorAction SilentlyContinue |
                Where-Object { $_.Message -match "NeuroVox" } |
                Select-Object -First 10
            if ($events) {
                $eventLines = New-Object System.Text.StringBuilder
                [void]$eventLines.AppendLine("")
                [void]$eventLines.AppendLine("### Записи журнала событий Windows")
                foreach ($event in $events) {
                    $header = "--- $($event.TimeCreated) | $($event.ProviderName) | $($event.LevelDisplayName) ---"
                    Write-Host $header
                    Write-Host $event.Message
                    [void]$eventLines.AppendLine("")
                    [void]$eventLines.AppendLine($header)
                    [void]$eventLines.AppendLine($event.Message)
                }
                Add-Content -Path $env:GITHUB_STEP_SUMMARY -Value $eventLines.ToString() -Encoding UTF8
            } else {
                Write-Host "Подходящих записей в журнале событий не найдено."
            }
        } catch {
            Write-Host "Не удалось прочитать журнал событий: $_"
        }
        Write-Host "::endgroup::"
    }
    if (-not $finished) {
        throw "${ReportName}: программа не завершилась за отведённое время."
    }
    throw "${ReportName}: самопроверка не пройдена (код $($process.ExitCode))."
}
