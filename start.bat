@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   NeuroVox - запуск
echo ============================================

rem Проверяем наличие Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ОШИБКА] Python не найден. Установите Python 3.10-3.12 с https://www.python.org/downloads/
    echo При установке отметьте галочку "Add python.exe to PATH".
    pause
    exit /b 1
)

rem Создаём виртуальное окружение при первом запуске
if not exist "venv\Scripts\python.exe" (
    echo Первый запуск: создаю виртуальное окружение...
    python -m venv venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать виртуальное окружение.
        pause
        exit /b 1
    )
    echo Устанавливаю зависимости. Это займёт несколько минут...
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось установить зависимости. Проверьте интернет и повторите запуск.
        rmdir /s /q venv
        pause
        exit /b 1
    )
)

"venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo Программа завершилась с ошибкой. Подробности в папке logs.
    pause
)
endlocal
