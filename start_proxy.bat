@echo off
title SmartProxy
cd /d "%~dp0"
call "C:\Users\user\software\Anaconda\Scripts\activate.bat" build

rem Kill any existing process on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Killing process %%a on port 8000...
    taskkill /f /pid %%a 2>nul
    timeout /t 2 /nobreak >nul
    rem Check if the same PID still listening; if so, skip (can't kill)
    for /f "tokens=5" %%b in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
        if "%%b"=="%%a" (
            echo [WARN] Cannot kill process %%a, starting anyway...
            goto :start_proxy
        )
    )
)

:start_proxy

if not exist logs mkdir logs
"C:\Users\user\software\Anaconda\envs\build\python.exe" -m smart_proxy
pause
