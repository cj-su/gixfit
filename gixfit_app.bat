@echo off
rem Double-click to start the gixfit app (no console window).
rem If nothing appears, run gixfit_app_debug.bat to see the error message.
rem Python is not on the Windows PATH here, so look for it explicitly:
rem   1) the Python 3.11 install that has gixfit's packages, 2) the pyw launcher (C:\Windows), 3) PATH.
cd /d "%~dp0"
set "PYW=%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe"
if exist "%PYW%" (
    start "" "%PYW%" -m gixfit.app
    exit /b
)
where pyw >nul 2>nul
if not errorlevel 1 (
    start "" pyw -3.11 -m gixfit.app
    exit /b
)
where pythonw >nul 2>nul
if not errorlevel 1 (
    start "" pythonw -m gixfit.app
    exit /b
)
echo Could not find Python (pythonw.exe). Install Python 3.11 or edit PYW in this file.
pause
