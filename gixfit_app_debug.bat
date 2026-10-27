@echo off
rem Same as gixfit_app.bat but keeps a console open, so errors are visible.
cd /d "%~dp0"
set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%PY%" (
    "%PY%" -m gixfit.app
    goto :end
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -m gixfit.app
    goto :end
)
python -m gixfit.app
:end
pause
