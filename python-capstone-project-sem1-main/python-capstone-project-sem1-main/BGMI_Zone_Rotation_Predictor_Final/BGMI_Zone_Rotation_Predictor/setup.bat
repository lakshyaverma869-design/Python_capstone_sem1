@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title BGMI Zone Rotation Predictor - Setup

echo ============================================================
echo   BGMI Zone Rotation Predictor - First-time setup
echo ============================================================
echo.
echo Setup installs pinned packages and may need internet access.
echo.

set "PYTHON_CMD="
for %%V in (3.14 3.13 3.12 3.11) do (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -%%V"
        goto :python_found
    )
)
python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"
if not defined PYTHON_CMD goto :python_missing

:python_found
echo Using Python: %PYTHON_CMD%
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :venv_failed
)
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
"%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :pip_failed
"%VENV_PY%" -m compileall -q app.py train_model.py
if errorlevel 1 goto :compile_failed

echo.
echo Setup completed. Double-click run.bat to launch the application.
echo Historical CSV data is not bundled. Add authorized data\historical_zone_data.csv to enable predictions.
echo.
pause
exit /b 0

:python_missing
echo ERROR: Python 3.11, 3.12, 3.13, or 3.14 was not found.
echo Install 64-bit Python from https://www.python.org/downloads/windows/
echo During installation, enable the Python Launcher (py) or add Python to PATH.
goto :failed
:venv_failed
echo ERROR: Could not create .venv. Check Python installation and permissions.
goto :failed
:pip_failed
echo ERROR: Dependency installation failed. Check internet access.
goto :failed
:compile_failed
echo ERROR: Python compilation failed. See the error above.
goto :failed
:failed
echo.
echo Setup did not finish. This window will remain open for review.
echo.
pause
exit /b 1
