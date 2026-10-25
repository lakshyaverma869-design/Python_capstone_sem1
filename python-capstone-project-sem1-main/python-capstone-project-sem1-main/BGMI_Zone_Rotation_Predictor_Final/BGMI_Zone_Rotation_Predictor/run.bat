@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title BGMI Zone Rotation Predictor

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto :setup_required

"%VENV_PY%" -c "import numpy, pandas, streamlit, plotly"
if errorlevel 1 goto :dependency_error
if exist "data\historical_zone_data.csv" (
    "%VENV_PY%" -c "from train_model import load_zone_data, build_transitions; o=load_zone_data(); t,r=build_transitions(o); print('Historical observations:',len(o),'| valid transitions:',len(t),'| flagged:',len(r))"
    if errorlevel 1 goto :data_error
) else (
    echo No local CSV found. The app will try the public source in memory if internet access is available.
)

echo.
echo Starting Streamlit application...
echo Open http://localhost:8501 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
echo.
"%VENV_PY%" -m streamlit run app.py --server.headless true --server.port 8501
if errorlevel 1 goto :streamlit_error
exit /b 0

:setup_required
echo ERROR: Project environment .venv is missing. Double-click setup.bat first.
goto :failed
:dependency_error
echo ERROR: Required packages are missing or incompatible. Run setup.bat.
goto :failed
:data_error
echo ERROR: Historical CSV validation failed. Review the CSV schema, coordinates and phases.
goto :failed
:streamlit_error
echo ERROR: Streamlit stopped with an error. See the message above.
goto :failed
:failed
echo.
echo The application did not start. This window will remain open for review.
echo.
pause
exit /b 1
