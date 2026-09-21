@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" photo_finder.py scan --config config.json --device cpu
set "PHOTO_FINDER_EXIT=%ERRORLEVEL%"
pause
exit /b %PHOTO_FINDER_EXIT%
