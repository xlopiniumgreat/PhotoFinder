@echo off
cd /d "%~dp0"
".venv\Scripts\pythonw.exe" photo_finder_gui.py
if errorlevel 1 pause
