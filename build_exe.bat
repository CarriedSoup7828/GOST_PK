@echo off
REM ============================================================
REM  Build EXE "PRV_Calc" (Windows). Requires Python 3.10+ and pip.
REM  (ASCII only - do not add Cyrillic text to this file)
REM ============================================================

echo [1/2] Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :err
python -m pip install numpy scipy matplotlib CoolProp PySide6 pyinstaller
if errorlevel 1 goto :err

echo [2/2] Building exe with PyInstaller...
cd /d "%~dp0"
pyinstaller --noconfirm --clean --onedir --name PRV_Calc --noconsole --collect-all CoolProp --hidden-import PySide6.QtSvg --hidden-import PySide6.QtOpenGLWidgets --exclude-module tkinter main.py
if errorlevel 1 goto :err

echo.
echo DONE. Run: dist\PRV_Calc\PRV_Calc.exe
pause
exit /b 0

:err
echo ERROR. Check that Python is installed and internet access is available.
pause
exit /b 1
