@echo off
rem IMS ICON Viewer -- double-click to open the app.
rem
rem The first run sets everything up (a private Python environment in .\venv with the
rem packages from requirements.txt), which takes a few minutes and needs the internet.
rem Every run after that just opens the window. If requirements.txt changes (after an
rem update), the packages are brought up to date automatically on the next start.
rem
rem Drop a .nc or .nc.bz2 file onto this icon to open that file directly.

setlocal
cd /d "%~dp0"

set "VPY=venv\Scripts\python.exe"
set "VPYW=venv\Scripts\pythonw.exe"
set "STAMP=venv\requirements-installed.txt"

if not exist "%VPYW%" goto setup
fc /b "requirements.txt" "%STAMP%" >nul 2>nul || goto install
goto run

:setup
echo.
echo  IMS ICON Viewer - first-time setup
echo  ----------------------------------
echo  Creating a private Python environment in:
echo    %CD%\venv
echo.
rem Prefer the "py" launcher: a bare "python" on Windows may be the Microsoft Store
rem placeholder, which opens the Store instead of running anything.
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
    echo  Python 3.11 or newer was not found.
    echo.
    echo  Install it from https://www.python.org/downloads/windows/
    echo  ^(tick "Add python.exe to PATH" in the installer^), then double-click this
    echo  file again.
    echo.
    pause
    exit /b 1
)
%PY% -m venv venv
if errorlevel 1 goto failed

:install
echo.
echo  Installing the packages the viewer needs (this takes a few minutes)...
echo.
"%VPY%" -m pip install --disable-pip-version-check --upgrade pip
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
rem One check with a console, so a broken install says why instead of the window
rem silently never appearing (the app itself runs without a console).
"%VPY%" -c "import imsicon.ui.main"
if errorlevel 1 goto failed
copy /y "requirements.txt" "%STAMP%" >nul
echo.
echo  Setup finished. Opening the viewer...

:run
start "" "%VPYW%" -m imsicon %*
exit /b 0

:failed
echo.
echo  Setup did not finish -- see the messages above. Check the internet connection
echo  and double-click this file again; it picks up where it stopped.
echo.
pause
exit /b 1
