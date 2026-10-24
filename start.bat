@echo off
echo ==================================================
echo   ONE BEYOND - Spatial 3D Room Reconstruction
echo   Prototype 1.0 (Localhost Server)
echo ==================================================

cd /d "%~dp0backend"
start "One Beyond Backend API" cmd /k "python app.py"

timeout /t 2 /nobreak >nul

cd /d "%~dp0frontend"
start "One Beyond Frontend" cmd /k "npm.cmd run dev"

echo.
echo ==================================================
echo   One Beyond is running!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:5000
echo ==================================================
