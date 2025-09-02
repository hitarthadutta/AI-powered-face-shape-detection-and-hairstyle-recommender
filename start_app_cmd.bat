@echo off
echo 🚀 Starting StyleAI Application...
echo.

echo 📦 Starting Backend Server...
start "Backend Server" cmd /k "cd backend & python app_consistent.py"

echo ⏳ Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo 🎨 Starting Frontend Server...
start "Frontend Server" cmd /k "cd frontend & npm start"

echo.
echo ✅ Both servers are starting!
echo.
echo 🌐 Backend: http://localhost:5000
echo 🎨 Frontend: http://localhost:3000
echo.
echo 📱 Open your browser and go to: http://localhost:3000
echo.
pause
