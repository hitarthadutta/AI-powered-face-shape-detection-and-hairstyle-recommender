@echo off
echo Starting StyleAI - Hairstyle Recommendation System...
echo.

echo Starting Backend (Flask)...
start "Backend" cmd /k "cd backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && python app.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo Starting Frontend (React)...
start "Frontend" cmd /k "cd frontend && npm install && npm start"

echo.
echo Both services are starting up!
echo Backend will be available at: http://localhost:5000
echo Frontend will be available at: http://localhost:3000
echo.
echo Press any key to close this window...
pause >nul
