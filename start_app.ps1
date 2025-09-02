Write-Host "🚀 Starting StyleAI Application..." -ForegroundColor Green
Write-Host ""

Write-Host "📦 Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python app_simple.py"

Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

Write-Host "🎨 Starting Frontend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm start"

Write-Host ""
Write-Host "✅ Both servers are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Backend: http://localhost:5000" -ForegroundColor Blue
Write-Host "🎨 Frontend: http://localhost:3000" -ForegroundColor Blue
Write-Host ""
Write-Host "📱 Open your browser and go to: http://localhost:3000" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
