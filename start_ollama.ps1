# Launch Ollama in pure CPU mode to prevent CUDA memory allocation crashes
$env:OLLAMA_LLM_LIBRARY="cpu"
$env:OLLAMA_NUM_GPU="0"
$env:CUDA_VISIBLE_DEVICES=""

Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "Starting Ollama in CPU mode..." -ForegroundColor Cyan
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve

