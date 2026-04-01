@echo off
echo ============================================
echo   IDS Bayesien Naif - Lancement...
echo ============================================
python main.py
if %errorlevel% neq 0 (
    echo.
    echo Erreur : verifiez que Python 3 est installe avec les dependances.
    echo Executez : pip install -r requirements.txt
)
pause
