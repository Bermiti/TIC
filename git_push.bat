@echo off
REM Change directory to the Code Scripts folder
cd "C:\Users\Utilizador\OneDrive - Universidade de Lisboa\TICRiskAnalysis\Code Scripts"

REM Add all changes (files) in the folder
git add .

REM Commit the changes with a commit message
git commit -m "Automated commit from batch file"

REM Push the changes to the remote repository (assuming branch is 'main')
git push origin main

REM Pause so the window stays open to view any output or errors
pause
