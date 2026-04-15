@echo off
setlocal EnableDelayedExpansion
title RL-GACA GitHub Setup

echo.
echo ============================================================
echo   RL-GACA -- GitHub Upload Setup
echo ============================================================
echo.

:: ── Step 1: Check Git is installed ──────────────────────────
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed.
    echo.
    echo Please download and install Git from:
    echo   https://git-scm.com/download/win
    echo.
    echo Then re-run this script.
    pause
    exit /b 1
)
echo [OK] Git found.

:: ── Step 2: Ask for GitHub details ──────────────────────────
echo.
set /p GITHUB_USER=Enter your GitHub username: 
set /p REPO_NAME=Enter repository name (e.g. rl-gaca): 
set /p GIT_EMAIL=Enter your GitHub email: 
set /p GIT_NAME=Enter your name (for commits): 

echo.
echo You will need a GitHub Personal Access Token (PAT) as password.
echo Get one at: https://github.com/settings/tokens  (scope: repo)
echo.
set /p GIT_PAT=Paste your GitHub Personal Access Token here: 

:: ── Step 3: Configure git identity ──────────────────────────
git config --global user.email "%GIT_EMAIL%"
git config --global user.name "%GIT_NAME%"
echo [OK] Git identity configured.

:: ── Step 4: Init repo ────────────────────────────────────────
cd /d "%~dp0"
if exist ".git" (
    echo [OK] Git repo already initialised.
) else (
    git init
    echo [OK] Git repo initialised.
)

:: ── Step 5: Create .gitignore if missing ─────────────────────
if not exist ".gitignore" (
    echo results/ > .gitignore
    echo data/sumo_sim/ >> .gitignore
    echo __pycache__/ >> .gitignore
    echo *.pyc >> .gitignore
    echo .env >> .gitignore
    echo [OK] .gitignore created.
)

:: ── Step 6: Stage and commit ─────────────────────────────────
git add .
git commit -m "Initial release: RL-GACA IEEE Access 2025"
echo [OK] Files committed.

:: ── Step 7: Rename branch to main ────────────────────────────
git branch -M main

:: ── Step 8: Add remote ───────────────────────────────────────
git remote remove origin >nul 2>&1
git remote add origin https://%GITHUB_USER%:%GIT_PAT%@github.com/%GITHUB_USER%/%REPO_NAME%.git
echo [OK] Remote set to: https://github.com/%GITHUB_USER%/%REPO_NAME%

:: ── Step 9: Push ─────────────────────────────────────────────
echo.
echo Pushing to GitHub...
git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Common reasons:
    echo   1. Repository does not exist yet on GitHub.
    echo      Create it at: https://github.com/new
    echo      Name it exactly: %REPO_NAME%
    echo      Leave it EMPTY (no README, no .gitignore)
    echo      Then re-run this script.
    echo.
    echo   2. Wrong token or username.
    echo      Double-check at: https://github.com/settings/tokens
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   SUCCESS!
echo   Your repo is live at:
echo   https://github.com/%GITHUB_USER%/%REPO_NAME%
echo ============================================================
echo.
pause
