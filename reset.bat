@echo off

::for %%a in (*) do (if /i "%%~xa" neq ".bat" del "%%a")

::for %%a in (*) do (if "%%a" neq "reset.bat" echo file "%%a" will be deleted)

for %%a in (*) do (if "%%a" neq "reset.bat" del "%%a")

::for /d %%a in (*) do (if "%%a" neq ".git" echo file folder "%%a" will be deleted)

for /d %%a in (*) do (if "%%a" neq ".git" rd /q /s "%%a")

git fetch --all

git reset --hard origin/master

git pull

pause
