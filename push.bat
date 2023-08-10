@echo off

git add --all

echo please input the commit message:

set /p message=

::echo begin push!

git commit -m "%message%"

git push

pause
