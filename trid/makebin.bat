set SRC_PATH=%1
set PYINSTALLER_PATH=D:\soft\Programming\Python\pyinstaller-1.5.1
set SRC=trid

if "%1" == "" (set SRC_PATH=d:\prj\hc\trid)

%PYINSTALLER_PATH%\Makespec.py -F %SRC_PATH%\%SRC%.py
%PYINSTALLER_PATH%\Build.py %SRC_PATH%\%SRC%.spec

del /Q %SRC_PATH%\%SRC%.spec

del /Q %SRC_PATH%\warn%SRC%.txt

copy %SRC_PATH%\config.json %SRC_PATH%\dist
copy %SRC_PATH%\regions.json %SRC_PATH%\dist

rd /S /Q %SRC_PATH%\build
