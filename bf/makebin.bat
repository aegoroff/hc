set SRC_PATH=%1
set PYINSTALLER_PATH=D:\soft\Programming\Python\pyinstaller-1.5.1

if "%1" == "" (set SRC_PATH=d:\prj\hc\bf)

%PYINSTALLER_PATH%\Makespec.py -F %SRC_PATH%\bf.py

%PYINSTALLER_PATH%\Build.py %SRC_PATH%\bf.spec

del /Q %SRC_PATH%\bf.spec

del /Q %SRC_PATH%\warnbf.txt

rd /S /Q %SRC_PATH%\build
