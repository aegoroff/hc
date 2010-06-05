@echo off

IF (%1)==() GOTO error
dir /b /ad %1 >nul 2>nul && GOTO indentDir
IF NOT EXIST %1 GOTO error
goto indentFile

:indentDir
set searchdir=%1

IF (%2)==() GOTO assignDefaultSuffix
set filesuffix=%2

GOTO run

:assignDefaultSuffix
::echo !!!!DEFAULT SUFFIX!!!
set filesuffix=*

:run
FOR /F "tokens=*" %%G IN ('DIR /B /S %searchdir%\*.%filesuffix%') DO (
echo Indenting file "%%G"
"%UNCRUSTIFY_PATH%/uncrustify.exe" -f "%%G" -c "uncrustify.cfg" -o indentoutput.tmp
move /Y indentoutput.tmp "%%G"

)
GOTO ende

:indentFile
echo Indenting one file %1
"%UNCRUSTIFY_PATH%/uncrustify.exe" -f "%1" -c "uncrustify.cfg" -o indentoutput.tmp
move /Y indentoutput.tmp "%1"


GOTO ende

:error
echo .
echo ERROR: As parameter given directory or file does not exist!
echo Syntax is: uncrustify.bat dirname filesuffix
echo Syntax is: uncrustify.bat filename
echo Example: uncrustify.bat temp cpp
echo .

:ende
