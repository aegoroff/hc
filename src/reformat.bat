rem Reformats all files specified by path pattern
rem example: reformat.bat dir\*.cpp
for %%F in (%1) do (call uncrustify.bat %%F)