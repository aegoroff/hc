call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
msbuild src\hc.xml /p:Configuration=Release;GnuBasePath=C:\Gnu;Platform=x64
src\x64\Release\_tst.exe
src\Release\_tst.exe