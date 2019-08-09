call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"
msbuild hc.xml /p:Configuration=Release;GnuBasePath=C:\Gnu
src\x64\Release\_tst.exe
src\Release\_tst.exe