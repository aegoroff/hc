call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
msbuild src\linq2hash.sln /p:Configuration=Release;GnuBasePath=C:\Gnu
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86
msbuild src\linq2hash.sln /p:Configuration=Release;GnuBasePath=C:\Gnu
src\x64\Release\_tst.exe
src\Release\_tst.exe