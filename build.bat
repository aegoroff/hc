call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
setx GnuBasePath C:\Gnu
msbuild src\linq2hash.sln /p:Configuration=Release