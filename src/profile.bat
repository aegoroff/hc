set PATH=%1
set CONFIG=%2
set MD5=md5.exe
set SHA1=sha1.exe

call "C:\Program Files\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" x86

cd /D %PATH%\md5\%CONFIG%
%MD5% -s 12345
%MD5% -c -m 81DC9BDB52D04DC20036DBD8313ED055
%MD5% -d %PATH%\%CONFIG%

cd /D %PATH%\sha1\%CONFIG%
%SHA1% -s 12345
%SHA1% -c -m 7110EDA4D09E062AA5E4A390B0A572AC0D2C0220
%SHA1% -d %PATH%\%CONFIG%
