SHA512 Calculator Readme 

SHA512 Calculator is a simple console application that can calculate SHA512 hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's SHA512 hash specified using brute force method (dictionary search)
- Searching file using it's SHA512 hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- Calculating hash of the part of the file setting file part size and offset from the beginning
- File validation using SHA512 hash specified
- Calculation time of every file (option -t)
- SHA512 hash different case output (by default upper case)

Usage

sha512 [OPTION] ...

Available options:

-f [ --file ] <path>              Full path to file to calculate SHA512 of
-d [ --dir ] <path>	              Full path to directory to calculate SHA512 hash of all it's files
-e [ --exclude ] <mask>           File's mask (pattern) that must be excluded from SHA512 hash calculating. 
                                  It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>           File's mask (pattern) to process SHA512 hash calculating (other files will be excluded from process).
                                  It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>          String to calculate SHA512 hash of
-m [ --sha512 ] <SHA512 hash>     SHA512 hash to validate file(specified by option -f) or 
                                  restore original string (specified by option -c)
-a [ --dict ] arg                 Dictionary to restore original string using it's SHA512 hash
-n [ --min ] arg                  The minimal length of the string to restore. 1 by default
-x [ --max ] arg                  The maximal length of the string to restore. The length of the dictionary by default
-z [ --limit ] arg                set the limit in bytes of the part of the file to calculate hash for.
                                  The whole file by default will be applied
-q [ --offset ] arg               set start position in the file to calculate hash from. Zero by default
-h [ --search ] <SHA512 hash>     SHA512 hash to search file that matches it
-o [ --save ] arg                 save files' SHA512 hashes into the file specified by full path
-c [ --crack ]                    Restrore original string using it's SHA512 hash that specified by option sha512 (m)
-l [ --lower ]                    Output SHA512 using low case
-r [ --recursively ]              Scan subdirectories
-t [ --time ]                     Show calculation time (off by default)
-? [ --help ]                     Show help


Examples

Calculate SHA512 hash of string 123

sha512.exe -s 123


Calculate SHA512 hash of a file

sha512.exe -f file.txt


Calculate SHA512 hash of the part of the file (the first kilobyte)

sha512.exe -f file.txt -z 1024


Calculate SHA512 hash of the part of the file (one kilobyte skiping the first 512 bytes)

sha512.exe -f file.txt -z 1024 -q 512


Validate file using it's SHA512 hash

sha512.exe -f file.txt -m 6F6C7ED600C5E27023D63AF4F3943DDEF0309FE4CF2F6C4630985F06639FCDE93AB55EE9821D576C625A99AD62A0E3E9CC2396622B271BA8D94BC29866F46923


Calculate SHA512 of all files in c:\dir directory

sha512.exe -d c:\dir


Calculate SHA512 of all files in c:\dir directory including all it's subdirectories

sha512.exe -r -d c:\dir


Calculate SHA512 of all exe files in c:\dir directory

sha512.exe -d c:\dir -i *.exe


Calculate SHA512 of all files in c:\dir directory excluding files with tmp extension

sha512.exe -d c:\dir -e *.tmp


Calculate SHA512 of all exe and dll files in c:\dir directory

sha512.exe -d c:\dir -i *.exe;*.dll


Calculate SHA512 of all exe files in c:\dir directory excluding files beginning with bad

sha512.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known SHA512 hash

sha512.exe -d c:\ -r -h 3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2


Restore string by it's SHA512 hash using default dictionary

sha512.exe -с -m 3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2


Restore string by it's SHA512 hash using user defined dictionary

sha512.exe -с -m 3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2 -a 0123456789


Restore string by it's SHA512 hash using user defined dictionary and string to restore min and max length

sha512.exe -с -m 3C9909AFEC25354D551DAE21590BB26E38D53F2173B8D3DC3EEE4C047E7AB1C1EB8B85103E3BE7BA613B31BB5C9C36214DC9F14A42FD7A2FDB84856BCA5C44C2 -a 0123456789 -n 2 -x 6
