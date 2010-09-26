SHA256 Calculator Readme 

SHA256 Calculator is a simple console application that can calculate SHA256 hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's SHA256 hash specified using brute force method (dictionary search)
- Searching file using it's SHA256 hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- Calculating hash of the part of the file setting file part size and offset from the beginning
- File validation using SHA256 hash specified
- Calculation time of every file (option -t)
- SHA256 hash different case output (by default upper case)

Usage

sha256 [OPTION] ...

Available options:

-f [ --file ] <path>              Full path to file to calculate SHA256 of
-d [ --dir ] <path>	              Full path to directory to calculate SHA256 hash of all it's files
-e [ --exclude ] <mask>           File's mask (pattern) that must be excluded from SHA256 hash calculating. 
                                  It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>           File's mask (pattern) to process SHA256 hash calculating (other files will be excluded from process).
                                  It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>          String to calculate SHA256 hash of
-m [ --sha256 ] <SHA256 hash>     SHA256 hash to validate file(specified by option -f) or 
                                  restore original string (specified by option -c)
-a [ --dict ] arg                 Dictionary to restore original string using it's SHA256 hash
-n [ --min ] arg                  The minimal length of the string to restore. 1 by default
-x [ --max ] arg                  The maximal length of the string to restore. The length of the dictionary by default
-z [ --limit ] arg                set the limit in bytes of the part of the file to calculate hash for.
                                  The whole file by default will be applied
-q [ --offset ] arg               set start position in the file to calculate hash from. Zero by default
-h [ --search ] <SHA256 hash>     SHA256 hash to search file that matches it
-o [ --save ] arg                 save files' SHA256 hashes into the file specified by full path
-c [ --crack ]                    Restrore original string using it's SHA256 hash that specified by option sha256 (m)
-l [ --lower ]                    Output SHA256 using low case
-r [ --recursively ]              Scan subdirectories
-t [ --time ]                     Show calculation time (off by default)
-? [ --help ]                     Show help


Examples

Calculate SHA256 hash of string 123

sha256.exe -s 123


Calculate SHA256 hash of a file

sha256.exe -f file.txt


Calculate SHA256 hash of the part of the file (the first kilobyte)

sha256.exe -f file.txt -z 1024


Calculate SHA256 hash of the part of the file (one kilobyte skiping the first 512 bytes)

sha256.exe -f file.txt -z 1024 -q 512


Validate file using it's SHA256 hash

sha256.exe -f file.txt -m 0A3B10B4A34A250A87B47D538333F4B06589171C7DFEEE26FF84CC82BAC874FB


Calculate SHA256 of all files in c:\dir directory

sha256.exe -d c:\dir


Calculate SHA256 of all files in c:\dir directory including all it's subdirectories

sha256.exe -r -d c:\dir


Calculate SHA256 of all exe files in c:\dir directory

sha256.exe -d c:\dir -i *.exe


Calculate SHA256 of all files in c:\dir directory excluding files with tmp extension

sha256.exe -d c:\dir -e *.tmp


Calculate SHA256 of all exe and dll files in c:\dir directory

sha256.exe -d c:\dir -i *.exe;*.dll


Calculate SHA256 of all exe files in c:\dir directory excluding files beginning with bad

sha256.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known SHA256 hash

sha256.exe -d c:\ -r -h A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3


Restore string by it's SHA256 hash using default dictionary

sha256.exe -с -m A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3


Restore string by it's SHA256 hash using user defined dictionary

sha256.exe -с -m A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3 -a 0123456789


Restore string by it's SHA256 hash using user defined dictionary and string to restore min and max length

sha256.exe -с -m A665A45920422F9D417E4867EFDC4FB8A04A1F3FFF1FA07E998E86F7F7A27AE3 -a 0123456789 -n 2 -x 6
