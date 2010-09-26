SHA384 Calculator Readme 

SHA384 Calculator is a simple console application that can calculate SHA384 hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's SHA384 hash specified using brute force method (dictionary search)
- Searching file using it's SHA384 hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- Calculating hash of the part of the file setting file part size and offset from the beginning
- File validation using SHA384 hash specified
- Calculation time of every file (option -t)
- SHA384 hash different case output (by default upper case)

Usage

sha384 [OPTION] ...

Available options:

-f [ --file ] <path>              Full path to file to calculate SHA384 of
-d [ --dir ] <path>	              Full path to directory to calculate SHA384 hash of all it's files
-e [ --exclude ] <mask>           File's mask (pattern) that must be excluded from SHA384 hash calculating. 
                                  It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>           File's mask (pattern) to process SHA384 hash calculating (other files will be excluded from process).
                                  It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>          String to calculate SHA384 hash of
-m [ --sha384 ] <SHA384 hash>     SHA384 hash to validate file(specified by option -f) or 
                                  restore original string (specified by option -c)
-a [ --dict ] arg                 Dictionary to restore original string using it's SHA384 hash
-n [ --min ] arg                  The minimal length of the string to restore. 1 by default
-x [ --max ] arg                  The maximal length of the string to restore. The length of the dictionary by default
-z [ --limit ] arg                set the limit in bytes of the part of the file to calculate hash for.
                                  The whole file by default will be applied
-q [ --offset ] arg               set start position in the file to calculate hash from. Zero by default
-h [ --search ] <SHA384 hash>     SHA384 hash to search file that matches it
-o [ --save ] arg                 save files' SHA384 hashes into the file specified by full path
-c [ --crack ]                    Restrore original string using it's SHA384 hash that specified by option sha384 (m)
-l [ --lower ]                    Output SHA384 using low case
-r [ --recursively ]              Scan subdirectories
-t [ --time ]                     Show calculation time (off by default)
-? [ --help ]                     Show help


Examples

Calculate SHA384 hash of string 123

sha384.exe -s 123


Calculate SHA384 hash of a file

sha384.exe -f file.txt


Calculate SHA384 hash of the part of the file (the first kilobyte)

sha384.exe -f file.txt -z 1024


Calculate SHA384 hash of the part of the file (one kilobyte skiping the first 512 bytes)

sha384.exe -f file.txt -z 1024 -q 512


Validate file using it's SHA384 hash

sha384.exe -f file.txt -m AFE0F32AFCA5A9A8422A82FAFB369C14342791EC780D8825465D3B8960A6EA6575EFF9DC5A7C8C563EC39E043E76CCC5


Calculate SHA384 of all files in c:\dir directory

sha384.exe -d c:\dir


Calculate SHA384 of all files in c:\dir directory including all it's subdirectories

sha384.exe -r -d c:\dir


Calculate SHA384 of all exe files in c:\dir directory

sha384.exe -d c:\dir -i *.exe


Calculate SHA384 of all files in c:\dir directory excluding files with tmp extension

sha384.exe -d c:\dir -e *.tmp


Calculate SHA384 of all exe and dll files in c:\dir directory

sha384.exe -d c:\dir -i *.exe;*.dll


Calculate SHA384 of all exe files in c:\dir directory excluding files beginning with bad

sha384.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known SHA384 hash

sha384.exe -d c:\ -r -h 9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F


Restore string by it's SHA384 hash using default dictionary

sha384.exe -с -m 9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F


Restore string by it's SHA384 hash using user defined dictionary

sha384.exe -с -m 9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F -a 0123456789


Restore string by it's SHA384 hash using user defined dictionary and string to restore min and max length

sha384.exe -с -m 9A0A82F0C0CF31470D7AFFEDE3406CC9AA8410671520B727044EDA15B4C25532A9B5CD8AAF9CEC4919D76255B6BFB00F -a 0123456789 -n 2 -x 6
