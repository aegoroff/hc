SHA1 Calculator Readme 

SHA1 Calculator is a simple console application that can calculate SHA1 hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's SHA1 hash specified using brute force method (dictionary search)
- Searching file using it's SHA1 hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- File validation using SHA1 hash specified
- Calculation time of every file (option -t)
- SHA1 hash different case output (by default upper case)

Usage

sha1 [OPTION] ...

Available options:

-f [ --file ] <path>        Full path to file to calculate SHA1 of
-d [ --dir ] <path>	        Full path to directory to calculate SHA1 hash of all it's files
-e [ --exclude ] <mask>     File's mask (pattern) that must be excluded from SHA1 hash calculating. 
                            It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>     File's mask (pattern) to process SHA1 hash calculating (other files will be excluded from process).
                            It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>    String to calculate SHA1 hash of
-m [ --sha1 ] <SHA1 hash>     SHA1 hash to validate file(specified by option -f) or 
                            restore original string (specified by option -c)
-a [ --dict ] arg           Dictionary to restore original string using it's SHA1 hash
-n [ --min ] arg            The minimal length of the string to restore. 1 by default
-x [ --max ] arg            The maximal length of the string to restore. The length of the dictionary by default
-h [ --search ] <SHA1 hash>  SHA1 hash to search file that matches it
-o [ --save ] arg           save files' SHA1 hashes into the file specified by full path
-c [ --crack ]              Restrore original string using it's SHA1 hash that specified by option sha1 (m)
-l [ --lower ]              Output SHA1 using low case
-r [ --recursively ]        Scan subdirectories
-t [ --time ]               Show calculation time (off by default)
-? [ --help ]               Show help


Examples

Calculate SHA1 hash of string 123

sha1.exe -s 123


Calculate SHA1 hash of a file

sha1.exe -f file.txt


Validate file using it's SHA1 hash

sha1.exe -f file.txt -m 274F856438363F4032C8B87CF6BF49CEB9B5AC3C


Calculate SHA1 of all files in c:\dir directory

sha1.exe -d c:\dir


Calculate SHA1 of all files in c:\dir directory including all it's subdirectories

sha1.exe -r -d c:\dir


Calculate SHA1 of all exe files in c:\dir directory

sha1.exe -d c:\dir -i *.exe


Calculate SHA1 of all files in c:\dir directory excluding files with tmp extension

sha1.exe -d c:\dir -e *.tmp


Calculate SHA1 of all exe and dll files in c:\dir directory

sha1.exe -d c:\dir -i *.exe;*.dll


Calculate SHA1 of all exe files in c:\dir directory excluding files beginning with bad

sha1.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known SHA1 hash

sha1.exe -d c:\ -r -h 40BD001563085FC35165329EA1FF5C5ECBDBBEEF


Restore string by it's SHA1 hash using default dictionary

sha1.exe -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF


Restore string by it's SHA1 hash using user defined dictionary

sha1.exe -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789


Restore string by it's SHA1 hash using user defined dictionary and string to restore min and max length

sha1.exe -с -m 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789 -n 2 -x 6
