MD5 Calculator Readme 

MD5 Calculator is a simple console application that can calculate MD5 hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's MD5 hash specified using brute force method (dictionary search)
- Searching file using it's MD5 hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- File validation using MD5 hash specified
- Calculation time of every file (option -t)
- MD5 hash different case output (by default upper case)

Usage

md5 [OPTION] ...

Available options:

-f [ --file ] <path>        Full path to file to calculate MD5 of
-d [ --dir ] <path>	        Full path to directory to calculate MD5 hash of all it's files
-e [ --exclude ] <mask>     File's mask (pattern) that must be excluded from MD5 hash calculating. 
                            It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>     File's mask (pattern) to process MD5 hash calculating (other files will be excluded from process).
                            It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>    String to calculate MD5 hash of
-m [ --md5 ] <MD5 hash>     MD5 hash to validate file(specified by option -f) or 
                            restore original string (specified by option -c)
-a [ --dict ] arg           Dictionary to restore original string using it's MD5 hash
-n [ --min ] arg            The minimal length of the string to restore. 1 by default
-x [ --max ] arg            The maximal length of the string to restore. The length of the dictionary by default
-h [ --search ] <MD5 hash>  MD5 hash to search file that matches it
-o [ --save ] arg           save files' MD5 hashes into the file specified by full path
-c [ --crack ]              Restrore original string using it's MD5 hash that specified by option md5 (m)
-l [ --lower ]              Output MD5 using low case
-r [ --recursively ]        Scan subdirectories
-t [ --time ]               Show calculation time (off by default)
-? [ --help ]               Show help


Examples

Calculate MD5 hash of string 123

md5.exe -s 123


Calculate MD5 hash of a file

md5.exe -f file.txt


Validate file using it's MD5 hash

md5.exe -f file.txt -m E0C110627FA4B42189C8DFD717957537


Calculate MD5 of all files in c:\dir directory

md5.exe -d c:\dir


Calculate MD5 of all files in c:\dir directory including all it's subdirectories

md5.exe -r -d c:\dir


Calculate MD5 of all exe files in c:\dir directory

md5.exe -d c:\dir -i *.exe


Calculate MD5 of all files in c:\dir directory excluding files with tmp extension

md5.exe -d c:\dir -e *.tmp


Calculate MD5 of all exe and dll files in c:\dir directory

md5.exe -d c:\dir -i *.exe;*.dll


Calculate MD5 of all exe files in c:\dir directory excluding files beginning with bad

md5.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known MD5 hash

md5.exe -d c:\ -r -h 202CB962AC59075B964B07152D234B70


Restore string by it's MD5 hash using default dictionary

md5.exe -с -m 202CB962AC59075B964B07152D234B70


Restore string by it's MD5 hash using user defined dictionary

md5.exe -с -m 202CB962AC59075B964B07152D234B70 -a 0123456789


Restore string by it's MD5 hash using user defined dictionary and string to restore min and max length

md5.exe -с -m 202CB962AC59075B964B07152D234B70 -a 0123456789 -n 2 -x 6
