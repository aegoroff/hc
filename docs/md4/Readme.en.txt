MD4 Calculator Readme 

MD4 Calculator is a simple console application that can calculate MD4 hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's MD4 hash specified using brute force method (dictionary search)
- Searching file using it's MD4 hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- File validation using MD4 hash specified
- Calculation time of every file (option -t)
- MD4 hash different case output (by default upper case)

Usage

md4 [OPTION] ...

Available options:

-f [ --file ] <path>        Full path to file to calculate MD4 of
-d [ --dir ] <path>	        Full path to directory to calculate MD4 hash of all it's files
-e [ --exclude ] <mask>     File's mask (pattern) that must be excluded from MD4 hash calculating. 
                            It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>     File's mask (pattern) to process MD4 hash calculating (other files will be excluded from process).
                            It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>    String to calculate MD4 hash of
-m [ --md4 ] <MD4 hash>     MD4 hash to validate file(specified by option -f) or 
                            restore original string (specified by option -c)
-a [ --dict ] arg           Dictionary to restore original string using it's MD4 hash
-n [ --min ] arg            The minimal length of the string to restore. 1 by default
-x [ --max ] arg            The maximal length of the string to restore. The length of the dictionary by default
-h [ --search ] <MD4 hash>  MD4 hash to search file that matches it
-c [ --crack ]              Restrore original string using it's MD4 hash that specified by option md4 (m)
-l [ --lower ]              Output MD4 using low case
-r [ --recursively ]        Scan subdirectories
-t [ --time ]               Show calculation time (off by default)
-? [ --help ]               Show help


Examples

Calculate MD4 hash of string 123

md4.exe -s 123


Calculate MD4 hash of a file

md4.exe -f file.txt


Validate file using it's MD4 hash

md4.exe -f file.txt -m 3689CA24BF71B39B6612549D87DCEA68


Calculate MD4 of all files in c:\dir directory

md4.exe -d c:\dir


Calculate MD4 of all files in c:\dir directory including all it's subdirectories

md4.exe -r -d c:\dir


Calculate MD4 of all exe files in c:\dir directory

md4.exe -d c:\dir -i *.exe


Calculate MD4 of all files in c:\dir directory excluding files with tmp extension

md4.exe -d c:\dir -e *.tmp


Calculate MD4 of all exe and dll files in c:\dir directory

md4.exe -d c:\dir -i *.exe;*.dll


Calculate MD4 of all exe files in c:\dir directory excluding files beginning with bad

md4.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known MD4 hash

md4.exe -d c:\ -r -h C58CDA49F00748A3BC0FCFA511D516CB


Restore string by it's MD4 hash using default dictionary

md4.exe -с -m C58CDA49F00748A3BC0FCFA511D516CB


Restore string by it's MD4 hash using user defined dictionary

md4.exe -с -m C58CDA49F00748A3BC0FCFA511D516CB -a 0123456789


Restore string by it's MD4 hash using user defined dictionary and string to restore min and max length

md4.exe -с -m C58CDA49F00748A3BC0FCFA511D516CB -a 0123456789 -n 2 -x 6
