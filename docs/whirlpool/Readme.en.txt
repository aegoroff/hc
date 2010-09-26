WHIRLPOOL Calculator Readme 

WHIRLPOOL Calculator is a simple console application that can calculate WHIRLPOOL hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's WHIRLPOOL hash specified using brute force method (dictionary search)
- Searching file using it's WHIRLPOOL hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- Calculating hash of the part of the file setting file part size and offset from the beginning
- File validation using WHIRLPOOL hash specified
- Calculation time of every file (option -t)
- WHIRLPOOL hash different case output (by default upper case)

Usage

whirlpool [OPTION] ...

Available options:

-f [ --file ] <path>                 Full path to file to calculate WHIRLPOOL of
-d [ --dir ] <path>	                 Full path to directory to calculate WHIRLPOOL hash of all it's files
-e [ --exclude ] <mask>              File's mask (pattern) that must be excluded from WHIRLPOOL hash calculating. 
                                     It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>              File's mask (pattern) to process WHIRLPOOL hash calculating (other files will be excluded from process).
                                     It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>             String to calculate WHIRLPOOL hash of
-m [ --whirlpool ] <WHIRLPOOL hash>  WHIRLPOOL hash to validate file(specified by option -f) or 
                                     restore original string (specified by option -c)
-a [ --dict ] arg                    Dictionary to restore original string using it's WHIRLPOOL hash
-n [ --min ] arg                     The minimal length of the string to restore. 1 by default
-x [ --max ] arg                     The maximal length of the string to restore. The length of the dictionary by default
-z [ --limit ] arg                   set the limit in bytes of the part of the file to calculate hash for.
                                     The whole file by default will be applied
-q [ --offset ] arg                  set start position in the file to calculate hash from. Zero by default
-h [ --search ] <WHIRLPOOL hash>     WHIRLPOOL hash to search file that matches it
-o [ --save ] arg                    save files' WHIRLPOOL hashes into the file specified by full path
-c [ --crack ]                       Restrore original string using it's WHIRLPOOL hash that specified by option whirlpool (m)
-l [ --lower ]                       Output WHIRLPOOL using low case
-r [ --recursively ]                 Scan subdirectories
-t [ --time ]                        Show calculation time (off by default)
-? [ --help ]                        Show help


Examples

Calculate WHIRLPOOL hash of string 123

whirlpool.exe -s 123


Calculate WHIRLPOOL hash of a file

whirlpool.exe -f file.txt


Calculate WHIRLPOOL hash of the part of the file (the first kilobyte)

whirlpool.exe -f file.txt -z 1024


Calculate WHIRLPOOL hash of the part of the file (one kilobyte skiping the first 512 bytes)

whirlpool.exe -f file.txt -z 1024 -q 512


Validate file using it's WHIRLPOOL hash

whirlpool.exe -f file.txt -m BD801451D24470DF899173BFF3C04E875BE46C97D1529F84269C70C26C0F7D31D1AD21CBD985E7CFD7E1496B3BA5905789BC0790817DA26DC36D7ECA14B689D7


Calculate WHIRLPOOL of all files in c:\dir directory

whirlpool.exe -d c:\dir


Calculate WHIRLPOOL of all files in c:\dir directory including all it's subdirectories

whirlpool.exe -r -d c:\dir


Calculate WHIRLPOOL of all exe files in c:\dir directory

whirlpool.exe -d c:\dir -i *.exe


Calculate WHIRLPOOL of all files in c:\dir directory excluding files with tmp extension

whirlpool.exe -d c:\dir -e *.tmp


Calculate WHIRLPOOL of all exe and dll files in c:\dir directory

whirlpool.exe -d c:\dir -i *.exe;*.dll


Calculate WHIRLPOOL of all exe files in c:\dir directory excluding files beginning with bad

whirlpool.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known WHIRLPOOL hash

whirlpool.exe -d c:\ -r -h 344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F


Restore string by it's WHIRLPOOL hash using default dictionary

whirlpool.exe -с -m 344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F


Restore string by it's WHIRLPOOL hash using user defined dictionary

whirlpool.exe -с -m 344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F -a 0123456789


Restore string by it's WHIRLPOOL hash using user defined dictionary and string to restore min and max length

whirlpool.exe -с -m 344907E89B981CAF221D05F597EB57A6AF408F15F4DD7895BBD1B96A2938EC24A7DCF23ACB94ECE0B6D7B0640358BC56BDB448194B9305311AFF038A834A079F -a 0123456789 -n 2 -x 6
