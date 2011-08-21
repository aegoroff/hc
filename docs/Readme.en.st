$hashName$ Calculator Readme 

$hashName$ Calculator is a simple console application that can calculate $hashName$ hash of:
- A string specified (passed through command line)
- A file
- All files in a directory specified, including subdirectories (option -r)

Also there are:

- Restoring original string by it's $hashName$ hash specified using brute force method (dictionary search)
- Brute force restoring time calculation
- Searching file using it's $hashName$ hash known

Supported:

- Excluding directory files by mask (option -e)
- Including directory files by mask (option -i)
- Subdirectory search (option -r)
- Files which size is greater then 4 Gb
- Calculating hash of the part of the file setting file part size and offset from the beginning
- File validation using $hashName$ hash specified
- Calculation time of every file (option -t)
- $hashName$ hash different case output (by default upper case)

Usage

$appName$ [OPTION] ...

Available options:

-f [ --file ] <path>                 Full path to file to calculate $hashName$ of
-d [ --dir ] <path>	                 Full path to directory to calculate $hashName$ hash of all it's files
-e [ --exclude ] <mask>              File's mask (pattern) that must be excluded from $hashName$ hash calculating. 
                                     It's possible to use several masks separated by ";". Available only with option -d (--dir)
-i [ --include ] <mask>              File's mask (pattern) to process $hashName$ hash calculating (other files will be excluded from process).
                                     It's possible to use several masks separated by ";". Available only with option -d (--dir)
-s [ --string ] <string>             String to calculate $hashName$ hash of
-m [ --$mOpt$ ] <$hashName$ hash>$spaceCountMOpt$$hashName$ hash to validate file(specified by option -f) or 
                                     restore original string (specified by option -c)
-a [ --dict ] arg                    Dictionary to restore original string using it's $hashName$ hash
-n [ --min ] arg                     The minimal length of the string to restore. 1 by default
-x [ --max ] arg                     The maximal length of the string to restore. 10 by default
-z [ --limit ] arg                   set the limit in bytes of the part of the file to calculate hash for.
                                     The whole file by default will be applied
-q [ --offset ] arg                  set start position in the file to calculate hash from. Zero by default
-h [ --search ] <$hashName$ hash>$spaceCountSearchOpt$$hashName$ hash to search file that matches it
-o [ --save ] arg                    save files' $hashName$ hashes into the file specified by full path
-c [ --crack ]                       Restrore original string using it's $hashName$ hash that specified by option $appName$ (m)
-l [ --lower ]                       Output $hashName$ using low case
-r [ --recursively ]                 Scan subdirectories
-t [ --time ]                        Show calculation time (off by default)
-? [ --help ]                        Show help


Examples

Calculate $hashName$ hash of string 123

$appName$.exe -s 123


Calculate $hashName$ hash of a file

$appName$.exe -f file.txt


Calculate $hashName$ hash of the part of the file (the first kilobyte)

$appName$.exe -f file.txt -z 1024


Calculate $hashName$ hash of the part of the file (one kilobyte skiping the first 512 bytes)

$appName$.exe -f file.txt -z 1024 -q 512


Validate file using it's $hashName$ hash

$appName$.exe -f file.txt -m $hashOfFile$


Calculate $hashName$ of all files in c:\dir directory

$appName$.exe -d c:\dir


Calculate $hashName$ of all files in c:\dir directory including all it's subdirectories

$appName$.exe -r -d c:\dir


Calculate $hashName$ of all exe files in c:\dir directory

$appName$.exe -d c:\dir -i *.exe


Calculate $hashName$ of all files in c:\dir directory excluding files with tmp extension

$appName$.exe -d c:\dir -e *.tmp


Calculate $hashName$ of all exe and dll files in c:\dir directory

$appName$.exe -d c:\dir -i *.exe;*.dll


Calculate $hashName$ of all exe files in c:\dir directory excluding files beginning with bad

$appName$.exe -d c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known $hashName$ hash

$appName$.exe -d c:\ -r -h $hashOfFile$


Restore string by it's $hashName$ hash using default dictionary

$appName$.exe -с -m $hashOf123$


Restore string by it's $hashName$ hash using user defined dictionary

$appName$.exe -с -m $hashOf123$ -a 0123456789


Restore string by it's $hashName$ hash using user defined dictionary and string to restore min and max length

$appName$.exe -с -m $hashOf123$ -a 0123456789 -n 2 -x 6
