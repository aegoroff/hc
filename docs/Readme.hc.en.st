$langName$ Readme 

$langName$ is the console tool that can calculate about 50 cryptographic hashes of strings and files. $langName$ features:

- string hash calculation
- file hash calculation, including only part file hash (defined by file part size and offset from the beginning)
- restoring original string by it's hash specified using brute force method (dictionary search)
- directory's files hash calculation with support of filtering files by size, name, path
- file validation using it's hash
- file searching using file hashes of the whole file or only the part of the file

Also there are:

- Brute force restoring time assumption
- Multithreading brute force restoring
- Nvidia GPU brute force for some hashes. Only CRC32, MD-2, MD-5, Whirlpool, Ripemd 160, SHA-1 and SHA-2 Family supported so far.
- Different case hash output (by default upper case)
- Output in SFV format (simple file verification)

The app supports the following hashes:

- MD2
- MD4
- MD5
- SHA1
- SHA224
- SHA256
- SHA384
- SHA512
- SHA3 224 (FIPS 202)
- SHA3 256 (FIPS 202)
- SHA3 384 (FIPS 202)
- SHA3 512 (FIPS 202)
- SHA3 224 (Keccak)
- SHA3 256 (Keccak)
- SHA3 384 (Keccak)
- SHA3 512 (Keccak)
- Whirlpool
- Ripemd 128
- Ripemd 160
- Ripemd 256
- Ripemd 320
- Tiger-192
- Tiger2-192
- CRC32
- GOST 34.11-94
- Snerfu 128
- Snerfu 256
- TTH (Tiger Tree Hash)
- HAVAL 128, 3
- HAVAL 128, 4
- HAVAL 128, 5
- HAVAL 160, 3
- HAVAL 160, 4
- HAVAL 160, 5
- HAVAL 192, 3
- HAVAL 192, 4
- HAVAL 192, 5
- HAVAL 224, 3
- HAVAL 224, 4
- HAVAL 224, 5
- HAVAL 256, 3
- HAVAL 256, 4
- HAVAL 256, 5
- EDON-R 256
- EDON-R 512
- NTLM
- BLAKE 2b
- BLAKE 2s

Usage

This tool supports many different hash algorithms so, to setup necessary hash algorithm we use the first command line option.
See full list below. After that we need to define desired work mode (string, file, dir, hash) that supposed to calculate   
string hash (string), file hash (file), all files hash in selected directory (dir) and to restore string using it's hash (hash).
So the next option will be selected working mode, for example:

$appName$.exe sha1 dir ...

After these two common options, you should use specific for each working mode options. Here is their full syntax:

String mode

$appName$ [-lh] <algorithm> string -s <string>

  <algorithm>               hash algorithm. See all possible values below
  -s, --source=<string>     string to calculate hash sum for
  -b, --base64              output hash as Base64
  -l, --lower               output hash using low case (false by default)
  -h, --help                print this help and exit

  
Hash or string restore mode

$appName$ [-bplh] <algorithm> hash [-m <string>] [-a <string>] [-n <int>] [-x <int>] [--noprobe] [-T <int>]

  <algorithm>               hash algorithm. See all possible values below
  -s, --source=<string>     hash to restore initial string by
  -b, --base64              interpret hash as Base64
  -a, --dict=<string>       initial string's dictionary. All digits, upper and 
                            lower case latin symbols by default
  -n, --min=<int>           set minimum length of the string to restore. 1 by default
  -x, --max=<int>           set maximum length of the string to restore. 10 by default
  -p, --performance         test performance by cracking 12345 string hash
      --noprobe             Disable hash crack time probing (how much time it may take)
  -T, --threads=<int>       the number of threads to crack hash. The half of system processors by default. The value must be between 1 and processor count.
  -l, --lower               output hash using low case (false by default)
  -h, --help                print this help and exit


File mode

$appName$ [-ctlh] <algorithm> <command> -f file [-m <string>] [-z <number>] [-q <number>] [-o <file>] [--sfv]

  <algorithm>               hash algorithm. See all possible values below
  -s, --source=<file>       full path to file to calculate hash sum of
  -m, --hash=<string>       hash to validate file
  -z, --limit=<number>      set the limit in bytes of the part of the file to calculate hash for. The whole file by default will be applied
  -q, --offset=<number>     set start position within file to calculate hash from. Zero by default
  -c, --checksumfile        output hash in file checksum format
  -o, --save=<file>         save files' hashes into the file specified instead of console.
  -t, --time                show calculation time (false by default)
      --sfv                 output hash in the SFV (Simple File Verification)  
                            format (false by default). Only for CRC32.
  -l, --lower               output hash using low case (false by default)
  -b, --base64              output hash as Base64
  -h, --help                print this help and exit


Dir mode

$appName$ [-rctlh] <algorithm> dir -s <string> [-m <string>] [-e <string>] [-i <string>] [-z <number>] [-q <number>] [-H <string>] [-o <file>] [--sfv] [--noerroronfind]

  <algorithm>               hash algorithm. See all possible values below
  -s, --source=<string>     full path to dir to calculate all content's hashes
  -m, --hash=<string>       hash to validate files in directory
  -e, --exclude=<string>    exclude files that match the pattern specified. It's possible to use several patterns separated by ;
  -i, --include=<string>    include only files that match the pattern specified. It's possible to use several patterns separated by ;
  -z, --limit=<number>      set the limit in bytes of the part of the file to calculate hash for. The whole file by default will be applied
  -q, --offset=<number>     set start position within file to calculate hash from. Zero by default
  -H, --search=<string>     hash to search a file that matches it
  -r, --recursively         scan directory recursively
  -c, --checksumfile        output hash in file checksum format
  -o, --save=<file>         save files' hashes into the file specified instead of console.
  -t, --time                show calculation time (false by default)
      --sfv                 output hash in the SFV (Simple File Verification)  
                            format (false by default). Only for CRC32.
  -l, --lower               output hash using low case (false by default)
  -b, --base64              output hash as Base64
      --noerroronfind       Disable error output while search files. False by default.
  -h, --help                print this help and exit



  Supported hash algorithms definitions:
    blake2b
    blake2s
    crc32
    edonr256
    edonr512
    gost
    haval-128-3
    haval-128-4
    haval-128-5
    haval-160-3
    haval-160-4
    haval-160-5
    haval-192-3
    haval-192-4
    haval-192-5
    haval-224-3
    haval-224-4
    haval-224-5
    haval-256-3
    haval-256-4
    haval-256-5
    md2
    md4
    md5
    ntlm
    ripemd128
    ripemd160
    ripemd256
    ripemd320
    sha-3-224
    sha-3-256
    sha-3-384
    sha-3-512
    sha-3k-224
    sha-3k-256
    sha-3k-384
    sha-3k-512
    sha1
    sha224
    sha256
    sha384
    sha512
    snefru128
    snefru256
    tiger
    tiger2
    tth
    whirlpool
  

Examples

Calculate SHA1 hash of string 123

$appName$.exe sha1 string -s 123


Calculate SHA1 hash of a file

$appName$.exe sha1 file -s file.txt


Calculate SHA1 hash of a file and output result in base64 form

$appName$.exe sha1 file -s file.txt -b


Calculate SHA1 hash of the part of the file (the first kilobyte)

$appName$.exe sha1 file -s file.txt -z 1024


Calculate SHA1 hash of the part of the file (one kilobyte skiping the first 512 bytes)

$appName$.exe sha1 file -s file.txt -z 1024 -q 512


Validate file using it's SHA1 hash

$appName$.exe sha1 file -s file.txt -m 8BD7954C40C1E59A900F71EA3A266732609915B1


Calculate SHA1 of all files in c:\dir directory

$appName$.exe sha1 dir -s c:\dir


Calculate SHA1 of all files in c:\dir directory including all it's subdirectories

$appName$.exe sha1 dir -r -s c:\dir


Calculate SHA1 of all files in c:\dir directory and output results in checksum file format

$appName$.exe sha1 dir -s c:\dir --checksumfile


Calculate SHA1 of all files in c:\dir directory and save results to dir.sha1 in checksum file format

$appName$.exe sha1 dir -s c:\dir --checksumfile -o dir.sha1


Calculate CRC32 of all files in c:\dir directory and save results to dir.sfv in SFV format

$appName$.exe crc32 dir -s c:\dir --sfv -o dir.sfv


Calculate SHA1 of all exe files in c:\dir directory

$appName$.exe sha1 dir -s c:\dir -i *.exe


Calculate SHA1 of all files in c:\dir directory excluding files with tmp extension

$appName$.exe sha1 dir -s c:\dir -e *.tmp


Calculate SHA1 of all exe and dll files in c:\dir directory

$appName$.exe sha1 dir -s c:\dir -i *.exe;*.dll


Calculate SHA1 of all exe files in c:\dir directory excluding files beginning with bad

$appName$.exe sha1 dir -s c:\dir -i *.exe -e bad*


Searching file on C:\ drive using known SHA1 hash

$appName$.exe sha1 dir -s c:\ -r -H 8BD7954C40C1E59A900F71EA3A266732609915B1


Restore string by it's SHA1 hash using default dictionary

$appName$.exe sha1 hash -s 40BD001563085FC35165329EA1FF5C5ECBDBBEEF


Restore string by it's SHA1 hash using user defined dictionary

$appName$.exe sha1 hash -s 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789


Restore string by it's SHA1 hash using user defined dictionary short form (only digits)

$appName$.exe sha1 hash -s 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9


Restore string by it's SHA1 hash using user defined dictionary short form (low case latin letters)

$appName$.exe sha1 hash -s 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a a-z


Restore string by it's SHA1 hash using user defined dictionary short form (digits, upper and low case latin letters)

$appName$.exe sha1 hash -s 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0-9a-zA-Z


Restore string by it's SHA1 hash using user defined dictionary and string to restore min and max length

$appName$.exe sha1 hash -s 40BD001563085FC35165329EA1FF5C5ECBDBBEEF -a 0123456789 -n 2 -x 6


Test performance SHA1 using default dictionary

$appName$.exe sha1 hash -p


Test performance using user defined dictionary short form (only digits)

$appName$.exe sha1 hash -p -a 0-9