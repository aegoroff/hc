Hash Calculator
==

What is Hash Calculator? In short, it's declarative query language interpreter (or compiler, if you want) to calculate   string and file hashes. Hash Calculator features:

- string hash calculation
- file hash calculation, including only part file hash (defined by file part size and offset from the beginning)
- restoring original string by it's hash specified using brute force method (dictionary search)
- directory's files hash calculation with support of filtering files by size, name, path
- file validation using it's hash
- file searching using file hashes of the whole file or only the part of the file

Also there are:

- Brute force restoring time assumption
- Multithreading brute force restoring
- Different case hash output (by default upper case)
- Output in SFV format (simple file verification)
- Support queries from command line and files
- Support comments in queries' files
- Variables support

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

