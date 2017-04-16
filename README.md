LINQ to Hash
======
[![Build status](https://ci.appveyor.com/api/projects/status/cn563po680fcgqa4?svg=true)](https://ci.appveyor.com/project/aegoroff/hc)

LINQ to hash is the console tool that can calculate about 50 cryptographic hashes of strings and files. This tool name is not suitable for the app at the moment
because old query language support has been removed from it. It has been done by several reasons, but the new full LINQ like syntax will be available in the next major version
but it will be in the separate tool. LINQ to hash features

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
- Variables support
