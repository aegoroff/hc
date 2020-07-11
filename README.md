Hash Calculator
======
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/0d8224be8eac4324b81f8846c6b14161)](https://app.codacy.com/manual/egoroff/hc?utm_source=github.com&utm_medium=referral&utm_content=aegoroff/hc&utm_campaign=Badge_Grade_Dashboard)
[![Build status](https://ci.appveyor.com/api/projects/status/cn563po680fcgqa4?svg=true)](https://ci.appveyor.com/project/aegoroff/hc)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Faegoroff%2Fhc.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Faegoroff%2Fhc?ref=badge_shield)

Hash Calculator is the console tool that can calculate about 50 cryptographic hashes of strings and files. Hash Calculator main features are:

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


## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Faegoroff%2Fhc.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Faegoroff%2Fhc?ref=badge_large)