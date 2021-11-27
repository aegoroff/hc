Hash Calculator
======
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/0d8224be8eac4324b81f8846c6b14161)](https://app.codacy.com/manual/egoroff/hc?utm_source=github.com&utm_medium=referral&utm_content=aegoroff/hc&utm_campaign=Badge_Grade_Dashboard)
[![CI](https://github.com/aegoroff/hc/actions/workflows/ci.yml/badge.svg)](https://github.com/aegoroff/hc/actions/workflows/ci.yml)
[![](https://tokei.rs/b1/github/aegoroff/hc)](https://github.com/aegoroff/hc)

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

## Install the pre-compiled binary

**scoop**:

```sh
scoop bucket add aegoroff https://github.com/aegoroff/scoop-bucket.git
scoop install hc
```

**manually**:

Download the pre-compiled binaries from the [releases](https://github.com/aegoroff/hc/releases) and
copy to the desired location.