Hash Calculator
======
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bd007c6fe23b46ef923613ac3e81fd5d)](https://app.codacy.com/gh/aegoroff/hc/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![CI](https://github.com/aegoroff/hc/actions/workflows/ci.yml/badge.svg)](https://github.com/aegoroff/hc/actions/workflows/ci.yml)
[![](https://tokei.rs/b1/github/aegoroff/hc?category=code)](https://github.com/XAMPPRocky/tokei)

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
- GPU brute force (CUDA and OpenCL) for CRC32, MD2, MD4, MD5, NTLM, Whirlpool, Ripemd 128/160/256/320, SHA1, SHA-2 family, SHA3 (FIPS 202 and Keccak), Tiger-192, Tiger2-192, BLAKE 2b, BLAKE 2s and BLAKE 3
- Different case hash output (by default upper case)
- Output in SFV format (simple file verification)
- Variables support
- **l2h** — LINQ-style hash query language (`src/l2h/`). Semantics: [docs/l2h-semantics.md](docs/l2h-semantics.md)

## Install the pre-compiled binary

**homebrew** (macOS and Linux):

```sh
brew tap aegoroff/tap
brew install aegoroff/tap/hc
```

**scoop**:

```sh
scoop bucket add aegoroff https://github.com/aegoroff/scoop-bucket.git
scoop install hc
```

**AUR (Arch Linux User Repository)**:

install binary package:
```sh
 yay -S hash-calculator-bin
```
or if yay reports that package not found force updating repo info
```sh
yay -Syyu hash-calculator-bin
```

**deb (Debian / Ubuntu)**:

Download the `.deb` for your architecture (`amd64` or `arm64`) from the
[releases](https://github.com/aegoroff/hc/releases) page, then:

```sh
sudo apt install ./hash-calculator_*_amd64.deb
# or: sudo apt install ./hash-calculator_*_arm64.deb
```

**rpm (Fedora / RHEL / Alma / Rocky)**:

Download the `.rpm` for your architecture (`x86_64` or `aarch64`) from the
[releases](https://github.com/aegoroff/hc/releases) page, then:

```sh
sudo dnf install ./hash-calculator-*-1.x86_64.rpm
# or: sudo dnf install ./hash-calculator-*-1.aarch64.rpm
# openSUSE: sudo zypper install ./hash-calculator-*-1.x86_64.rpm
```

**apk (Alpine Linux)**:

Download the `.apk` for your architecture (`x86_64` or `aarch64`) from the
[releases](https://github.com/aegoroff/hc/releases) page, then:

```sh
sudo apk add --allow-untrusted ./hash-calculator-*-r0.x86_64.apk
# or: sudo apk add --allow-untrusted ./hash-calculator-*-r0.aarch64.apk
```

**apk (OpenWrt aarch64_cortex-a53)**:

Same musl binaries, packaged with OpenWrt's arch name. From the
[releases](https://github.com/aegoroff/hc/releases) page:

```sh
apk add --allow-untrusted ./hash-calculator-*-r0.aarch64_cortex-a53.apk
```

**manually**:

Download the pre-compiled binaries from the [releases](https://github.com/aegoroff/hc/releases) and
copy to the desired location.
