# l2h Query Language: Semantics

This document is the source of truth for how LINQ-style hash queries behave. If it says something and the query-plan interpreter does something else, that's a bug, unless a section explicitly flags it as a known limitation.

> **How to read this document**
> §1 sets up the mental model and says what's deliberately out of scope. §2 is a short, runnable tour. §3 through §8 are the reference proper (values, properties, queries, clauses, output, errors). §9 walks through how it's actually implemented. §10 is a changelog of settled decisions, kept around so old arguments don't get relitigated.
> Where you see a table or a formal rule, that's the exact behavior. The prose around it explains why it's that way and how you'd actually use it.

---

## 1. Purpose & mental model

l2h is a small query language built for one job: hashing work. Every query expression does four things, in order:

1. Binds **range variables** to data sources (`string`, `file`, `dir`, `hash`). A *range variable* is just the LINQ term for the per-element loop variable (`f` in `from file f in …`), bound to each source element in turn. The name comes from relational calculus, where the variable *ranges over* a relation; it has nothing to do with numeric intervals, despite what the word suggests.
2. Pipes a **sequence of environments** through LINQ-style clauses (`where`, `let`, `join`, `orderby`, `group`, …).
3. Reads **computed properties** (file size, digests, and so on), but only when the query actually asks for them. Nothing is computed speculatively.
4. Either **prints** the final result, or hands it off to the next stage via **`into`**.

A query is a pipeline: each clause takes whatever sequence it's handed and produces a new one. Most stages can stream environments one at a time. Only `orderby`, `group by`, and nested queries that build a `Seq` collect first. Either way, **property and hash I/O** stays demand-driven: a filesystem read or digest runs only when a property forces it, or when the terminal step prints (including Hash restore, §4.4).

```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```

Read it left to right: open one file, keep it only if it's non-empty, then print its MD5. `size` is a cheap `stat` call; `md5` means a full read plus a hash. The file body only gets hashed once it's already passed the filter, so you never pay for the hash of a file you were going to discard anyway.

> **Cost note.** Hashing dominates runtime; interpreter overhead basically doesn't matter next to it. So the interpreter is built to favor clarity and testability over speed. Micro-optimizing it is explicitly not a goal (see §1.1).

### 1.1 Boundaries

A few things are deliberately **not** part of l2h:

- **Method calls outside the catalogs.** Record formatters (§4.7), hash-check methods on `File`/`String` (§4.8), `Dir.tree()` / `Dir.skipErrors()` (§4.6), `File.offset(n)` / `File.limit(n)` (§4.5), and `Seq.count()` (§4.9) are the only method calls that exist. Property access (`x.prop`) covers digests and metadata; anything else (other receivers, other method names) is an error.
- **A bytecode / register VM.** The runtime uses Volcano-lite pull operators over the compiled plan tree. There's no global instruction tape and nothing coupled to an instruction index.
- **Interpreter performance tuning beyond correctness.** Not a goal here, by design.

---

## 2. A quick tour

A handful of real queries, referenced throughout the rest of the document.

**Hash one file, keep only non-empty ones:**
```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```
`size` is a `stat` call (cheap); `md5` forces a read plus a hash. The file body is hashed only for files that pass `where`.

**Find a file by a known digest, then read its size:**
```text
from file f in '/home/user/file'
where f.md5 == 'd41d8cd98f00b204e9800998ecf8427e'
select f.size;
```
Here `md5` gets forced inside `where`, and `size` stays cheap afterward. Digest comparison is case-insensitive (§5.3), so you don't need to match case in the literal.

**List large files under a directory:**
```text
from dir d in '/tmp'
from file f in d
where f.size > 1000
select f.sha1;
```
`from dir` binds a directory; the *second* `from` then walks its immediate regular files. Subdirectories and symlinks get skipped automatically (§3.4).

**Same idea, but recursing into subdirectories:**
```text
from dir d in '/tmp'
from file f in d.tree()
where f.size > 1000
select f.sha1;
```
`d.tree()` returns a new `Dir` that walks without a depth limit; feed that into `from file` and you get the whole tree (§4.6). `d.tree(n)` limits descent to `n` levels (`tree(0)` is the same as flat). The original `d` is unchanged: `from file f in d` still means flat.

**Join two sources on equal digests:**
```text
from string a in 'abc'
join string b in 'abc' on a.md5 equals b.md5
select { digest = a.md5, len = b.size };
```
This is a plain inner join: only the row pairs whose digests actually match (case-insensitively) survive. The anonymous object uses explicit field names (auto-names from `a.md5` and `b.md5` would both be `md5` and collide; see §5.4). Selecting two fields prints two lines per joined row (§7).

**Restore / reverse a known digest:**
```text
from hash x in 'D41D8CD98F00B204E9800998ECF8427E'
select x.md5;
```
`from hash` does **not** hash the literal characters you gave it. Instead it treats the digest as input to a restore lookup for `md5` (§4.4). That's the inverse of hashing, not the forward path.

**Carry a result into a second query with `into`:**
```text
from file f in 'x'
select f.md5 into h
select h;
```
The first `select` doesn't print anything; it hands its result off to `h`. The second `select` is the one that's actually terminal, and it's the one that prints the resulting sequence of digest strings.

**Hash only a byte window of a file (`limit` / `offset`):**
```text
from file f in '/home/user/file'
select f.offset(2).limit(4).md5;
```
Same idea as `hc`'s `--offset` / `--limit`. `f.offset(2).limit(4)` returns a new `File` for that byte range; the original `f` stays whole-file. Hashing (or a hash-check) on the result uses the window (§4.5).

---

## 3. Values & sources

This section covers what actually flows through a query: the kinds of values you can have, how names get tracked (the *environment*), and how each `from` form seeds the pipeline.

### 3.1 Value kinds

Every value at runtime is one of these:

| Kind | Meaning |
|------|---------|
| `String` | Text payload (also used for digests and paths expressed as strings) |
| `File` | A regular file identified by its path |
| `Dir` | A directory identified by path: the source of enumeration, not a row by itself |
| `Hash` | A restore source: a digest string; the algorithm gets chosen via a property or `select` |
| `Int` | Signed integer (sizes, numeric literals) |
| `Bool` | Predicate results, plus the literals `true` and `false` |
| `Record` | An anonymous object / product of `let`, `join`, or `{…}` shaping |
| `Seq(T)` | An eager sequence (materialized bag) of values |

### 3.2 Environment

An **environment** (`Env`) is just a finite map from range-variable names to `Value`. Each clause takes a sequence of environments in and produces a new one, with one exception: after `select`/`group`, you're instead working with a sequence of projected values.

Wherever a rule writes "`Env ∪ { id ↦ value }`", read it as: bind `id` to that value, shadowing whatever was previously bound to that name.

### 3.3 Source sequences (`from`)

The opening `from`, and any later `from` further down in the body, is where data actually enters the pipeline. Each form has its own way of seeding the starting sequence:

| Declaration | Produced sequence |
|-------------|-------------------|
| `from string x in E` | Singleton: one `String` from evaluating `E` (must be a `String` value; no path/digest coercion from `File`/`Dir`/`Hash`) |
| `from file x in E` | Singleton: one `File` for path `E` when `E` is a string path (error if missing or not a regular file); or a Dir walk when `E` is a `Dir` (§3.4) |
| `from dir x in E` | Singleton: one `Dir` for path `E` when `E` is a string path (error if missing or not a directory) |
| `from hash x in E` | Singleton: one `Hash` whose digest comes from `E` (must be a `String` digest payload; no coercion from `File`/`Dir` paths) |

**Directory contents are not implicitly flattened into the range variable of `from dir`.** `from dir` gives you the directory *itself*. If you want the files inside it, you need an explicit second `from`:

```text
from dir d in '/tmp'
from file f in d
where f.size > 0
select f.md5;
```

Here, `from file f in d` means: for the current `Dir` bound to `d`, emit one environment per **immediate child regular file** (see §3.4), binding `f` to that `File` (`d` stays in scope unless it gets shadowed).

**Type rule for that form:** whatever comes after `in` has to evaluate to a **`Dir`**. A bare path string won't do here: you need `from dir d in '/tmp'` first, and only then `from file f in d`. A runtime kind mismatch is an error.

Any additional `from` in the body works like a **SelectMany**: for each outer row, it evaluates the inner source and concatenates the extended environments. Nested `from`s flatten naturally, the way you'd expect.

### 3.4 Directory enumeration

When `from file f in <Dir>` walks a directory, a few rules apply:

- **Flat by default.** Only the files sitting directly in that folder get visited. If you want more, pass `d.tree()` (unlimited) or `d.tree(n)` (depth-limited) instead of `d` (§4.6).
- **Regular files only.** Symlinks are always skipped, whether they point at a file or a directory. Flat mode also skips subdirectories entirely; recursive modes descend into real directories but still ignore symlink entries (they never follow them).
- **No magic recursive `from dir`.** Recursion is a depth limit on the `Dir` value, set by the `tree` method, not a separate source form of its own.

File order is whatever the directory walk returns; the language does not promise lexicographic order. Use `orderby` when you need a fixed order (for example `orderby f.path`).

---

## 4. Computed properties

Properties (`size`, `md5`, `path`, …) are what connect a value to the work you actually care about. **They're computed on demand.** That's what makes filter-before-hash possible.

### 4.1 Demand-driven evaluation (key idea)

A property gets computed on **first read**, during expression evaluation, wherever that happens to be: `where`, `let`, join keys, `orderby`, `group by`, `select`, anywhere. An implementation **may** cache the result on the value for the rest of the query, so you don't pay twice.

There's no separate "hashing phase" that runs before or after filtering. Whether a hash runs before or after a filter is entirely a consequence of **which properties the query happens to force**:

```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```
`size` gets read in `where` (stat only); `md5` gets read in the terminal `select`.

```text
from file f in '/home/user/file'
where f.md5 == 'D41D8CD98F00B204E9800998ECF8427E'
select f.size;
```
Now `md5` is forced in `where` (file read + hash), while `size` stays cheap afterward.

In practice: put cheap predicates (`size`, `path`) before expensive ones (`<hash>`) in `where`, so you're not hashing rows you're about to throw away.

### 4.2 Access syntax

The syntax is `range.prop` for property access. **Method calls** use `receiver.method(args…)`, where the receiver can be either a range identifier or a record literal `{…}` (record literals only work for formatters; see §4.7). That covers Record formatters (§4.7), hash-check on `File`/`String` (§4.8), `Dir.tree()` / `Dir.skipErrors()` (§4.6), `File.offset(n)` / `File.limit(n)` (§4.5), and `Seq.count()` (§4.9). Unknown methods, wrong arity, or an invalid receiver are all errors.

### 4.3 Property catalog

Which properties are available depends entirely on the **runtime kind** of the receiver. Asking for a property a given kind doesn't have is an error, and ideally that gets caught statically, since the range type is usually already known at compile time.

| Receiver | Property | Result | Notes |
|----------|----------|--------|-------|
| `File` | `path` | `String` | Path identifying the file (no I/O; just projects the bound path) |
| `File` | `name` | `String` | Basename only (no directory), same extraction as `hc`'s SFV filename; no I/O |
| `File` | `size` | `Int` | File size in bytes (full file; unaffected by `limit`/`offset`) |
| `File` | `offset` | `Int` | Start byte for hashing (default `0`). Read via the property; set with `offset(n)` (§4.5) |
| `File` | `limit` | `Int` | Max bytes to hash from `offset` (default: whole file). Read via the property; set with `limit(n)` (§4.5) |
| `File` | `readable` | `Bool` | `true` if the path opens as a regular file (probe open+stat); `false` on permission/missing/non-file; never raises I/O. Use `where f.readable` before `size` / `<hash>` |
| `File` | `<hash>` | `String` | Hex digest of file contents (honoring that value's window); `<hash>` can be any algorithm name `hc` knows (`md5`, `sha1`, `tiger`, …) |
| `String` | `size` | `Int` | Length in bytes (UTF-8 payload length as stored) |
| `String` | `<hash>` | `String` | Hex digest of the string's bytes |
| `Hash` | `<hash>` | `String` | **Restore** path: treats the bound digest as the input digest for algorithm `<hash>` (same meaning as legacy `from hash … select x.md5`). This is *not* "hash the digest characters as a string" |
| `Dir` | `path` | `String` | Path identifying the directory (no I/O; projects the bound path). Use `from file f in d` (or `d.tree()` / `d.tree(n)`) to reach the files inside |
| `Record` | field name | field value | Fields introduced by `{…}`, `let`, or join shaping |
| `Int` / `Bool` | - | - | No properties in v1.0 |
| `Seq` | - | - | No properties; use `count()` (§4.9) |

Hex digests from **computed** hash properties (`File` / `String`) are always **lowercase** when printed or produced as a `String` value. A `Hash` restore property returns the **bound digest as stored** (input casing preserved); the restore runner's own output is separate (§4.4). Equality, join keys, and `orderby` still use **case-insensitive** comparison whenever either operand is a digest value (§5.3).

### 4.4 `from hash` + select (restore)

```text
from hash x in 'D41D8CD98F00B204E9800998ECF8427E'
select x.md5;
```

This restores / reverses using algorithm `md5` against the given digest literal. The actual work gets delegated to the existing hash-restore runners in `modes`. Again, it does **not** mean "compute md5 of the hex string"; that would be a completely different (and much less useful) operation.

**Stdout contract.** Evaluating a Hash `<hash>` property may write restore runner output to stdout as a side effect. The property still returns the bound digest string (input casing preserved). When that property is the **terminal** `select` projection on a bare Hash range variable (e.g. `select x.md5`), the sink **does not** print the returned string again; otherwise you'd get the restore output plus a duplicate digest line.

### 4.5 File hash window (`offset(n)` / `limit(n)`)

`offset` and `limit` exist only on `File`. They match `hc`'s file options: `offset` is the start byte, `limit` is how many bytes to feed the hasher from there. Leave `limit` at its default and hashing runs through EOF. A non-positive `limit` means "no limit," same as `hc`.

You set the window with methods that return a **new** `File` (same path, one field updated), the same pattern as `Dir.tree()`:

| Call | Effect |
|------|--------|
| `f.offset(n)` | New `File` with start byte `n` (`n ≥ 0`); other fields copied |
| `f.limit(n)` | New `File` with max-bytes `n` (`n ≥ 0`); other fields copied |

```text
from file f in '/home/user/file'
select f.offset(2).limit(4).md5;
```

```text
from file f in '/home/user/file'
let w = f.offset(2).limit(4)
where w.size > 0
select w.md5;
```

The original `f` is not mutated. Order of the two calls does not matter: `f.offset(2).limit(4)` and `f.limit(4).offset(2)` are the same. Any hash property or hash-check on a `File` uses that value's window fields.

Fresh files from `from file` start at `offset = 0` and `limit = maxInt(i64)` (hc's whole-file sentinel). A negative argument is a runtime error (`InvalidWindow`). An `offset` past EOF (`offset > 0` and `offset >=` file size, including empty files with `offset ≥ 1`) fails the same way `hc` does ("offset too big"), not as a generic missing-path I/O error. `offset(0)` on an empty file is valid (hashes the empty payload).

Bare `f.offset` / `f.limit` (no parentheses) just read the current integers on that value. `select f.limit` on a fresh file yields `maxInt(i64)`, not an "EOF" token. And `where f.offset == 2` is a normal comparison against those integers; it does not set the window.

Calling `offset(n)` / `limit(n)` on a non-`File` is an invalid method receiver. Property access to `limit` / `offset` on other kinds is still an invalid property (§4.3). There is no parentheses-less method form: bare `f.offset` is the Int property.

### 4.6 Directory recursion (`Dir.tree()` / `Dir.tree(n)`)

`tree` is a **`Dir`-only** method. It returns a **new** `Dir` with the same path and a depth limit for enumeration:

| Call | Depth | Effect |
|------|-------|--------|
| `d.tree()` | unlimited | Walk the whole tree under `d` |
| `d.tree(n)` | `n` (`Int`, `n ≥ 0`) | Enter at most `n` directory levels below `d` (files in entered dirs are yielded; files are not filtered by their own depth) |
| `d.tree(0)` | `0` | Same as flat `from file f in d`: only the current directory |
| `d.skipErrors()` | (unchanged) | Soft walk: skip walk/`enter` failures for subdirectories and continue |

Depth counts how many directory levels below `d` you may enter: `tree(1)` yields files in `d` plus files in immediate subdirectories, but not deeper. Leave the bare `d` alone and `from file f in d` only sees files sitting in that folder; pass a `tree` result and the same `from` walks according to the limit:

```text
from dir d in '/tmp'
from file f in d.tree()
select f.path;
```

```text
from dir d in '/tmp'
from file f in d.tree(1)
select f.path;
```

The original `d` is not mutated. You can still use `from file f in d` for a flat listing in the same query. Symlinks stay skipped even while recursing, same as flat listing, and same as `hc -r`.

**Unreadable subdirectories.** By default, failing to enter a subdirectory during a recursive walk is an **I/O error** (the query stops; the message includes the directory path). `skipErrors()` returns a **new** `Dir` that soft-skips both **enter** failures and other walk-iteration errors for subdirectories, and continues with siblings:

```text
from dir d in '/tmp'
from file f in d.tree().skipErrors()
select f.path;
```

`tree` / `tree(n)` and `skipErrors()` compose in either order; each copies the other's flags onto the new `Dir`. Calling `tree` / `skipErrors` on a non-`Dir`, wrong arity/types, or a negative tree depth is an error. There is no `tree` / `skipErrors` property; bare `d.tree` / `d.skipErrors` without `()` is an invalid property.

### 4.7 Record methods (formatters)

Methods on a **`Record`** are formatters. A call evaluates to a **`String`**, after which the usual sink / `into` / `let` rules apply as normal.

The receiver can be a bound identifier (`let` / `into`), or you can just call the formatter straight on a record literal:

```text
from file f in '/tmp/a'
select { f.crc32, f.name }.sfv();   -- order in the object does not matter
# → a    <crc>          (always name, then digest)
```

```text
from file f in '/tmp/a'
let o = { f.path, f.crc32 }
select o.checksum();
# → <crc> /tmp/a     (always digest, then path; one space, GNU *sum -c compatible)
```

`sfv` and `checksum` look fields up **by name** and always emit a **fixed** layout. The declaration order inside `{…}` doesn't matter at all:

| Method | Args | Required fields | Output |
|--------|------|-----------------|--------|
| `sfv()` | none | exactly 2 fields including **`name`**; the other is the digest | `name    digest` (like `hc --sfv`) |
| `checksum()` | none | exactly 2 fields including **`path`**; the other is the digest | `digest path` (like `hc -c`) |
| `json()` | none | scalars, nested `Record`, `Seq` (any depth) | Compact JSON object (`std.json` minified). Nested records → objects; sequences → arrays. Terminal sink → one object per element (NDJSON) |
| `jsonPretty()` | none | same as `json()` | Same as `json()`, with 2-space indentation |
| `csv()` | none | scalar fields only | Joins all fields **in record field order** with `,` (no CSV escaping) |
| `spaced()` | none | scalar fields only | Joins all fields in record field order with a single space |
| `tabbed()` | none | scalar fields only | Joins all fields in record field order with a tab |

`sfv` / `checksum` / `csv` / `spaced` / `tabbed` only accept scalar fields (`String`, `Int`, `Bool`). `json` / `jsonPretty` additionally accept nested `Record` and `Seq` of JSON-compatible values (`File` / `Dir` / `Hash` are still errors, though). There's **no** check that the digest field in `sfv`/`checksum` is actually CRC32; that's on you. Delimited joins are naive: paths containing `,` don't get quoted or escaped.

### 4.8 Hash-check methods (`File` / `String`)

On a `File` or `String` receiver, a call whose name is a known hash algorithm and that takes **one** string argument compares the computed digest against the expected value and returns a **`Bool`**:

```text
from file f in 'x'
where f.md5('529DF104CA7D7EC2E4B9E4EAB5557CF8')
select f.path;
```

```text
from string s in 'abc'
let valid = s.md5('900150983CD24FB0D6963F7D28E17F72')
let result = { path = 'x', valid }
select result.json();
# → {"path":"x","valid":true}
```

| Form | Args | Result | Notes |
|------|------|--------|-------|
| `recv.<hash>(expected)` | one `String` | `Bool` | `<hash>` can be any algorithm `hc` knows (same set as the hash properties). Comparison is **case-insensitive**, same as `recv.<hash> == expected` (§5.3). On `File`, it honors that value's `limit`/`offset` window (§4.5). A one-element sequence (nested query or `let`-bound `Seq`) unwraps to that string (§5.2). |

This is sugar for an equality check against the hash property. A mismatch returns `false`; it doesn't raise an error. Wrong arity, a non-string argument, or a receiver that isn't `File`/`String` are all errors. If a Record formatter name (§4.7) ever collides with an algorithm name, the formatter wins.

Bare `recv.<hash>` without a call is still just a property, and still yields the hex digest as a `String`.

### 4.9 Counting a sequence (`Seq.count()`)

`count` is a **`Seq`-only** method. It returns an **`Int`**: how many elements are in a sequence that has already been collected.

| Call | Args | Result | Notes |
|------|------|--------|-------|
| `recv.count()` | none | `Int` | Receiver must be a `Seq`. An empty sequence yields `0`. This reads the stored length; it does not re-walk sources or recompute properties. |

`let` is a query-body clause (§5.1), so it cannot open a statement. Every query still starts with `from`, and the outer query still needs its own `select` or `group` after the `let`. The nested query's `select` only closes the right-hand side of the `let`. It does not finish the outer query. If you put a semicolon after the inner `select` and then write a bare `select …count…`, that second line is not a valid query on its own.

The usual pattern is an outer `from` for context, a `let` that binds a nested sequence, then an outer `select` of the count:

```text
from string s in 'abc'
let items = from string t in s select t
select items.count();
# → 1
```

```text
from string s in 'abc'
let items = from string t in s where false select t
select items.count();
# → 0
```

```text
from dir d in '/tmp'
let files = from file f in d.tree().skipErrors()
            where f.readable
            select f
select files.count();
```

```text
… group f by f.size into g
select g.items.count();
```

Nested queries (including ones bound with `let`) always produce a `Seq`, even when the result is empty or has a single element, so they are the easy receivers for `count()`. Script-level `select … into id;` is different: a single projected row binds `id` as a scalar, not a `Seq` (§5.1), so `id.count()` is an invalid method receiver when there was exactly one row. Use `let` or a nested query when you care about the length. If a later statement only prints the count of a script-bound name, that statement still needs a leading `from` (for example `from string _ in 'x' select files.count();` after `… select f into files;`).

Calling `count()` on a non-`Seq`, or with arguments, is an error. There is no `count` property; bare `recv.count` without `()` is an invalid property on `Seq`.

---

## 5. Queries & expressions

This section covers how a query is put together, and what the expression language inside its clauses looks like.

### 5.1 Query structure

The concrete syntax follows l2h's own grammar (`from` … body … `;`), not SQL's `select … from` ordering.

Conceptually, the shape is:

```text
from_clause
query_body_clauses*     -- where | let | join | orderby | additional from | …
select_or_group_clause
query_continuation?     -- into identifier query_body
```

You can put multiple queries in one translation unit, separated by semicolons. A comment is a `#…` line of its own between top-level queries; it's ignored. You can't put one inside a query body or at the end of a code line: the lexer only recognizes `#` at the start of a line, and the grammar only accepts a comment where a whole query could stand.

If a query ends with `into id` and nothing after it (just the semicolon), the projected result is stored under `id` in a **script environment**. Later queries in the same translation unit can use that name:

```text
from string s in 'abc' select s.md5 into h;
from string t in 'xyz' where t.md5 != h select t;
```

- `select expr into id;` and `group … into id;` with no following clauses do not print; they only bind `id`.
- One projected row → `id` is that value. Several rows → `id` is a `Seq` (iterate with `from T x in id`).
- Each later query starts with the script env as its outer environment, so prior binds show up in expressions and as `from`/`join` sources.
- Binding the same `id` again overwrites the earlier value.
- Query-continuation `into` (§6.8) is different: it still needs a following body and runs in a fresh one-name env.

### 5.2 Expression forms

Inside clauses you write expressions, and the supported forms are:

- String, integer, and boolean literals, including bare `true` / `false` in `where` and `select`, and signed integer literals like `-1`
- String literals `'…'` / `"…"` have no escapes, so a path like `'c:\Windows'` keeps its backslash
- Byte-string literals `b'…'` and `b"…"` are the same thing and support `\xNN`, `\\`, `\'`, `\"`, `\n`, `\r`, `\t`. Bad or truncated escapes are compile errors. They still evaluate to `String`; digests from hash properties stay ASCII hex (`is_digest`)
- A range identifier on its own
- Property access `id.prop`
- Method call `id.method(args…)` or `{…}.method(args…)`: Record formatters §4.7, hash-check on `File`/`String` §4.8, `Dir.tree()` / `Dir.skipErrors()` §4.6, `File.offset(n)` / `File.limit(n)` §4.5, or `Seq.count()` §4.9 (hash-check needs a bound `File`/`String` identifier; you can't call it on a bare literal record)
- Bool-typed expressions as bare `where` predicates (hash-check methods, `f.readable`, `let`-bound `Bool`, nested-query **exists**, and named `Seq` values such as `g.items`: non-empty → true)
- Relational operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `~`, `!~`. Ordering comparisons `>` / `>=` / `<` / `<=` are **`Int`-only**; `==` / `!=` follow §5.3; `~` / `!~` are **`String`-only** (§5.3)
- Boolean operators: `&&`, `||`, `!`, and parentheses. `!` on a `Seq` (named or nested) is negated exists
- Anonymous objects: `{ e1, e2, … }` and `{ name = e, … }` → `Record` (§5.4)
- Nested query expressions used as **values** in `let`, `select`, and anonymous-record fields
- Nested queries and named `Seq` values as method arguments: a one-element sequence unwraps to that element; anything else is a runtime `TypeMismatch`. Comparisons are stricter: they only unwrap nested-query operands, not names like `g.items`
- Nested query expressions in **`where`** and **`orderby`**:
  - as a predicate: a non-empty result counts as true (**exists**); the same exists rule applies to a bare named `Seq`
  - as a comparison / order key operand: **singleton unwrap** applies (exactly one element is expected; anything else is a runtime `TypeMismatch`). Named `Seq` values (like `g.items`) are **not** unwrapped this way.
- Nested query expressions in **`from … in …` / `join … in …` sources**: the nested query has to yield a `Seq` whose item kind matches the declared range type (or a scalar path payload, for singleton sources).
- Nested query expressions in **join keys** (`on … equals …`): the same **singleton unwrap** rule as comparisons applies.
- Nested query expressions in **`group … by` keys**: the same **singleton unwrap** rule applies; the stored group `key` is the unwrapped scalar.

A nested query used in a value position **doesn't carry its own `into` continuation**. An `into` that follows a nested `select`/`group` actually binds to the **outer** query instead. Top-level queries still support `into` the normal way.

### 5.3 Equality, ordering, and join-key normalization

Comparisons (join keys, and `orderby` keys) normalize their operands like this:

- `Int` / `Bool`: exact equality, nothing fancy.
- `String` keys that are **hex digests** coming from **hash-property results** (and comparisons against digest string literals): **case-insensitive**, whenever either operand is a digest value. Sorting digest strings in `orderby` uses the same rule.
- Other strings, including hex-looking plain text that isn't actually a digest, get exact equality (byte / code-unit identity, as stored).
- Mixed kinds in `==`: an error in v1.0. No implicit coercion (that could change later).
- Ordering operators `>` / `>=` / `<` / `<=`: both operands must be **`Int`**. `String` / `Bool` ordering is only via `orderby`, not these operators.

For the regex operators `~` / `!~`: both operands must be **`String`**. The left is the subject, the right is the pattern (existing `matchRe` behavior). No implicit stringification of `Int` / `Bool` / other kinds. A pattern that fails to compile is a **runtime error** (bad regex), not a silent non-match.

### 5.4 Anonymous object field names

Every element of `{ … }` becomes a named field, and there are exactly three ways to name one:

| Field syntax | Result |
|--------------|--------|
| `name = expr` | Explicit field name `name` |
| `id.prop` | Auto-name `prop` |
| bare `id` | Auto-name `id` |

Anything else unnamed inside `{…}` is a compile-time error. It either needs an explicit name, or it needs to be one of the auto-nameable forms above. Duplicate field names within one record, whether from explicit aliases or from auto-names colliding, are also a compile-time error.

---

## 6. Clauses

Each clause is a pure transformation of the current sequence, unless a note says otherwise. I/O only shows up when a property forces filesystem/hash work, or when a **terminal sink** prints (§7).

### 6.1 `from`
See §3.3-3.4. Extends or replaces the working sequence of environments.

### 6.2 `let id = expr`
Binds a name per row: for each `Env`, evaluate `expr` and yield `Env ∪ { id ↦ value }`. This just adds `id`, shadowing any existing binding of that name.

### 6.3 `where pred`
Keeps only the environments for which `pred` is true. Any properties named inside `pred` get forced as needed, on demand.

### 6.4 `join`

**Inner equijoin** ("keep pairs whose keys are equal"), without `into`:
```text
join T y in src on e1 equals e2
```
For each outer `Env` and each element `y` from `src` (typed/`from`-rules as in §3.3; if `src` is a `Dir` value, the same rules as `from file … in dir` apply when `T` is `file`, and so on), if `normalize(e1(outer)) == normalize(e2(inner_env))`, yield outer ∪ `{ y ↦ inner }`.

**Group join** (`join … into g`):
For each outer `Env`, bind `g` to the **sequence** of matching inner elements (which may well be empty). It doesn't flatten; typically you'd follow it with `from z in g` to SelectMany over it.

### 6.5 `orderby`
```text
orderby e1 [ascending|descending], e2 …
```
Materializes the sequence and sorts it stably by the evaluated keys. Ascending is the default direction if you don't specify one.

Keys have to be order-comparable in v1.0 (`Int`, `String`, `Bool`); unsupported key shapes should get rejected at compile time whenever the type's already known. If an incomparable value shows up at runtime anyway, `orderby` fails with `TypeMismatch`. Digest `String` keys sort case-insensitively (§5.3).

### 6.6 `group expr by key`
Groups the current sequence by `key`. Each group element comes out as an ordinary **`Record`** with two fields:

- `key`: the grouping key value
- `items`: a `Seq` of the evaluated **group projection** (`expr`) for each row in the group, not full environments

Grouping supports `into` and a subsequent `select` over those two fields, same as anywhere else.

`key` has to be equality-comparable in v1.0 (`Int`, `String`, `Bool`); unsupported key shapes should get rejected at compile time whenever the type's known. If an incomparable value turns up at runtime, grouping fails with `TypeMismatch`.

### 6.7 `select expr`
Maps each environment to a projected `Value` (`expr`).

The same `select` keyword does two different jobs depending on what follows it:

| Context | Effect |
|---------|--------|
| `select` / `group` is the **last** operation (no `into` continuation) | **Sink**: prints the projected sequence to stdout (§7) |
| Followed by `into id` and a continuation body | **No print**; projected values stream into the continuation with `id` bound per row (§6.8) |
| Followed by `into id;` with no continuation body | **No print**; bind `id` in the script environment for later queries in the same unit (§5) |

### 6.8 `into id` (query continuation)
```text
… select expr into id
  where …
  select …
```
1. Finishes the projection as a `Seq` (not a sink).
2. Binds `id` as the range variable over that sequence for the following `query_body`.
3. The continuation runs in a **fresh** environment that contains **only** `id`. Outer range variables from before the `select`/`group` are not visible. (Compile and runtime share this rule.)
4. Identifier registration has to **define** `id` in scope. It must not delete the name (a past `INTO` bug did that).

The same continuation idea applies after `group … by … into id`. **Group-join** `join … into g` is different on purpose: it keeps the outer row and adds `g` as the group sequence (C# query semantics).

---

## 7. Terminal output

When a projection acts as a **sink** (the last operation, with no `into`), the result gets printed:

- Each projected element produces output.
- For a **single** property / scalar projection (`String` / `Int` / `Bool`): one line per element (e.g. a hex digest).
- For a projected `File` / `Dir`: one line with the bound path. For a projected `Hash`: one line with the bound digest string.
- For a projected `Seq`: expand recursively, one sink pass per item (so N items can produce more than N lines if items are themselves records).
- For an **anonymous object** with multiple fields, like `{ f.md5, f.sha1 }`: **one line per field**, in field order. So two fields means **two lines** per input element, not one. Each field value must be a scalar or path-like (`String` / `Int` / `Bool` / `File` / `Dir` / `Hash`); a `Seq`- or nested-`Record`-valued field is a runtime `TypeMismatch` (use `into` + `from` to flatten group `items`).

Exact line formatting (prefixed names vs. bare values) should match whichever golden format the tests settle on; the default proposal is **bare values only**, one per line.

The sink flushes each line as it goes. If a later row fails, earlier lines may already be on stdout.

---

## 8. Errors

| Class | Examples |
|-------|----------|
| Syntax | Existing grammar failures |
| Semantic (compile) | Undefined range variable; disallowed property for declared type; unknown/invalid method calls |
| Runtime | Missing file/dir; I/O errors; hash failures; bad regex; offset past EOF |

Queries fail fast once an error is raised. Sink output is still progressive (§7), so earlier rows may already be on stdout.

---

## 9. Implementation architecture

The runtime lowers the compiled plan into **Volcano-lite pull operators** (`open` / `next` / `close`) over `From` / `Clause` / `Expr`. There is still no bytecode / register VM and no global instruction tape (see §1.1). Query operators and expression evaluation stay separate; sink and collect are outer drivers over the operator tree. Nested query values get compiled and executed recursively, yielding `Seq(Value)`.

Pipeline:

```text
source text
  → parse (flex/bison) → AST
  → compile-time check (`compile.zig`) → `*From` plan
  → buildOps + interpret (`interpret.zig`)
       ↳ pull operators over the plan (where/let/from/join/orderby/group/select)
       ↳ eval Expr against Env (demand-driven props)
       ↳ Dir walks hand off one file at a time (no full path list up front)
       ↳ `orderby` / `group by` (and nested queries that build a `Seq`) collect first, then continue
       ↳ terminal select/group → sink or collect driver; `into` → continuation body
```

Each clause maps onto a plan shape in `plan.zig`:

| Clause | Plan shape |
|--------|------------|
| `from T x in E` | `From` / `Clause.from`, `source=.expr` |
| `from file f in d` (`d`: Dir) | same `source=.expr`; runtime walks the Dir and yields files one by one (full list only when a later `orderby` / `group by` needs it) |
| `let` / `where` | `Clause.let` / `Clause.where` |
| `join` / `join … into g` | `Join` (`group_into` null / set) |
| `orderby` | `Clause.order_by` (collect + stable sort) |
| `group … by` | `Clause.group_by` → Record `{ key, items }`; optional `into` |
| `select` / `select … into` | `Select` sink vs continuation |

There's no global `sources` tape, and nothing coupled to an instruction index.

| Area | Status |
|------|--------|
| IR modules | `plan.zig`, `expr.zig`, `value.zig`, `interpret.zig` |
| Compile-time check / IR | `compile.zig` |
| LINQ clauses | `from`, `where`, `let`, `join`, `join … into`, `orderby`, `group by`, `select`, `into` |
| Properties | Demand-driven catalog in `props.zig` (§4.3) |
| Methods | Catalog + formatters in `method.zig`: §4.7 formatters; §4.8 hash-check; §4.6 `Dir.tree` / `Dir.skipErrors`; §4.5 `File.offset` / `File.limit`; §4.9 `Seq.count` (`arityRange`) |
| Recursive dir walk | Yes: `from file f in d.tree()` / `d.tree(n)` / `d.skipErrors()` (§3.4 / §4.6) |
| Runtime model | Pull operators over the plan tree (`open` / `next` / `close`) |

**Known limitations**: some mixed/`unknown` sequence shapes and I/O failures only get detected at runtime, not statically.

Rejected for this stack (historical note): packed bytecode / register VM; a SQL-style cost-based optimizer.

---

## 10. Design decisions

This section exists to explain why the behavior is what it is. It's reference material, not a source of new rules.

| Topic | Decision |
|-------|----------|
| Record auto-names | `id.prop` → field `prop`; bare `id` → `id`; any other expr in `{…}` → **error** (§5.4) |
| `from file f in d` | Receiver must be **`Dir`** only |
| Symlinks in flat dir listing | **Skip** all symlinks |
| Hex digests | Computed (`File`/`String`) **lowercase**; `Hash` restore keeps bound casing; compare / `orderby` case-insensitive (§5.3) |
| Multi-statement `into id;` | Bind in script env (no print); one row → scalar, many → `Seq`; later queries see the name (§5) |
| `group proj by key` element | Record `{ key, items }` where `items` is the `Seq` of evaluated projections |
| File `limit` / `offset` | `f.offset(n)` / `f.limit(n)` return a new `File`; properties only read; default `limit` is `maxInt(i64)`; hashes on that value follow `hc`; offset past EOF is an error (§4.5) |
| `~` / `!~` operands | Both **`String`** (subject ~ pattern); no stringify; bad pattern → runtime error (§5.3) |
| `>` / `<` / `>=` / `<=` | **`Int`-only**; `String`/`Bool` ordering only via `orderby` (§5.3) |
| Dir `tree` / `skipErrors` | `tree()` unlimited, `tree(n)` enter-depth limited (`tree(0)` ≡ flat); `skipErrors()` soft-skips walk/enter failures; compose freely; never follows symlinks; file order is walk order, sort with `orderby` (§4.6 / §3.4) |
| Boolean literals | `true` / `false` work as values and as bare predicates (§5.2) |
| String literals | `'…'`/`"…"` have no escapes; `b'…'`/`b"…"` add `\xNN` and friends; both are `String`; digests stay ASCII hex (§5.2) |
| Bare bool predicates | Hash-check / `let`-bound `Bool` / nested-query exists / named-`Seq` exists are valid `where` predicates (§5.2) |
| Record methods | Formatters only on `Record`; return `String`; lowercase names like properties (§4.7) |
| Hash-check methods | `File`/`String`.<hash>(expected) → `Bool`; case-insensitive; same window rules as hash props; one-element `Seq` args unwrap (§4.8) |
| `Seq.count()` | `Seq`-only; arity 0; returns `Int` (stored length); empty → `0`; not a property; prefer `let`/nested over singleton script-`into` (§4.9) |
| Method arg unwrap | Method args unwrap a one-element `Seq` (name or nested query); comparisons only unwrap nested queries (§5.2) |
| Sink output | Flush per line; `File`/`Dir`/`Hash` → path/digest line; projected `Seq` expands; Record → one line per field (§7) |
| `sfv` vs `checksum` | Lookup by field name; fixed emit order: `sfv` → `name    digest`, `checksum` → `digest path` |
| File `name` | Basename of `path` (no I/O), required field name for `sfv()` |
| Method receiver syntax | Identifier (`let` / `into`) or a record literal `{…}.method()` (§4.7) |
| Delimited methods | `csv` / `spaced` / `tabbed` still join in record field order |
| `json` shape | One object per element (NDJSON when sunk per row); not a Seq-level JSON array |
| Comments | `#…` lines of their own between queries are ignored; a comment can't sit inside a query body or after code on the same line (§5.1) |

No remaining open questions.
