# l2h Query Language — Semantics

Source of truth for how LINQ-style hash queries behave. Observable behavior matches the current `QueryPlan` interpreter unless a section notes a known limitation.

> **How to read this document**
> §1 is the mental model and what is out of scope. §2 is a short tour with runnable examples. §3–§8 are the reference (values, properties, queries, clauses, output, errors). §9 is the implementation layout. §10 records settled design choices.
> Tables and formal rules give the exact behavior; the prose around them is for *why* and *how to use it*.

---

## 1. Purpose & mental model

l2h is a small query language for hashing work. A **query expression**:

1. Binds **range variables** to data sources (`string`, `file`, `dir`, `hash`). A *range variable* is the LINQ name for the per-element loop variable — `f` in `from file f in …` — bound to each source element in turn. (The term comes from relational calculus: the variable *ranges over* a relation. It is not a numeric interval.)
2. Pipes a **sequence of environments** through LINQ-style clauses (`where`, `let`, `join`, `orderby`, `group`, …).
3. Reads **demand-driven computed properties** — file size, digests, and the like — only when the query asks for them.
4. Either **prints** the final result, or hands it to the next stage with **`into`**.

A query is a pipeline: each clause takes the current sequence and produces a new one. I/O is lazy — it happens when a property forces a filesystem read or a hash, or when the terminal step prints.

```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```

Left to right: open one file, keep it if non-empty, print its MD5. `size` is a cheap `stat`; `md5` is a full read + hash. The body is hashed only when the file passes the filter.

> **Cost note.** Hashing dominates runtime, not interpreter overhead. The interpreter favors clarity and testability; micro-optimizations are out of scope (see §1.1).

### 1.1 Boundaries

These are **not** part of l2h:

- **Method calls** (`x.foo(...)`) — parsed, then rejected as a semantic error. Use property access (`x.prop`) instead (§4).
- **Built-in recursive directory walk** — `from dir` lists only immediate children. Recursion is left for a later version (§3.4).
- **Bytecode / register VM** — the runtime is a tree-walking interpreter. No global instruction tape, no instruction-index coupling.
- **Interpreter performance tuning beyond correctness.**

---

## 2. A quick tour

Real queries first; the rest of the document refers back here.

**Hash one file, keep non-empty ones:**
```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```
`size` is a `stat` (cheap); `md5` forces a read + hash. The body is hashed only for files that pass `where`.

**Find a file by a known digest, then read its size:**
```text
from file f in '/home/user/file'
where f.md5 == 'd41d8cd98f00b204e9800998ecf8427e'
select f.size;
```
Here `md5` is forced inside `where`; `size` is cheap afterward. Digest comparison is case-insensitive (§5.2).

**List large files under a directory:**
```text
from dir d in '/tmp'
from file f in d
where f.size > 1000
select f.sha1;
```
`from dir` binds a directory; the *second* `from` iterates its immediate regular files. Subdirectories and symlinks are skipped (§3.4).

**Join two sources on equal digests:**
```text
from string a in 'abc'
join string b in 'abc' on a.md5 equals b.md5
select { a.md5, b.md5 };
```
An inner equijoin: keep pairs whose digests match (case-insensitive). `select` of two fields prints two lines per joined row (§7).

**Restore / reverse a known digest:**
```text
from hash x in 'D41D8CD98F00B204E9800998ECF8427E'
select x.md5;
```
`from hash` does **not** hash the literal characters — it treats the digest as input to a restore lookup for `md5` (§4.4).

**Carry a result into a second query with `into`:**
```text
from file f in 'x'
select f.md5 into h
select h;
```
The first `select` does **not** print — it hands its result to `h`. The second `select` is the terminal sink and prints the sequence of digest strings.

---

## 3. Values & sources

What flows through a query: value kinds, how names are tracked (the *environment*), and how each `from` form feeds the pipeline.

### 3.1 Value kinds

A value is one of these runtime kinds:

| Kind | Meaning |
|------|---------|
| `String` | Text payload (also used for digests and paths expressed as strings) |
| `File` | A regular file identified by its path |
| `Dir` | A directory identified by path — the source of enumeration, not a row by itself |
| `Hash` | A restore source: a digest string; the algorithm is chosen via a property or `select` |
| `Int` | Signed integer (sizes, numeric literals) |
| `Bool` | A predicate result |
| `Record` | An anonymous object / product of `let`, `join`, or `{…}` shaping |
| `Seq(T)` | A lazy sequence of values or environments |

### 3.2 Environment

An **environment** (`Env`) is a finite map from range-variable names to `Value`. Each clause consumes a sequence of environments and produces a new one — except after `select`/`group`, which produce a sequence of projected values.

When a rule writes "`Env ∪ { id ↦ value }`", read it as: bind `id` to the value, shadowing any previous binding of that name.

### 3.3 Source sequences (`from`)

The opening `from` (and any later `from` in the body) is where data enters the pipeline. Each form yields its own starting sequence:

| Declaration | Produced sequence |
|-------------|-------------------|
| `from string x in E` | Singleton: one `String` from evaluating `E` |
| `from file x in E` | Singleton: one `File` for path `E` (error if missing or not a regular file) |
| `from dir x in E` | Singleton: one `Dir` for path `E` (error if missing or not a directory) |
| `from hash x in E` | Singleton: one `Hash` whose digest comes from `E` |

**Directory contents are not implicitly flattened into the range variable of `from dir`.** `from dir` gives you the directory *itself*; to reach the files inside, write an explicit second `from`:

```text
from dir d in '/tmp'
from file f in d
where f.size > 0
select f.md5;
```

Here `from file f in d` means: for the current `Dir` bound to `d`, emit one environment per **immediate child regular file** (see §3.4), binding `f` to that `File` (`d` stays in scope unless shadowed).

**Type rule for that form:** the expression after `in` must evaluate to a **`Dir`**. A bare path string is *not* accepted here — write `from dir d in '/tmp'` first, then `from file f in d`. A runtime kind mismatch is an error.

Any additional `from` in the body is a **SelectMany** ("for each outer row, evaluate the inner source and concatenate the extended environments"). Nested `from`s flatten naturally.

### 3.4 Directory enumeration

When `from file f in <Dir>` iterates a directory:

- **Flat only** — immediate children, nothing deeper.
- **Regular files only** — **skip all symlinks** (whether they point at a file or a directory) and skip subdirectories.
- **No recursive walk** — a future version may expose recursion via filters and synthetic properties, not a built-in recursive `from dir`.

Child order is implementation-defined but **deterministic** for a given filesystem snapshot (the chosen order is documented in tests, e.g. lexicographic by name).

---

## 4. Computed properties

Properties (`size`, `md5`, `path`, …) connect a value to the work you care about. **They are computed on demand** — that is what makes filter-before-hash work.

### 4.1 Demand-driven evaluation (key idea)

A property is computed on **first read** during expression evaluation — wherever it appears: `where`, `let`, join keys, `orderby`, `group by`, `select`, and so on. An implementation **may** cache the result on the value for the rest of the query.

There is no separate "hashing phase" before or after filtering. Whether a hash runs before or after a filter follows from **which properties the query forces**:

```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```
`size` is read in `where` (stat only); `md5` is read in the terminal `select`.

```text
from file f in '/home/user/file'
where f.md5 == 'D41D8CD98F00B204E9800998ECF8427E'
select f.size;
```
`md5` is forced in `where` (file read + hash); `size` is still cheap afterward.

Put cheap predicates (`size`, `path`) before expensive ones (`<hash>`) in `where` so you do not hash rows you will discard.

### 4.2 Access syntax

Syntax: `range.prop` (property access). **Method calls** (`range.m(...)`) are **out of scope** for v1.0 and must be rejected (parse or semantic error).

### 4.3 Property catalog

Allowed properties depend on the **runtime kind** of the receiver. An unknown property for that kind is an error — preferably static when the range type is known at compile time.

| Receiver | Property | Result | Notes |
|----------|----------|--------|-------|
| `File` | `path` | `String` | Path identifying the file (no I/O; projects the bound path) |
| `File` | `size` | `Int` | File size in bytes |
| `File` | `<hash>` | `String` | Hex digest of file contents; `<hash>` is any algorithm name known to `hc` (e.g. `md5`, `sha1`, `tiger`, …) |
| `String` | `size` | `Int` | Length in bytes (UTF-8 payload length as stored) |
| `String` | `<hash>` | `String` | Hex digest of string bytes |
| `Hash` | `<hash>` | `String` | **Restore** path: treat the bound digest as the input digest for algorithm `<hash>` (same meaning as legacy `from hash … select x.md5`), *not* "hash the digest characters as a string" |
| `Dir` | `path` | `String` | Path identifying the directory (no I/O; projects the bound path). Use `from file f in d` to reach files |
| `Record` | field name | field value | Fields introduced by `{…}`, `let`, or join shaping |
| `Int` / `Bool` / `Seq` | — | — | No properties in v1.0 |

Hex digests from hash properties are **lowercase** when printed and when produced as `String` values; equality and join keys still use **case-insensitive** comparison (§5.2).

### 4.4 `from hash` + select (restore)

```text
from hash x in 'D41D8CD98F00B204E9800998ECF8427E'
select x.md5;
```

This restores / reverses with algorithm `md5` and the digest literal — work goes to the existing hash-restore runners in `modes`. It does **not** mean "compute md5 of the hex string".

---

## 5. Queries & expressions

How a query is built, and the expression language inside its clauses.

### 5.1 Query structure

Concrete syntax follows the existing grammar (`from` … body … `;`), not SQL's `select … from`.

Conceptual shape:

```text
from_clause
query_body_clauses*     -- where | let | join | orderby | additional from | …
select_or_group_clause
query_continuation?     -- into identifier query_body
```

Multiple queries may appear in one translation unit (semicolon-separated); comments (`#…`) are ignored.

### 5.2 Expression forms

Inside clauses you write expressions. Supported forms:

- String and integer literals
- Range identifier
- Property access `id.prop`
- Relational: `==`, `!=`, `>`, `>=`, `<`, `<=`, `~`, `!~`
- Boolean: `&&`, `||`, `!`, parentheses
- Anonymous object: `{ e1, e2, … }` and `{ name = e, … }` → `Record` (§5.4)
- Nested query expressions as **values** in `let`, `select`, and anonymous-record fields
- Nested query expressions in **`where`** and **`orderby`**:
  - as a predicate: non-empty result → true (**exists**)
  - as a comparison / order key operand: **singleton unwrap** (exactly one element; otherwise runtime `TypeMismatch`). Named `Seq` values (e.g. `g.items`) are **not** unwrapped.
- Nested query expressions in **`from … in …` / `join … in …` sources**: the nested query must yield a `Seq` whose item kind matches the declared range type (or a scalar path payload for singleton sources).
- Nested query expressions in **join keys** (`on … equals …`): same **singleton unwrap** rule as comparisons.
- Nested query expressions in **`group … by` keys**: same **singleton unwrap** rule; the stored group `key` is the unwrapped scalar.

A nested query in a value position **does not carry its own `into` continuation** — an `into` after a nested `select`/`group` binds to the **outer** query. Top-level queries still support `into` as usual.

### 5.3 Equality and join-key normalization

Comparisons (and join keys) normalize operands as follows:

- `Int` / `Bool`: exact equality.
- `String` keys that are **hex digests** from **hash-property results** (and comparisons against digest string literals): **case-insensitive** when either operand is a digest value.
- Other strings (including hex-looking plain text): exact equality (byte / code-unit identity as stored).
- Mixed kinds in `==`: error, unless an explicit coercion is added later (v1.0: error).

For the regex operators `~` / `!~`: the left operand is stringified; the right is a pattern string (the existing `matchRe` intent).

### 5.4 Anonymous object field names

Each element of `{ … }` becomes a named field. Ways to name one:

| Field syntax | Result |
|--------------|--------|
| `name = expr` | Explicit field name `name` |
| `id.prop` | Auto-name `prop` |
| bare `id` | Auto-name `id` |

Any other unnamed expression is a compile-time error — it must be either explicitly named or auto-nameable. Duplicate field names in one record (from explicit aliases or auto-names) are also a compile-time error.

---

## 6. Clauses

Each clause is a pure transformation of the current sequence unless noted. I/O appears only when a property forces filesystem/hash work, or when a **terminal sink** prints (§7).

### 6.1 `from`
See §3.3–3.4. Extends or replaces the working sequence of environments.

### 6.2 `let id = expr`
Bind a name per row: for each `Env`, evaluate `expr` and yield `Env ∪ { id ↦ value }` — add `id`, shadowing any existing binding.

### 6.3 `where pred`
Keep only environments for which `pred` is true. Properties named in `pred` are forced as needed.

### 6.4 `join`

**Inner equijoin** ("keep pairs whose keys are equal") — no `into`:
```text
join T y in src on e1 equals e2
```
For each outer `Env` and each element `y` from `src` (typed/`from`-rules as in §3.3; if `src` is a `Dir` value, the same rules as `from file … in dir` apply when `T` is `file`, etc.), if `normalize(e1(outer)) == normalize(e2(inner_env))`, yield outer ∪ `{ y ↦ inner }`.

**Group join** (`join … into g`):
For each outer `Env`, bind `g` to the **sequence** of matching inner elements (possibly empty). It does not flatten; typically follow with `from z in g` to SelectMany.

### 6.5 `orderby`
```text
orderby e1 [ascending|descending], e2 …
```
Materialize the sequence and sort it stably by the evaluated keys. Default direction is ascending.

Keys must be order-comparable in v1.0 (`Int`, `String`, `Bool`); unsupported key shapes should be rejected at compile time when the type is known. If incomparable values appear at runtime, `orderby` fails with `TypeMismatch`.

### 6.6 `group expr by key`
Group the current sequence by `key`. Each group element is an ordinary **`Record`** with fields:

- `key` — the grouping key value
- `items` — `Seq` of the grouped elements (environments or prior projections, as produced by the upstream clause)

Grouping must support `into` and a subsequent `select` over those fields.

`key` must be equality-comparable in v1.0 (`Int`, `String`, `Bool`); unsupported key shapes should be rejected at compile time when the type is known. If incomparable values appear at runtime, grouping fails with `TypeMismatch`.

### 6.7 `select expr`
Map each environment to a projected `Value` (`expr`).

The same `select` keyword does two jobs depending on what follows:

| Context | Effect |
|---------|--------|
| `select` / `group` is the **last** operation (no `into` continuation) | **Sink**: print the projected sequence to stdout (§7) |
| Followed by `into id` | **No print**; the projected sequence becomes the input bound to `id` for the continuation body |

### 6.8 `into id` (query continuation)
```text
… select expr into id
  where …
  select …
```
1. Finish the projection as a `Seq` (not a sink).
2. Bind `id` as the range variable over that sequence for the following `query_body`.
3. Identifier registration must **define** `id` in scope (a legacy bug — deletion on `INTO` — must not come back).

The same continuation idea applies after `group … by … into id` and after `join … into id` (a group-join already binds `id` as the group sequence per outer row; naming aligns with C# query semantics).

---

## 7. Terminal output

When a projection is a **sink** (last operation, no `into`), the result is printed:

- Each projected element produces output.
- For a **single** property / scalar projection: one line per element (e.g. a hex digest).
- For an **anonymous object** with multiple fields, e.g. `{ f.md5, f.sha1 }`: **one line per field**, in field order — so two fields ⇒ **two lines** per input element.

Exact line formatting (prefixed names or bare values) should match a chosen golden format in tests; the default proposal is **bare values only**, one per line.

---

## 8. Errors

| Class | Examples |
|-------|----------|
| Syntax | Existing grammar failures |
| Semantic (compile) | Undefined range variable; disallowed property for declared type; methods |
| Runtime | Missing file/dir; I/O errors; hash failures; bad regex |

Failed queries should not partially commit confusing sink output beyond what tests specify (prefer fail-fast per query).

---

## 9. Implementation architecture

The runtime is a **tree-walking** interpreter over `QueryPlan` / `Expr` (not a register/bytecode VM). Query operators and expression evaluation live in separate modules. Nested query values are compiled and executed recursively and yield `Seq(Value)`.

Pipeline:

```text
source text
  → parse (flex/bison) → AST
  → compile-time check (`compile.zig`) → QueryPlan
  → interpret (`interpret.zig`)
       ↳ eval Expr against Env (demand-driven props)
       ↳ terminal select/group → sink print; `into` → continuation body
```

Each clause maps to a plan shape in `plan.zig`:

| Clause | Plan shape |
|--------|------------|
| `from T x in E` | `From` / `Clause.from`, `source=.expr` |
| `from file f in d` (`d`: Dir) | `From` + `files_in_dir` |
| `let` / `where` | `Clause.let` / `Clause.where` |
| `join` / `join … into g` | `Join` (`group_into` null / set) |
| `orderby` | `Clause.order_by` (materialize + stable sort) |
| `group … by` | `Clause.group_by` → Record `{ key, items }`; optional `into` |
| `select` / `select … into` | `Select` sink vs continuation |

There is no global `sources` tape and no instruction-index coupling.

| Area | Status |
|------|--------|
| IR modules | `plan.zig`, `expr.zig`, `value.zig`, `interpret.zig` |
| Compile-time check / IR | `compile.zig` |
| LINQ clauses | `from`, `where`, `let`, `join`, `join … into`, `orderby`, `group by`, `select`, `into` |
| Properties | Demand-driven catalog §4.3 |
| Value language | Nested queries in value / where / orderby / from·join sources / join keys; record aliases |
| Static checks | Compile-time types for properties, join/group keys, records, many sources |
| Diagnostics | `fehler` via `diag.zig` (parse + compile-time/runtime spans from AST/`Expr`) |
| Tests | `frontend_test.zig`, `compile_test.zig`, `interpret.zig`; `zig build test-l2h` |
| Methods | **Out of scope** for v1.0 — parse only; compile-time check → `UnsupportedMethodCall` |
| Recursive dir walk | **Out of scope** for v1.0 — flat listing only (§3.4) |

**Known limitations**: some mixed/`unknown` sequence shapes and I/O failures are detected only at runtime.

Rejected for this stack (kept for history): packed bytecode / register VM; SQL cost-based optimizer.

---

## 10. Design decisions & versioning

Why the behavior is what it is — reference material, not new rules.

### 10.1 Resolved decisions (formerly open)

| Topic | Decision |
|-------|----------|
| Record auto-names | `id.prop` → field `prop`; bare `id` → `id`; any other expr in `{…}` → **error** (§5.4) |
| `from file f in d` | Receiver must be **`Dir`** only |
| Symlinks in flat dir listing | **Skip** all symlinks |
| Hex digests | **Print lowercase**; compare case-insensitive |
| `group proj by key` element | Record `{ key, items }` where `items` is the sequence of grouped elements |

No remaining open questions. Further semantic changes should bump the documented version and note the delta here.
