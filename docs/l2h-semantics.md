# l2h Query Language — Semantics

Status: **frozen** (v1.1).  
This document is the semantic source of truth for the LINQ-style hash query surface. Observable behavior matches the current `QueryPlan` interpreter unless a section marks a known limitation.

Compatibility with the legacy C / triple pipeline is **not** required.

**Languages:** **English** · [Русский](l2h-semantics.ru.md)

**Amendment policy:** behavioral changes require a version bump (v1.2+ for compatible extensions; v2.0 for breaking changes) and parallel updates to the Russian translation. Editorial clarifications that do not change observable behavior may land without a bump. Implementation lives in `src/l2h/plan.zig`, `expr.zig`, `value.zig`, `lower.zig`, `interpret.zig`, `diag.zig`.

---

## 1. Purpose

l2h evaluates **query expressions** that:

1. Bind **range variables** to data sources (`string`, `file`, `dir`, `hash`).
2. Transform **sequences of environments** via LINQ-like clauses.
3. Read **demand-driven computed properties** (size, digests, …).
4. Either **print** a terminal projection or pass a sequence through **`into`** for further querying.

Hash algorithm throughput dominates runtime cost; interpreter design may prioritize clarity and testability over micro-optimizations.

---

## 2. Value model

### 2.1 Value kinds

| Kind | Meaning |
|------|---------|
| `String` | Text payload (also used for digests and paths as strings) |
| `File` | Regular file identified by path |
| `Dir` | Directory identified by path (source of enumeration, not a long-lived row by itself) |
| `Hash` | Restore source: digest string (+ algorithm chosen via property / select) |
| `Int` | Signed integer (sizes, numeric literals) |
| `Bool` | Predicate result |
| `Record` | Anonymous object / product of `let` / `join` row shaping |
| `Seq(T)` | Lazy sequence of values or environments |

### 2.2 Environment

An **environment** (`Env`) is a finite map from range-variable names to `Value`.

A query clause consumes a sequence of environments and produces a new sequence of environments (or, after `select`/`group`, a sequence of projected values).

### 2.3 Source sequences (`from`)

| Declaration | Produced sequence |
|-------------|-------------------|
| `from string x in E` | Singleton: one `String` from evaluating `E` |
| `from file x in E` | Singleton: one `File` for path `E` (error if missing or not a regular file) |
| `from dir x in E` | Singleton: one `Dir` for path `E` (error if missing or not a directory) |
| `from hash x in E` | Singleton: one `Hash` whose digest comes from `E` |

**Directory contents are not implicitly flattened into the range variable of `from dir`.** To iterate files, write an explicit second `from` (SelectMany), for example:

```text
from dir d in '/tmp'
from file f in d
where f.size > 0
select f.md5;
```

Here `from file f in d` means: for the current `Dir` bound to `d`, emit one environment per **immediate child regular file** (see §2.4), with `f` bound to that `File` (and `d` still in scope unless shadowed).

**Type rule:** the expression after `in` for this form must evaluate to a **`Dir`**. A bare path string is **not** accepted here — use `from dir d in '/tmp'` first, then `from file f in d`. Runtime kind mismatch → error.

Nested / additional `from` clauses in the query body are always **SelectMany**: for each outer `Env`, evaluate the inner source and concatenate extended environments.

### 2.4 Directory enumeration (v1.1)

- **Flat only**: immediate children of the directory.
- Include **regular files** only; **skip all symlinks** (whether to file or directory) and skip subdirectories.
- **No recursive walk** in v1.1 (frozen non-goal). A future version may expose recursion via filters and synthetic properties rather than a built-in recursive `from dir`.

Order of children: implementation-defined but **deterministic** for a given filesystem snapshot (document the chosen order in tests, e.g. lexicographic by name).

---

## 3. Computed properties

### 3.1 Access

Syntax: `range.prop` (property call). **Method calls** (`range.m(...)`) are **out of scope** for v1.1 and should be rejected (parse or semantic error).

### 3.2 Demand-driven evaluation

A property is computed on **first read** during expression evaluation (in `where`, `let`, join keys, `orderby`, `group by`, `select`, …). Implementations **may** cache the result on the value for the rest of the query.

There is no separate VM “phase” before/after hashing. Whether a hash runs “before” or “after” filtering follows solely from **which properties expressions force**:

```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```

`size` is read in `where` (stat only); `md5` is read in terminal `select`.

```text
from file f in '/home/user/file'
where f.md5 == 'D41D8CD98F00B204E9800998ECF8427E'
select f.size;
```

`md5` is forced in `where` (file read + hash); `size` may still be cheap afterward.

### 3.3 Property catalog (v1.1)

Allowed properties depend on the **runtime kind** of the receiver. Unknown property for that kind → error (prefer static error when the range type is known at compile time).

| Receiver | Property | Result | Notes |
|----------|----------|--------|-------|
| `File` | `size` | `Int` | File size in bytes |
| `File` | `<hash>` | `String` | Hex digest of file contents; `<hash>` is any algorithm name known to `hc` (e.g. `md5`, `sha1`, `tiger`, …) |
| `String` | `size` | `Int` | Length in bytes (UTF-8 payload length as stored) |
| `String` | `<hash>` | `String` | Hex digest of string bytes |
| `Hash` | `<hash>` | `String` | **Restore** path: treat bound digest as input digest for algorithm `<hash>` (same meaning as legacy `from hash … select x.md5`), not “hash the digest characters as a string” |
| `Dir` | *(none in v1.1)* | — | Use `from file f in d` to reach files |
| `Record` | field name | field value | Fields introduced by `{…}`, `let`, or join shaping |
| `Int` / `Bool` / `Seq` | — | — | No properties in v1.1 |

Hex digests produced by hash properties use **lowercase** when printed and when produced as `String` values; equality / join keys still use **case-insensitive** comparison (§5.2).

### 3.4 `from hash` + select (restore)

```text
from hash x in 'D41D8CD98F00B204E9800998ECF8427E'
select x.md5;
```

Means: restore / reverse using algorithm `md5` and digest literal (behavior delegated to existing hash-restore runners in `modes`). It does **not** mean “compute md5 of the hex string”.

---

## 4. Query structure

Concrete syntax follows the existing grammar (`from` … body … `;`), not SQL’s `select … from`.

Conceptual shape:

```text
from_clause
query_body_clauses*     -- where | let | join | orderby | additional from | …
select_or_group_clause
query_continuation?     -- into identifier query_body
```

Multiple queries may appear in one translation unit (semicolon-separated); comments (`#…`) are ignored.

---

## 5. Expressions

### 5.1 Forms (v1.1)

- String and integer literals  
- Range identifier  
- Property access `id.prop`  
- Relational: `==`, `!=`, `>`, `>=`, `<`, `<=`, `~`, `!~`  
- Boolean: `&&`, `||`, `!`, parentheses  
- Anonymous object: `{ e1, e2, … }` and `{ name = e, … }` → `Record` (see §5.3)
- Nested query expressions as **values** in `let`, `select`, and anonymous-record fields  
- Nested query expressions in **`where`** and **`orderby`**:
  - as a predicate: non-empty result → true (**exists**)
  - as a comparison / order key operand: **singleton unwrap** (exactly one element; otherwise runtime `TypeMismatch`). Named `Seq` values (e.g. `g.items`) are **not** unwrapped.
- Nested query expressions in **`from … in …` / `join … in …` sources**: the nested query must yield a `Seq` whose item kind matches the declared range type (or a scalar path payload for singleton sources).
- Nested query expressions in **join keys** (`on … equals …`): same **singleton unwrap** rule as comparisons.
- Nested query expressions in **`group … by` keys**: same **singleton unwrap** rule; the stored group `key` is the unwrapped scalar.

Nested queries in value positions **do not carry their own `into` continuation**; an `into` after a nested `select`/`group` binds to the **outer** query. Top-level queries still support `into` as usual.

### 5.2 Equality and join-key normalization

- `Int` / `Bool`: exact equality.  
- `String` keys that are **hex digests** (join/`==` on hash-property results and hash literals): compare with **case-insensitive** normalization (e.g. lowercase hex).  
- Other strings: exact equality (byte/code-unit identity as stored).  
- Mixed kinds in `==`: error or defined coercion only if explicitly added later (v1.1: error).

Regex operators `~` / `!~`: left operand stringified; right operand is a pattern string (existing `matchRe` intent).

### 5.3 Anonymous object field names

Each element of `{ … }` becomes a named field.

Supported forms:

| Field syntax | Result |
|--------------|--------|
| `name = expr` | Explicit field name `name` |
| `id.prop` | Auto-name `prop` |
| bare `id` | Auto-name `id` |

Any other unnamed expression is a lowering error (must be either explicitly named or auto-nameable).

Duplicate field names in one record (from either explicit aliases or auto-names) are a lowering error.

---

## 6. Clause semantics

Each clause is a pure transformation of the current sequence unless noted (I/O appears only when properties force filesystem/hash work, and when a **terminal sink** prints).

### 6.1 `from`

See §2.3–2.4. Extends or replaces the working sequence of environments.

### 6.2 `let id = expr`

For each `Env`, evaluate `expr`, yield `Env ∪ { id ↦ value }` (shadowing if `id` already exists).

### 6.3 `where pred`

Retain environments for which `pred` evaluates to true. Properties mentioned in `pred` are forced as needed.

### 6.4 `join`

**Inner equijoin** (no `into`):

```text
join T y in src on e1 equals e2
```

For each outer `Env` and each element `y` from `src` (typed/`from`-rules as in §2.3; if `src` is a `Dir` value, same rules as `from file … in dir` when `T` is `file`, etc.), if `normalize(e1(outer)) == normalize(e2(inner_env))`, yield outer ∪ `{ y ↦ inner }`.

**Group join** (`join … into g`):

For each outer `Env`, bind `g` to the **sequence** of matching inner elements (possibly empty). Does not flatten; typically followed by `from z in g` to SelectMany.

### 6.5 `orderby`

```text
orderby e1 [ascending|descending], e2 …
```

Materialize the sequence and sort stably by evaluated keys. Default direction: ascending.

### 6.6 `group expr by key`

Group the current sequence by `key`. Each group element is an ordinary **`Record`** with fields:

- `key` — the grouping key value  
- `items` — `Seq` of the grouped elements (environments or prior projections, as produced by the upstream clause)

Must support `into` and subsequent `select` over those fields.

`key` must be equality-comparable in v1.1 (`Int`, `String`, `Bool`); unsupported key shapes should be rejected during lowering when known statically. If incomparable values appear at runtime, grouping fails with `TypeMismatch`.

### 6.7 `select expr`

Maps each environment to a projected `Value` (`expr`).

**Terminal vs intermediate:**

| Context | Effect |
|---------|--------|
| `select` / `group` is the **last** operation of the query (no `into` continuation) | **Sink**: print the projected sequence to stdout (see §7) |
| Followed by `into id` | **No print**; the projected sequence becomes the input bound to `id` for the continuation body |

### 6.8 `into id` (query continuation)

```text
… select expr into id
  where …
  select …
```

1. Finish the projection as a `Seq` (not a sink).  
2. Bind `id` as the range variable over that sequence for the following `query_body`.  
3. Identifier registration must **define** `id` in scope (legacy bug: deletion on `INTO` — must not recur).

Same continuation idea applies after `group … by … into id` and after `join … into id` (group-join already binds `id` as the group sequence per outer row; naming aligns with C# query semantics).

---

## 7. Terminal output

When a projection is a **sink**:

- Each projected element produces output.  
- For a **single** property / scalar projection: one line per element (e.g. hex digest).  
- For an **anonymous object** with multiple fields, e.g. `{ f.md5, f.sha1 }`: **one line per field**, in field order — so two fields ⇒ **two lines** per input element.

Exact line formatting (prefix names or bare values) should match a chosen golden format in tests; default proposal: **bare values only**, one per line.

---

## 8. Errors

| Class | Examples |
|-------|----------|
| Syntax | Existing grammar failures |
| Semantic (compile) | Undefined range variable; disallowed property for declared type; methods |
| Runtime | Missing file/dir; I/O errors; hash failures; bad regex |

Failed queries should not partially commit confusing sink output beyond what tests specify (prefer fail-fast per query).

---

## 9. Non-goals (v1.1)

Methods · compatibility with the legacy triple IR · built-in recursive directory walk · interpreter micro-optimizations · bytecode / register VM.

- Method-call syntax and dispatch  
- Compatibility with legacy triple opcode stream / C `processor` NULL slots  
- Recursive directory traversal as a built-in  
- Performance tuning of the query interpreter beyond correctness  

---

## 10. Illustrative examples

```text
from file f in '/home/user/file'
where f.size > 0
select f.md5;
```

```text
from file f in '/home/user/file'
where f.md5 == 'd41d8cd98f00b204e9800998ecf8427e'
select f.size;
```

```text
from dir d in '/tmp'
from file f in d
where f.size > 1000
select f.sha1;
```

```text
from string a in 'abc'
join string b in 'abc' on a.md5 equals b.md5
select { a.md5, b.md5 };
```

(Terminal select of two fields ⇒ four lines total if both sides yield one row: two lines for the one record, or per §7: two lines per element — here one joined row → two lines.)

```text
from hash x in 'D41D8CD98F00B204E9800998ECF8427E'
select x.md5;
```

```text
from file f in 'x'
select f.md5 into h
select h;
```

First `select` does not print; second sinks the sequence of digest strings.

---

## 11. Implementation architecture

Pipeline:

```text
source text
  → parse (flex/bison) → AST
  → lower (`lower.zig`) → QueryPlan + static validation
  → interpret (`interpret.zig`)
       ↳ eval Expr against Env (demand-driven props)
       ↳ terminal select/group → sink print; `into` → continuation body
```

Runtime is a **tree-walking** interpreter over `QueryPlan` / `Expr` (not a register/bytecode VM). Query operators and expression evaluation are separate modules. Nested query values lower/execute recursively and yield `Seq(Value)`.

| Clause | Plan shape (see `plan.zig`) |
|--------|-----------------------------|
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
| AST lowering | `lower.zig` |
| LINQ clauses | `from`, `where`, `let`, `join`, `join … into`, `orderby`, `group by`, `select`, `into` |
| Properties | Demand-driven catalog §3.3 |
| Value language | Nested queries in value / where / orderby / from·join sources / join keys; record aliases |
| Static checks | Lowering-time types for properties, join/group keys, records, many sources |
| Diagnostics | `fehler` via `diag.zig` (parse + lowering/runtime spans from AST/`Expr`) |
| Tests | `frontend_test.zig`, `lower_test.zig`, `interpret.zig`; `zig build test-l2h` |
| Methods | **Out of scope** for v1.1 — parse only; lowering → `UnsupportedMethodCall` |
| Recursive dir walk | **Out of scope** for v1.1 — flat listing only (§2.4) |

**Known limitations** (accepted in frozen v1.1; not open design questions): some mixed/`unknown` sequence shapes and I/O failures are detected only at runtime.

Rejected for this stack (kept for history): packed bytecode / register VM; keeping legacy triples; SQL cost-based optimizer.

---

## 12. Resolved decisions (formerly open)

| Topic | Decision |
|-------|----------|
| Record auto-names | `id.prop` → field `prop`; bare `id` → `id`; any other expr in `{…}` → **error** (§5.3) |
| `from file f in d` | Receiver must be **`Dir`** only |
| Symlinks in flat dir listing | **Skip** all symlinks |
| Hex digests | **Print lowercase**; compare case-insensitive |
| `group proj by key` element | Record `{ key, items }` where `items` is the sequence of grouped elements |

No remaining open questions. Further semantic changes follow the amendment policy in the document header.

---

## 13. Freeze record

| Field | Value |
|-------|-------|
| Version | **v1.1** |
| Status | **frozen** |
| Freeze date | 2026-07-29 |
| Implements | Full LINQ surface of this document; nested queries without parentheses in value positions; demand-driven properties; `fehler` diagnostics with source spans |
| Non-goals (v1.1) | Methods; recursive directory walk; bytecode VM |
