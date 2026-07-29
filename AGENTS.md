# Rules for Hash Calculator Project

## Project Overview
- **Language**: Zig
- **Purpose**: Console tool that can calculate about 50 cryptographic hashes
- **Build System**: build.zig

## Code Style Guidelines
- Follow Zig standard library conventions
- Use snake_case for functions and variables
- Use PascalCase for types and structs
- Use SCREAMING_SNAKE_CASE for constants
- Prefer explicit error handling with `!` return types
- Keep functions small and focused on single responsibility
- Add doc comments (`///`) for public APIs

## Development Rules

### Before Making Changes
1. Read existing code to understand patterns and conventions
2. Check for existing tests related to modified functionality
3. Ensure changes are compatible with existing API

### When Writing Code
1. Write idiomatic Zig code following std lib patterns
2. Handle all errors explicitly - no silent failures
3. Add tests for new functionality
4. Keep backward compatibility when possible
5. Update documentation for public APIs

### When Fixing Bugs
1. Understand root cause before fixing
2. Add regression test if missing
3. Check for similar issues in related code
4. Verify fix doesn't break existing tests

## Build & Test Commands
```bash
# Build musl
zig build -Dtarget=x86_64-linux-musl --summary new

# Build gnu
zig build -Dtarget=x86_64-linux-gnu --summary new

# Run tests musl
zig build test -Dtarget=x86_64-linux-musl --summary new

# Run tests gnu
zig build test -Dtarget=x86_64-linux-gnu --summary new

```

## Important Notes
- Always verify build passes before completing tasks
- Run full test suite after significant changes
- Follow existing code organization patterns
- Write code comments only in English
- Don't write trivial code comments

## l2h (linq2hash)

Query frontend under `src/l2h/`. Semantics are **frozen v1.1** (source of truth):

- [docs/l2h-semantics.md](docs/l2h-semantics.md)

IR and interpreter live in `plan.zig` / `expr.zig` / `value.zig` / `lower.zig` / `interpret.zig` / `diag.zig`. Prefer the semantics docs over any leftover legacy comments about triples. Behavioral changes need a semantics version bump (see amendment policy in the docs header).
