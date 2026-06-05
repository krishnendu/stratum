# Stratum machine-readable schemas

JSON Schema (draft-07) definitions for every Stratum CLI surface that emits
`--json`. These are part of Stratum's stable contract: tooling that parses our
output should pin to one of these schemas.

## Index

| Schema | File | First shipped | Status |
|---|---|---|---|
| `stratum doctor --json` (v1) | [`doctor.v1.json`](doctor.v1.json) | Phase 1 | stable |

## Versioning policy

Every schema is versioned independently. The version number is encoded in the
filename (`<surface>.v<N>.json`) **and** in a top-level `schema_version`
integer field of the document the schema validates.

- **Within a major version (v1, v2, …): additive only.** New optional fields,
  new enum members in `additionalProperties: true` containers, new array
  entries — all fine. A v1 consumer must keep working against any future v1
  payload.
- **Removing a field, narrowing a type, or removing an enum value is a major
  bump.** A new file (`doctor.v2.json`) ships alongside the old one; both are
  served until the corresponding CLI surface drops v1.
- **`schema_version` is incremented in lockstep with the file's major.** A v1
  document always has `schema_version: 1`. The CLI prefers the highest version
  the caller asks for via a future `--schema-version` flag (Phase 4+); today
  it always emits v1.

## CI enforcement

Every shipped schema is validated by a dedicated integration test under
`crates/stratum-cli/tests/`. Those tests spawn the live binary, capture the
JSON output, and walk every required field, every enum value, and every regex
pattern from the schema. See
[`docs/verification-gates.md`](../verification-gates.md) §G11.

The CI job `doctor-schema-check` runs `cargo test --package stratum-cli
--test doctor_schema -- --include-ignored` so a schema drift surfaces as a
named, blocking check on the PR.

## Back-compat promise

A `vN` schema entry in this file is removed only when:

1. The CLI surface it describes is itself removed in a major Stratum bump, or
2. A `v(N+1)` of the same surface has been shipped for at least one minor
   release and the deprecation has been called out in the changelog.

Until then, the JSON Schema file is frozen byte-for-byte.
