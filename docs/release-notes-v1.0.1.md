# Stratum v1.0.1 — UX bandages on top of v1.0.0

Patch release. Fixes three real ship-blockers caught on the v1.0.0 GA:

1. **`stratum self-update --apply` corrupted the binary** on macOS hosts.
   BSD `tar -czf` bundles a 163-byte `AppleDouble` metadata sidecar
   (`._stratum`) BEFORE the real `stratum` entry in the archive. The
   self-update extractor took the first regular file unconditionally, so
   the metadata blob was written to disk as the new exe. Next invocation:
   `zsh: exec format error: stratum`.

   Fixed in two places:
   - **Defensive**: extractor now skips `._*` entries and continues
     scanning for the real binary. Future-proofs against any tool that
     emits AppleDouble sidecars.
   - **At the source**: the v1.0.0 GA tarballs were rebuilt with
     `COPYFILE_DISABLE=1 tar -czf` (suppresses the sidecar) and
     re-uploaded to the GH release; `stable.json` sha256s refreshed
     accordingly.

2. **`stratum chat` (no `--model`) silently ran EchoProvider** without
   any signal that the response was a stub. The prebuilt v1.0.0
   tarballs ship without the `provider-llama-cpp` feature (cold-build
   cost + ~50 MB binary), so a new user typing `stratum chat` had no
   way to discover the limitation or the install path that fixes it.

   Fixed: new one-shot banner at chat entry that points at the two
   install paths (Homebrew `stratum-llama-cpp` formula, or
   `cargo install --features provider-llama-cpp,voice`). Suppressed
   under `--json` and `STRATUM_QUIET_BANNERS=1` for scripted callers.

3. **`cargo install --features provider-llama-cpp` did not compile.**
   `LlamaCppProviderConfig` grew an `mmproj_path` field for the Phase
   5 vision wiring but two call sites in `app.rs` were never updated.
   The feature is off in per-PR CI so the regression slipped through.
   Without this fix, the v1.0.1 banner's own install hint would lead
   users to a broken cargo command.

   Both call sites now set `mmproj_path: None` with a TODO comment
   pointing at the upstream `mtmd` ABI seam (issue #172) where the
   real wiring lands.

## What did NOT change

- No new features. No protocol changes. No catalog changes.
- Identical voice / OpenAI egress / Phase 5/6/7 surface as v1.0.0.
- `stable.json` history retains the v1.0.0 entry; v1.0.1 is appended.

## Upgrade

```sh
stratum self-update --apply
```

…or the equivalent `brew upgrade stratum` / `brew reinstall
stratum-llama-cpp` depending on which formula is on your `PATH`.

The v1.0.0 binary's self-update extractor is the buggy one, but the
v1.0.0 GA tarballs are now clean (no `._*` sidecars), so the v1.0.1
upgrade arrives intact.

For users coming from v0.2.x, see `docs/release-notes-v1.0.0.md` for
the full v1.0.0 changelog — v1.0.1 is purely additive on top of it.
