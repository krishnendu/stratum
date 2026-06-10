# Stratum Homebrew Tap

This directory holds the **scaffold** Homebrew formula. The published, installable tap lives at:

  https://github.com/krishnendu/homebrew-stratum

## Why is this here?

So the formula version-control lives next to the source. On every release:

1. Cut the tag: `git tag v0.X.Y && git push --tags`.
2. Release workflow (`.github/workflows/release.yml`) produces binaries + a `stable.json` update manifest.
3. The maintainer extracts the per-platform sha256 values, copies them into a new commit on `dist/homebrew/stratum.rb`, and syncs the file to the published tap repo.

## Installing (once published)

```bash
brew tap krishnendu/stratum
brew install stratum
```

## Verifying

```bash
stratum --version
stratum doctor
stratum completions bash | head -5
```

## Updating the formula after a release

```bash
# In this repo, after the v0.X.Y release workflow completes:
gh release download v0.X.Y --pattern '*.sha256' --dir /tmp/stratum-checksums
cat /tmp/stratum-checksums/*aarch64-apple-darwin*.sha256
cat /tmp/stratum-checksums/*x86_64-unknown-linux-gnu*.sha256

# Edit dist/homebrew/stratum.rb:
#   - bump `version`
#   - update each `url` and `sha256` from the per-platform .sha256 files
# Open a PR to land the formula bump, then mirror it to homebrew-stratum.
```

## Per-platform notes

- macOS Apple Silicon (`aarch64-apple-darwin`) — primary supported target.
- Linux x86_64 (`x86_64-unknown-linux-gnu`) — second supported target.
- macOS x86_64, Linux ARM64, and Windows are not in v0.1.0; see `docs/release-process.md`.

## Audit

After every bump:

```bash
brew audit --new --strict dist/homebrew/stratum.rb
brew install --build-from-source dist/homebrew/stratum.rb
```
