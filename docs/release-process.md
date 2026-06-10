# Release Process

Stratum tags `v<major>.<minor>.<patch>` produce a GitHub Release with prebuilt binaries + a `stable.json` update manifest.

## Cutting a release

1. Update the version in workspace `Cargo.toml` (`[workspace.package].version`).
2. Bump `CARGO_PKG_VERSION` references in docs / changelog.
3. Commit: `release: v0.1.0`.
4. Tag: `git tag v0.1.0`.
5. Push: `git push origin main v0.1.0`.
6. The `release` workflow fires automatically:
   - Builds `stratum` for `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, and `x86_64-apple-darwin`.
   - Computes per-binary sha256.
   - Generates `stable.json` matching `stratum_runtime::UpdateManifest` shape.
   - Uploads everything to the `v0.1.0` GitHub Release.

## Manual dispatch

`gh workflow run release.yml -f tag=v0.1.0` re-runs against a specific tag.

## Architecture matrix

| Target                        | OS            | Builder        | Notes                                              |
|---|---|---|---|
| `x86_64-unknown-linux-gnu`     | ubuntu-latest | `cargo`        | CPU-only; sandbox via `bwrap`                      |
| `aarch64-unknown-linux-gnu`    | ubuntu-latest | `cargo-zigbuild` | Cross-compiled via zig; CPU-only                 |
| `aarch64-apple-darwin`         | macos-latest  | `cargo`        | Apple Silicon; Metal supported via on-demand build |
| `x86_64-apple-darwin`          | macos-13      | `cargo`        | Intel Mac (macos-13 free tier runner)              |

## Windows

Windows is deferred to a follow-up release (no MSVC toolchain wiring yet, no signing pipeline). Tracking issue to be filed alongside the first `aarch64-pc-windows-msvc` request.

## Linux packaging (deb / rpm)

Native `.deb` and `.rpm` packages are not yet produced by CI. The `dist/deb/` and `dist/rpm/` directories scaffold the configuration (READMEs + `Cargo.toml` snippets) so a maintainer can run `cargo deb` or `cargo generate-rpm` manually against the published `x86_64-unknown-linux-gnu` build. Wiring these into the matrix is tracked as a follow-up.

## Update manifest

`stable.json` matches `stratum_runtime::UpdateManifest`. It's served at the release URL: `https://github.com/krishnendu/stratum/releases/download/v0.1.0/stable.json`. The `stratum self-update --check --manifest-url <url>` command points at this URL.

Future enhancement (multi-binary manifest) tracked in `plan/27-self-update-and-migrations.md`.
