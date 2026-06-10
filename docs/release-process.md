# Release Process

Stratum tags `v<major>.<minor>.<patch>` produce a GitHub Release with prebuilt binaries + a `stable.json` update manifest.

## Cutting a release

1. Update the version in workspace `Cargo.toml` (`[workspace.package].version`).
2. Bump `CARGO_PKG_VERSION` references in docs / changelog.
3. Commit: `release: v0.1.0`.
4. Tag: `git tag v0.1.0`.
5. Push: `git push origin main v0.1.0`.
6. The `release` workflow fires automatically:
   - Builds `stratum` for `x86_64-unknown-linux-gnu` + `aarch64-apple-darwin`.
   - Computes per-binary sha256.
   - Generates `stable.json` matching `stratum_runtime::UpdateManifest` shape.
   - Uploads everything to the `v0.1.0` GitHub Release.

## Manual dispatch

`gh workflow run release.yml -f tag=v0.1.0` re-runs against a specific tag.

## Architecture matrix

| Target                      | OS            | Notes                                |
|---|---|---|
| `x86_64-unknown-linux-gnu`   | ubuntu-latest | CPU-only; sandbox via `bwrap`        |
| `aarch64-apple-darwin`       | macos-latest  | Metal supported via on-demand build  |

## Windows + Linux ARM64

Windows + `aarch64-unknown-linux-gnu` are deferred to a follow-up release. macOS x86_64 is also deferred (Apple Silicon runners are free; Intel macs are EOL).

## Update manifest

`stable.json` matches `stratum_runtime::UpdateManifest`. It's served at the release URL: `https://github.com/krishnendu/stratum/releases/download/v0.1.0/stable.json`. The `stratum self-update --check --manifest-url <url>` command points at this URL.

Future enhancement (multi-binary manifest) tracked in `plan/27-self-update-and-migrations.md`.
