# Debian / Ubuntu packaging (`.deb`)

Scaffold for Linux `.deb` packaging via [`cargo-deb`](https://github.com/kornelski/cargo-deb).

**Status:** not built in CI yet. The release workflow (`.github/workflows/release.yml`) currently publishes tarballs only. A maintainer can produce a `.deb` manually from any tagged release:

```sh
# One-time
cargo install cargo-deb

# Per release, from the repo root, against the matching tag
git checkout v0.2.0
cargo deb --target x86_64-unknown-linux-gnu
# -> target/x86_64-unknown-linux-gnu/debian/stratum_0.2.0-1_amd64.deb
```

For `aarch64-unknown-linux-gnu`, pair with `cargo-zigbuild` (same toolchain the release workflow uses for the Linux ARM64 tarball):

```sh
cargo install cargo-zigbuild
cargo zigbuild --release --target aarch64-unknown-linux-gnu
cargo deb --no-build --target aarch64-unknown-linux-gnu
```

## Wiring into `Cargo.toml`

`cargo-deb` reads `[package.metadata.deb]` from the package's `Cargo.toml`. See `Cargo.toml.snippet.toml` in this directory for a starting template. It is intentionally **not** merged into the root `Cargo.toml` yet — wiring it in is a source-touching change tracked separately.

## Follow-up

- CI matrix entry for `cargo deb` on the `x86_64-unknown-linux-gnu` runner.
- Signed `.deb`s via `dpkg-sig` once we have a Debian signing key in repo secrets.
- APT repo hosting (GitHub Pages + `apt-ftparchive`) so users can `apt install stratum` after a one-time source.list entry.
