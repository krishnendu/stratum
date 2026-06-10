# Fedora / RHEL packaging (`.rpm`)

Scaffold for Linux `.rpm` packaging via [`cargo-generate-rpm`](https://github.com/cat-in-136/cargo-generate-rpm).

**Status:** not built in CI yet. The release workflow (`.github/workflows/release.yml`) currently publishes tarballs only. A maintainer can produce an `.rpm` manually from any tagged release:

```sh
# One-time
cargo install cargo-generate-rpm

# Per release, from the repo root, against the matching tag
git checkout v0.2.0
cargo build --release --target x86_64-unknown-linux-gnu
cargo generate-rpm --target x86_64-unknown-linux-gnu
# -> target/x86_64-unknown-linux-gnu/generate-rpm/stratum-0.2.0-1.x86_64.rpm
```

For `aarch64-unknown-linux-gnu`, pair with `cargo-zigbuild`:

```sh
cargo install cargo-zigbuild
cargo zigbuild --release --target aarch64-unknown-linux-gnu
cargo generate-rpm --target aarch64-unknown-linux-gnu
```

## Wiring into `Cargo.toml`

`cargo-generate-rpm` reads `[package.metadata.generate-rpm]` from the package's `Cargo.toml`. See `Cargo.toml.snippet.toml` in this directory for a starting template. It is intentionally **not** merged into the root `Cargo.toml` yet — wiring it in is a source-touching change tracked separately.

## Follow-up

- CI matrix entry for `cargo generate-rpm` on the `x86_64-unknown-linux-gnu` runner.
- GPG-signed RPMs via `rpm --addsign` once a signing key lives in repo secrets.
- DNF repo hosting (GitHub Pages + `createrepo_c`) so users can `dnf install stratum` after a one-time `dnf config-manager --add-repo` step.
