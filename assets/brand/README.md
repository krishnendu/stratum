# Brand assets

Canonical visual identity for Stratum. Wired by `plan/44-brand-and-identity.md`. Logo usage rules live in `plan/24-trademark-policy.md`.

## Files

| File | Use |
|---|---|
| `stratum.svg` | Primary lockup — mark + wordmark + tagline. Use in READMEs, docs sites, social cards. |
| `stratum-mark.svg` | Mark only (the four layered bars). Favicon, app icon, social avatars. Brand-primary fill. |
| `stratum-wordmark.svg` | Mark + wordmark, no tagline. Inline doc references. |
| `stratum-mono.svg` | Single-color variant; uses `currentColor` so it picks up the surrounding text color. For terminals, dark backgrounds, e-ink. |

## Colors

| Token | Hex | Where it ships in code |
|---|---|---|
| Brand primary | `#1E5E5E` (warm slate teal) | `stratum-cli::brand::COLOR_PRIMARY` |
| Brand accent | `#D9844D` (warm amber) | `stratum-cli::brand::COLOR_ACCENT` |
| Error | `#C2384A` (muted brick) | `stratum-cli::brand::COLOR_ERROR` |
| Warning | `#C29A3A` (muted gold) | `stratum-cli::brand::COLOR_WARN` |
| Success | `#3A8A4A` (muted green) | `stratum-cli::brand::COLOR_SUCCESS` |

All colors meet WCAG AA contrast against both `#0E0E12` (dark term) and `#F4F1EA` (light term).

## Generating raster + Mac icons

```
# .icns (Mac app bundle) — manual until xtask brand lands
sips -z 1024 1024 stratum-mark.svg --out stratum-1024.png
iconutil -c icns iconset/
```

`xtask brand --check` (Phase 5) validates that the README + Cargo.toml + Homebrew formula reference the canonical tagline.
