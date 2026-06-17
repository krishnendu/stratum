# Stratum v1.0.0 — Desktop GA

Stratum reaches **1.0** on desktop. This is the first production release: the
local-LLM TUI agent runs end-to-end, the OpenAI-compatible HTTP egress is
chunked-SSE streaming, voice in/out is wired up via subprocess back-ends, the
Phase 7 evaluation protocol is in tree, and Phase 8 (mobile) groundwork has
landed in the form of a cross-compile audit and a Paths abstraction.

This release covers everything between **v0.2.10 → v1.0.0**.

---

## Highlights

### Phase 5 — Voice (in & out)

- **Whisper subprocess STT** — `whisper.cpp` is launched as a child process
  with a deterministic CLI contract. Audio frames in, JSON transcripts out;
  failures degrade to text-only mode rather than crashing the TUI.
- **Piper TTS subprocess** — mirror of the whisper plumbing. The TUI streams
  the assistant's reply line-by-line into `piper`, which renders WAV to the
  default audio sink.
- **cpal microphone capture** — push-to-talk (PTT) capture using `cpal` on
  macOS (CoreAudio) and Linux (ALSA / PulseAudio). PTT key is configurable;
  default is `ctrl+space`.
- **`/audio` palette command** — opens a panel showing live mic level,
  whisper STT status, piper TTS status, and the current PTT binding.
- **ReadAudioToolDispatcher** — agent-side tool that can ingest a `.wav` /
  `.mp3` / `.flac` attachment and transcribe it via the same whisper
  subprocess, so audio drops into the TUI work like text drops.
- **Transcript line in TUI** — voice-in turns are rendered with a
  microphone glyph; voice-out turns show a speaker glyph. The transcript
  pane is the single source of truth across text + voice.
- **5-turn voice-driven dialogue test** — the Phase 5 exit criterion. A
  synthetic dialogue (recorded wav → whisper → agent → piper → wav out) is
  driven through the runtime in CI to lock the voice loop down.

### Phase 6 — OpenAI-compatible HTTP egress

- **Real chunked-transfer SSE streaming on `/v1/chat/completions`** — no
  more buffered fake-streaming. Each backend token chunk is flushed as a
  `data: {...}\n\n` SSE frame with the matching `id`, `created`, and
  `model`. The handler honours client disconnects to cancel inference.
- **`stratum/<role>` virtual model name resolution** — clients can ask for
  `stratum/coder`, `stratum/judge`, `stratum/default`, etc., and the
  resolver maps the virtual name to the active backend + model on the
  serving host. The actual underlying model is reported back in
  `response.model` for transparency.
- **Memory-probe 503** — before binding the backend on a request, the
  handler runs a free-RAM probe against the configured model footprint.
  If we'd OOM we return `503 Service Unavailable` with a `Retry-After` and
  an explanatory JSON body, rather than letting the host swap to death.
- **Unified SIGINT/SIGTERM shutdown** — both the agent serve path and the
  OpenAI serve path now drain in-flight requests on signal and exit
  cleanly. No more orphaned child processes.
- **IDE integration templates** — `docs/integrations/` ships ready-to-paste
  recipes for **Aider**, **Cline**, **Continue**, and **OpenHands**
  pointing at a local `stratum serve --openai` endpoint with the virtual
  model names. Each integration has a smoke-test command.
- **Multimodal content array on `/v1/chat/completions`** — accepts the
  OpenAI `content: [{type:"text",...},{type:"image_url",...}]` shape and
  forwards image attachments to the agent via `BackendRequest`.

### Phase 7 — Evaluation protocol

- **Comparison protocol + runner skeleton** — `stratum-eval` defines a
  reproducible head-to-head protocol: a fixed task suite, deterministic
  prompts, side-by-side execution across N model configs, and structured
  output (JSON + Markdown) for the LLM judge. The runner skeleton handles
  config discovery, parallel execution, result collation, and crash
  isolation per task.
- **Bench harness, bench-floor, bench-history, nightly workflow** —
  landed earlier in the v0.2.10 cycle and now exercised by the comparison
  runner.

### Phase 8 — Mobile readiness

- **Cross-compile audit** (`docs/phase-8-readiness.md`) — exhaustive sweep
  of every workspace crate against `aarch64-linux-android`,
  `aarch64-apple-ios`, and `x86_64-linux-android`. Documents which deps
  block on-device builds (notably: `cpal` audio path on iOS, `whisper.cpp`
  build script on Android NDK) and lists the proposed feature-flag
  partition.
- **Paths abstraction** — the runtime now consults a single `Paths`
  service for cache, config, model, transcript, and runtime-state
  directories. Desktop implementation is unchanged behaviourally; the
  trait carves out the seam mobile back-ends need.

### Stability, coverage, polish

- Coverage backfill across whisper plumbing, OpenAI HTTP handler, and
  serve middleware.
- SHA-pinned external dependencies in homebrew formulas; audio polish
  (rate-mismatch handling); serve hardening on idle close.
- Workspace lint floor unchanged: `unwrap_used`, `expect_used`, `panic`,
  `print_stdout`, `print_stderr`, `unsafe_code` denied outside tests with
  reasoned `#[allow]` escape hatches.

---

## Installation

### Homebrew (recommended on macOS and Linuxbrew)

```sh
brew tap krishnendu/stratum https://github.com/krishnendu/stratum
brew install stratum
# optional: GPU-accelerated llama.cpp backend formula
brew install stratum-llama-cpp
```

The formulas are pinned per release; after this tag the maintainer will
publish v1.0.0 formulas pointing at the v1.0.0 release artifacts.

### From source

Requires Rust 1.90+.

```sh
git clone https://github.com/krishnendu/stratum
cd stratum
cargo install --path crates/stratum-cli --locked
stratum --version   # → stratum 1.0.0
```

For voice support (push-to-talk mic capture + TTS playback), build with
the `voice` feature:

```sh
cargo install --path crates/stratum-cli --locked --features voice
```

On Linux you also need ALSA development headers (`sudo apt install
libasound2-dev` on Debian/Ubuntu, `sudo dnf install alsa-lib-devel` on
Fedora, etc.) so `cpal` can build.

For voice transcription + synthesis, install `whisper.cpp` and `piper`
on `$PATH`; Stratum discovers them at startup and surfaces status in
`/audio`.

### Prebuilt tarballs

Downloaded from the v1.0.0 GitHub release page. Pick the tarball matching
your host:

| Host                          | Tarball                                              |
| ----------------------------- | ---------------------------------------------------- |
| Apple Silicon (M1/M2/M3/M4)   | `stratum-v1.0.0-aarch64-apple-darwin.tar.gz`         |
| Intel macOS                   | `stratum-v1.0.0-x86_64-apple-darwin.tar.gz`          |
| Linux arm64 (glibc)           | `stratum-v1.0.0-aarch64-unknown-linux-gnu.tar.gz`    |
| Linux x86_64 (glibc)          | `stratum-v1.0.0-x86_64-unknown-linux-gnu.tar.gz`     |

```sh
curl -L -o stratum.tar.gz \
  https://github.com/krishnendu/stratum/releases/download/v1.0.0/stratum-v1.0.0-aarch64-apple-darwin.tar.gz
tar -xzf stratum.tar.gz
install -m 755 stratum /usr/local/bin/stratum
stratum --version
```

Each tarball ships with a detached SHA-256 sum file; verify before
installing.

**Voice in/out (cpal mic capture + rodio TTS playback) is enabled in
the macOS tarballs.** The Linux tarballs are built without the `voice`
feature so they do not require ALSA headers at build time and do not
link `libasound2` at runtime — the prebuilt binary runs on minimal
distros. Linux users who want voice support compile from source with
`--features voice` after installing ALSA dev headers (see the "From
source" section above).

---

## Known limitations

- **mtmd (multimodal vision) deferred.** The vision path
  (`llama.cpp`-mtmd / clip / llava) is intentionally not wired up in 1.0.
  The OpenAI HTTP handler accepts `image_url` content blocks at the wire
  level and the agent receives attachments, but no backend in this release
  actually runs vision inference; image content is currently echoed as
  alt-text. Vision lands in a follow-up minor release.
- **Mobile back-ends not shipped.** Phase 8 work in 1.0 is purely a
  readiness audit + Paths seam. iOS and Android binaries are not produced
  by this release.
- **`stratum self-update` is desktop-only** and only writes channels
  `stable` / `beta` / `nightly` for the host triple it was installed on.
- **Voice subprocess discovery is `$PATH`-only.** No bundled binaries on
  desktop. Homebrew users get `whisper.cpp` and `piper` via their own
  formulas.

---

## Upgrading from v0.2.x

The settings file schema is unchanged from v0.2.10. `stratum` will read
existing settings transparently. If you used the OpenAI HTTP egress on
v0.2.x with `model: "stratum"`, that string still resolves to the default
role; clients that hard-coded a specific model name should switch to the
`stratum/<role>` form.

---

## Acknowledgements

Thanks to everyone who shipped Phase 5/6/7 and the Phase 8 readiness work.
