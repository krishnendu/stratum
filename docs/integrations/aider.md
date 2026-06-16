# Aider + Stratum

[Aider](https://aider.chat) is a CLI pair-programmer that edits files in your
git repo and commits the diffs. It speaks the OpenAI Chat Completions API, so
you can point it at `stratum serve --openai` and run a fully local edit loop.

## 1. Install Aider

```bash
python -m pip install --user aider-chat
# or:  pipx install aider-chat
aider --version
```

Any Aider release from `0.50` onward is known to work against Stratum's OpenAI
surface.

## 2. Start the Stratum daemon

In one terminal, leave this running:

```bash
stratum serve --openai --tcp-port 8123 --model qwen-0.5b
```

`--model` sets the default backing model the router uses when a request comes
in with a virtual name (`stratum/code`, `stratum/auto`). Swap `qwen-0.5b` for
any installed catalog slug — see `stratum models list`.

## 3. Configure Aider

Aider reads `~/.aider.conf.yml` (global) or `./.aider.conf.yml` (per-repo).
Drop in:

```yaml
# .aider.conf.yml — local Stratum
openai-api-base: http://127.0.0.1:8123/v1
openai-api-key:  stratum-local       # any non-empty string; not validated
model:           openai/stratum/code  # recommended virtual name
# Alternative — pin a specific catalog slug:
# model:         openai/qwen-0.5b
edit-format:     whole                # safer with smaller local models
auto-commits:    false                # opt-in until you trust the diffs
```

The `openai/` prefix tells Aider to use its OpenAI client (rather than
Anthropic, Bedrock, etc.); the segment after the slash is forwarded verbatim as
the `"model"` field in the HTTP request.

You can also pass these as flags, useful for one-off sessions:

```bash
aider --openai-api-base http://127.0.0.1:8123/v1 \
      --openai-api-key  stratum-local \
      --model           openai/stratum/code
```

## 4. Verify

From the repo you want to edit:

```bash
# (1) Sanity-check the daemon is up
curl -s http://127.0.0.1:8123/v1/models | jq -r '.data[].id'

# (2) Start aider with a trivial ask
aider --message "add a top-level README.md saying 'hello stratum'"

# (3) Check the result
git status && git diff
```

If Aider prints a diff, applies it, and exits cleanly, the integration is live.

## Known limitations

* **No tool-calling.** Aider's `--no-stream`/repo-map flows work fine, but
  Aider's optional "use function-calling for edits" mode (`--edit-format diff`
  with function-calling) requires server-side tool support, which the Stratum
  OpenAI surface does not yet expose. Use `--edit-format whole` or the default
  `udiff` text-based formats.
* **Small models miss edits.** With `qwen-0.5b` and similar sub-1B models,
  Aider's diff parser will sometimes reject the model's output. Bump to a
  7B-class model (`stratum models recommend` will suggest one for your tier).
* **Streaming chunks.** Aider streams by default; if you see partial output
  glitches, pass `--no-stream`.
* **No images / multimodal.** Aider doesn't send `image_url` parts today, so
  Stratum's multimodal path is unused here.

## Troubleshooting

* `aider: error: cannot connect to host 127.0.0.1:8123` — the daemon isn't
  running, or it bound to a different port. Re-run step 2 and double-check the
  `--tcp-port` value.
* `model 'stratum/code' not found` in Aider logs — the virtual router is not
  yet enabled on your Stratum build. Switch the `model:` line to an explicit
  catalog slug (e.g. `openai/qwen-0.5b`).
* Aider hangs on the first turn — your local model is loading. The first
  `chat.completions` request triggers backend init (GGUF mmap, mmproj load);
  subsequent turns are fast. Check `stratum events tail` to see progress.
