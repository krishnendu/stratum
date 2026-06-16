# IDE / agent integrations

Stratum ships an OpenAI-compatible HTTP daemon (`stratum serve --openai`) that
exposes `POST /v1/chat/completions` and `POST /v1/models` on loopback. Any IDE
extension or agent CLI that speaks the OpenAI Chat Completions wire format can
point its base URL at Stratum and get answers from a local model — no API key,
no outbound network, no rate limit.

This directory collects copy-pasteable how-tos for the integrations we've
verified end-to-end.

| Integration | What it is | Doc |
|---|---|---|
| **Aider** | Pair-programming CLI that edits files in a git repo. Drop-in OpenAI client; works against `stratum serve --openai` with a config file. | [`aider.md`](aider.md) |
| **Cline** (VSCode) | Autonomous coding agent VSCode extension. Configure the "OpenAI Compatible" provider to point at `stratum serve --openai`. | [`cline.md`](cline.md) |
| **Continue** (VSCode / JetBrains) | Open-source AI assistant for editors. Uses a JSON config; supports a custom OpenAI-compatible endpoint. | [`continue.md`](continue.md) |
| **OpenHands** | Open-source autonomous developer agent (formerly OpenDevin). Configures an `llm.*` section pointing at the local endpoint. | [`openhands.md`](openhands.md) |

## Shared baseline

Every doc in this directory assumes you've started the daemon:

```bash
stratum serve --openai --tcp-port 8123 --model qwen-0.5b
```

…and then sanity-checked it from another shell:

```bash
curl -s http://127.0.0.1:8123/v1/models | jq
curl -s http://127.0.0.1:8123/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-0.5b","messages":[{"role":"user","content":"say hi"}]}' \
  | jq -r '.choices[0].message.content'
```

If the second command prints a non-empty response, the daemon is healthy and any
of the integrations below should connect.

## Recommended model name

Stratum accepts two flavours of `"model"` in the OpenAI request:

* **Virtual names (recommended for IDE plugins)** — `stratum/code` and
  `stratum/auto`. These are router aliases that pick a sensible backing model
  based on the request shape and your installed catalog, so you don't have to
  rewire your IDE every time you swap models. (When the virtual router is
  unavailable, Stratum falls back to whatever `--model` you passed on the CLI.)
* **Explicit catalog slugs** — e.g. `qwen-0.5b`, `llama-3.1_8b`. Use these when
  you want a specific model and don't want the router to second-guess you. Run
  `stratum models list` to see what's installed.

In the per-integration docs we use `stratum/code` in the config snippets and
note the explicit-slug alternative inline.

## Known limitations (apply to every integration)

* **Tool / function-calling is not yet wired through the OpenAI surface.**
  Stratum exposes its own tools via the JSON-RPC daemon (`stratum serve` without
  `--openai`); the OpenAI-shaped endpoint returns plain assistant text only.
  Integrations that *require* server-side function-calling (some Cline
  workflows, certain OpenHands tasks) will fall back to prompt-driven tool use.
* **Streaming is supported** (`"stream": true` returns SSE `data:` chunks), but
  some clients assume an OpenAI-style `delta.role` arrives in the first chunk —
  Stratum emits it on the first chunk only, matching the 2024 OpenAI shape.
* **No embeddings endpoint.** `/v1/embeddings` is not implemented; integrations
  that need embeddings (RAG-heavy Continue setups) must point those at a
  separate provider.
* **Single concurrent turn.** The daemon serialises requests through one
  `AgentLoop`; parallel completions from the same IDE will queue.
