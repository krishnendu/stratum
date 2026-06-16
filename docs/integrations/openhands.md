# OpenHands + Stratum

[OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin) is
an open-source autonomous developer agent. Its LLM layer is built on LiteLLM,
which has first-class support for OpenAI-compatible endpoints — so pointing it
at `stratum serve --openai` is a one-config-file change.

## 1. Install OpenHands

The easiest path is the official Docker image:

```bash
docker pull docker.all-hands.dev/all-hands-ai/openhands:latest
```

Or run from source: see the OpenHands repo README for the Python toolchain.

## 2. Start the Stratum daemon

```bash
stratum serve --openai --tcp-port 8123 --model qwen-0.5b
```

**Networking note for Docker users:** when OpenHands runs in Docker and
Stratum runs on the host, `127.0.0.1` inside the container is the container
itself, not your laptop. Either:

* Run the daemon bound to all interfaces and use the host's IP, e.g.
  `stratum serve --openai --tcp-port 8123 --host 0.0.0.0` (if your build
  supports `--host`; otherwise stick to loopback and use `--network host` on
  the container), or
* Start the container with `--network host` (Linux only), or
* Use `host.docker.internal` from inside the container (Docker Desktop on
  macOS / Windows).

The snippets below use `http://host.docker.internal:8123/v1` as the safest
default; replace with `http://127.0.0.1:8123/v1` if you're running OpenHands
without Docker.

## 3. Configure OpenHands

OpenHands reads `config.toml` in its workspace root (or
`~/.openhands/config.toml`). Add an `[llm]` section:

```toml
# config.toml
[llm]
# LiteLLM's "openai/" prefix forces the OpenAI Chat Completions route
# rather than the Anthropic / Bedrock / etc. routes.
model        = "openai/stratum/code"
# Alternative — explicit catalog slug:
# model      = "openai/qwen-0.5b"

base_url     = "http://host.docker.internal:8123/v1"
api_key      = "stratum-local"     # any non-empty string; not validated

# Stratum's OpenAI surface doesn't advertise function-calling yet.
# Tell LiteLLM not to try.
native_tool_calling = false

# Reasonable defaults for local 7B-class models.
temperature  = 0.2
max_input_tokens  = 16000
max_output_tokens = 2048
```

If you launch OpenHands via Docker, mount this file into the container at the
expected path. Example:

```bash
docker run --rm -it \
  -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:latest \
  -v $PWD/config.toml:/app/config.toml \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -p 3000:3000 \
  docker.all-hands.dev/all-hands-ai/openhands:latest
```

## 4. Verify

1. Open [http://localhost:3000](http://localhost:3000) (the OpenHands web UI).
2. Start a new conversation; pick the "Stratum" model if a selector appears,
   otherwise the config-file model is used automatically.
3. Ask: `print the current directory and list its contents`.
4. OpenHands should fire a request to `stratum serve`; you'll see the request
   logged in the daemon's terminal. The agent will then execute `pwd` and
   `ls` in its sandboxed runtime and report back.
5. Cross-check from the host:

   ```bash
   curl -s http://127.0.0.1:8123/v1/models | jq -r '.data[].id'
   curl -s http://127.0.0.1:8123/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"stratum/code","messages":[{"role":"user","content":"hi"}]}' \
     | jq -r '.choices[0].message.content'
   ```

## Known limitations

* **No native function-calling.** OpenHands relies heavily on tool use; with
  `native_tool_calling = false` it switches to a prompt-based fallback. This
  works but is much more sensitive to model size — anything under a 7B-class
  model will struggle. Run `stratum models recommend` to pick a model that
  fits your hardware tier.
* **No vision.** OpenHands' "agent browser" feature sends screenshots as
  `image_url` parts. Stratum forwards these to the backend, but only models
  loaded with an mmproj projector can read them; without one, the image
  parts are silently dropped.
* **Single in-flight turn.** OpenHands occasionally fires speculative
  parallel completions (e.g. when summarising long contexts). Stratum
  serialises these, which adds a bit of latency on long conversations.
* **No `/v1/embeddings`.** If you've configured OpenHands' memory module to
  embed via the same endpoint, it will fail. Point the embeddings provider
  at a separate local server.

## Troubleshooting

* **`litellm.APIConnectionError: Connection refused`** in OpenHands logs —
  the container can't reach the host. Re-read the networking note in step 2;
  the most common fix is `host.docker.internal` instead of `127.0.0.1`.
* **`This model is not currently supported by OpenAI / litellm`** — drop the
  `openai/` prefix and LiteLLM will try to auto-detect, which will fail
  silently. Keep the prefix. Always `openai/<your-model>`.
* **Agent loops on its first step** — small model + prompt-based tool use is
  fragile. Switch to a larger model (`stratum models recommend`).
* **Empty responses** — check the daemon log; the first request triggers
  backend init (GGUF mmap, mmproj load) and can take 5-30s on the first hit.
