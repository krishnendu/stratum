# Continue (VSCode / JetBrains) + Stratum

[Continue](https://continue.dev) is an open-source AI assistant for VSCode and
JetBrains. Its `openai` provider type accepts any OpenAI-compatible base URL,
so it points at `stratum serve --openai` cleanly.

## 1. Install Continue

VSCode: Extensions → search "Continue" → install the extension by *Continue*.
JetBrains: Settings → Plugins → search "Continue" → install.

## 2. Start the Stratum daemon

```bash
stratum serve --openai --tcp-port 8123 --model qwen-0.5b
```

## 3. Edit `.continue/config.json`

Continue stores its config in `~/.continue/config.json`. Open it (Cmd/Ctrl-,
in Continue, or via Cmd/Ctrl-Shift-P → "Continue: Open config.json") and add
Stratum to the `models` array:

```jsonc
{
  "models": [
    {
      "title": "Stratum (local)",
      "provider": "openai",
      "model": "stratum/code",
      "apiBase": "http://127.0.0.1:8123/v1",
      "apiKey": "stratum-local",
      "completionOptions": {
        "temperature": 0.2,
        "maxTokens": 1024
      }
    }
    // Alternative — explicit catalog slug:
    // { "title": "Qwen 0.5B", "provider": "openai",
    //   "model": "qwen-0.5b", "apiBase": "http://127.0.0.1:8123/v1",
    //   "apiKey": "stratum-local" }
  ],

  // Optional: use Stratum as the autocomplete / tab-completion model too.
  // Only do this if you've got a small, fast model loaded; 7B will lag.
  "tabAutocompleteModel": {
    "title": "Stratum tabs",
    "provider": "openai",
    "model": "stratum/code",
    "apiBase": "http://127.0.0.1:8123/v1",
    "apiKey": "stratum-local"
  }
}
```

Save and reload Continue's sidebar (the dropdown at the top now lists
"Stratum (local)").

## 4. Verify

1. Open the Continue panel; select **Stratum (local)** in the model dropdown.
2. Type into the chat: `summarize this file` (with any file open).
3. You should see a streamed response in the panel and an HTTP request hit
   the running `stratum serve` process.
4. From a terminal cross-check:

   ```bash
   curl -s http://127.0.0.1:8123/v1/models | jq -r '.data[].id'
   ```

## Known limitations

* **No embeddings endpoint.** If you've wired Continue's RAG / `@codebase`
  context using Stratum, it will fall back to lexical search. Configure a
  separate `embeddingsProvider` (e.g. a local Ollama embeddings model) if
  you need vector retrieval. Continue silently degrades to lexical otherwise.
* **No function-calling.** Continue's "Tools" feature (the wrench in the
  chat input) calls function-calling internally; with Stratum it falls back
  to prompt-driven tool use. This works for simple cases but is less
  reliable than native function-calling on GPT-4-class servers.
* **Tab-autocomplete latency.** Continue fires autocomplete on every
  pause-in-typing. Even a 7B model running through `stratum serve` will be
  too slow for snappy ghost-text. Either omit `tabAutocompleteModel` or
  point it at a sub-1B model.
* **Streaming is on by default** and works; if you see split words, set
  `"completionOptions": { "stream": false }` for that model.

## Troubleshooting

* **"openai: server had an error processing your request"** with no detail —
  check the `stratum serve` terminal. Most often the model isn't loaded yet
  (first request triggers backend init; subsequent ones are fast).
* **Model dropdown is empty** — Continue couldn't parse your config.
  `~/.continue/config.json` must be valid JSON; comments (`//`) are tolerated
  by Continue's parser but not by strict JSON, so an editor flagging them is
  fine.
* **Long responses get cut off** — bump `completionOptions.maxTokens`. The
  Stratum-side default is generous, but Continue truncates at its own limit.
* **Wrong model answers** — Continue caches per-session context; switch
  models in the dropdown, then click "New session".
