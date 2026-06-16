# Cline (VSCode) + Stratum

[Cline](https://github.com/cline/cline) is an autonomous coding agent VSCode
extension. It supports an "OpenAI Compatible" provider, which is exactly the
wire shape `stratum serve --openai` speaks.

## 1. Install Cline

Open VSCode → Extensions → search for "Cline" → install the extension by
*saoudrizwan*. Reload the window if prompted.

## 2. Start the Stratum daemon

```bash
stratum serve --openai --tcp-port 8123 --model qwen-0.5b
```

Leave this running while you use VSCode. `--model` is the default backing
model the virtual router resolves to.

## 3. Point Cline at Stratum

Open Cline's settings panel (gear icon in the Cline sidebar) and set:

| Field | Value |
|---|---|
| API Provider | `OpenAI Compatible` |
| Base URL | `http://127.0.0.1:8123/v1` |
| API Key | `stratum-local` (any non-empty string) |
| Model ID | `stratum/code` (recommended) or an explicit slug like `qwen-0.5b` |

Cline also stores these under VSCode's settings JSON. If you prefer to edit
`settings.json` directly (Cmd/Ctrl-Shift-P → "Preferences: Open User
Settings (JSON)"):

```jsonc
{
  "cline.apiProvider": "openai",
  "cline.openAiBaseUrl": "http://127.0.0.1:8123/v1",
  "cline.openAiApiKey": "stratum-local",
  "cline.openAiModelId": "stratum/code",
  // Alternative: pin a catalog slug
  // "cline.openAiModelId": "qwen-0.5b",

  // Cline's "Use thinking" toggle is fine; Stratum just ignores the
  // unknown OpenAI fields. Streaming is on by default and works.
  "cline.alwaysAllowReadOnly": true
}
```

## 4. Verify

1. Click the Cline icon in the VSCode sidebar to open the chat panel.
2. Send: `list the files in the workspace root`.
3. Cline should fire a `chat.completions` request, you should see the daemon
   log it in the terminal running `stratum serve`, and a textual answer
   appears in the panel.
4. From a terminal, you can cross-check with:

   ```bash
   curl -s http://127.0.0.1:8123/v1/models | jq -r '.data[].id'
   ```

## Known limitations

* **Cline's "auto-approve tools" flow expects function-calling.** Stratum's
  OpenAI surface doesn't yet expose `tools` / `function_call`, so Cline falls
  back to its prompt-driven "ACT mode" — it asks the model to emit
  XML-tagged commands, then parses them. This works, but is more
  failure-prone than native function-calling and benefits from a 7B+ model.
* **MCP servers from inside Cline.** Cline has its own MCP wiring. Don't
  confuse it with Stratum's MCP — they're separate. Either side can host MCP
  servers, just pick one.
* **No image input.** Cline can send screenshots when configured, but its
  current OpenAI-Compatible path doesn't forward them; even if it did,
  Stratum needs an mmproj-equipped model to do anything useful with them.
* **Cancel handling.** Hitting Cline's stop button cancels the SSE stream;
  Stratum's `CancelToken` picks this up within a few hundred ms.

## Troubleshooting

* **"Failed to fetch" / "Network error"** in Cline — base URL typo. It must
  end in `/v1` (no trailing slash), and the host must be reachable from
  VSCode (`127.0.0.1`, not `localhost`, avoids IPv6 quirks on some setups).
* **`401 Unauthorized`** — Stratum doesn't check API keys, so a 401 means
  Cline failed before even hitting Stratum (likely a proxy in your VSCode
  settings). Check `http.proxy` in VSCode and unset it for `127.0.0.1`.
* **Empty answers / model "hallucinates" the wrong workspace** — small local
  models miss Cline's heavy system prompt. Try `stratum models recommend`
  and switch to a 7B-class model.
