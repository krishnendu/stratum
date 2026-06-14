# locale: en
# Stratum English message catalog.
#
# One row = one user-visible string the runtime emits. Translators
# clone this file to <locale>.ftl and replace each value. The id
# (left of `=`) MUST NOT change between locales — code references
# the id, the loader resolves to a value per active LocaleId.
#
# Per plan/27-i18n.md. Loaded via i18n::parse_simple_ftl.

# ---- App-level greetings + status ------------------------------------

stratum-greeting = Hi! What are we working on?
stratum-tagline = your local model crew
stratum-exit-armed = press Ctrl+C / Ctrl+D again to exit
stratum-exit-hint = Esc/Ctrl-C exit
stratum-status-ready = ready

# ---- Tool dispatch -----------------------------------------------------

tool-unknown = Unknown tool: { $tool }. Available: { $available }.
tool-missing-args = { $tool } called without required arg(s): { $missing }. Re-issue the call with the missing fields.
tool-permission-denied = Tool { $tool } not allowed by current permission rules.
tool-shell-rejected = shell.exec rejected: { $reason }. Use one of: ls, cat, pwd, head, tail, wc, echo, git.

# ---- Streaming + turn lifecycle ---------------------------------------

turn-thinking = thinking
turn-calling-tool = calling tool
turn-cancelled = (cancelled)
turn-no-output = (no output)
turn-empty-prompt = stratum: --prompt was empty; nothing to send

# ---- Slash-command outcomes -------------------------------------------

cmd-cleared = transcript cleared
cmd-cancel-sent = cancel signal sent
cmd-compact-noop = nothing to compact ({ $count } turns, keeping the most recent { $keep })
cmd-compact-done = compacted { $count } turn(s); kept the most recent { $keep }
cmd-theme-changed = theme: { $name }
cmd-no-models = no models in catalog; run `stratum models sync`

# ---- Memory + auto-memory ---------------------------------------------

memory-view-empty = no memory loaded for this workspace
memory-saved = saved memory: { $name }
memory-forgotten = forgot memory: { $name }
memory-auto-disabled = auto-memory disabled for this workspace

# ---- Errors (paired with STRAT-E#### codes) --------------------------

err-provider-no-text = provider returned no text blocks
err-session-not-found = no transcript for session { $id }
err-bad-session-id = invalid --resume session id: { $err }
err-empty-config = empty configuration value
