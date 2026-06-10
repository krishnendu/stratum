# Running `stratum serve` as a system service

`stratum serve` runs in the foreground by default. To make it a background daemon, install it via systemd (Linux) or launchd (macOS).

## Linux (systemd, user unit)

Per-user unit. No root required.

```bash
# 1. Install the unit
mkdir -p ~/.config/systemd/user
cp dist/systemd/stratum.service ~/.config/systemd/user/

# 2. Enable + start
systemctl --user daemon-reload
systemctl --user enable --now stratum.service

# 3. Verify
systemctl --user status stratum.service
journalctl --user -u stratum.service -f
```

To uninstall:

```bash
systemctl --user disable --now stratum.service
rm ~/.config/systemd/user/stratum.service
systemctl --user daemon-reload
```

## Linux (systemd, system-wide)

Same file, different path. Run as root or with `sudo`.

```bash
sudo cp dist/systemd/stratum.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stratum.service
```

## macOS (launchd, user agent)

Loads at login. No root.

```bash
# 1. Install
cp dist/launchd/dev.stratum.serve.plist ~/Library/LaunchAgents/

# 2. Load
launchctl load ~/Library/LaunchAgents/dev.stratum.serve.plist

# 3. Verify
launchctl list | grep stratum
tail -f /tmp/stratum-serve.err.log
```

To uninstall:

```bash
launchctl unload ~/Library/LaunchAgents/dev.stratum.serve.plist
rm ~/Library/LaunchAgents/dev.stratum.serve.plist
```

## Verifying the daemon

Once running, hit the daemon from any shell:

```bash
stratum client --method ping --tcp 127.0.0.1:47474 --json
stratum client --method health --tcp 127.0.0.1:47474 --json
```

## Port choice

Default `47474` is unregistered in the IANA Service Name and Transport Protocol Port Number Registry; pick a different port via the `ExecStart` line / `ProgramArguments` array if it clashes with something else on your machine.

## Resource limits (Linux only)

The systemd unit sets sandbox hardening: `NoNewPrivileges`, `PrivateTmp`, `PrivateDevices`, `ProtectSystem=strict`, `ProtectHome=read-only`. To allow Stratum to read `~/.config/stratum` and write `~/.local/state/stratum`, the `ReadWritePaths` line lists them explicitly.

## launchd notes

`KeepAlive` with `SuccessfulExit=false` means launchd restarts the daemon if it exits non-zero but lets clean exits (e.g., `stratum serve --stop-after-ms`) terminate normally.
