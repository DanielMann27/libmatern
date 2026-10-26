#!/bin/bash
# notify.sh <title> <message> -- reports the end of a long benchmark step: prints it, rings the
# bell and shows a desktop notification where possible.  NOTIFY=auto (default) | none | <command>
# (run by sh with BENCH_NOTIFY_TITLE and BENCH_NOTIFY_MESSAGE set).  Always exits 0.
set -u
TITLE="${1:-benchmark}"
MESSAGE="${2:-finished}"
MODE="${NOTIFY:-auto}"

echo "$TITLE: $MESSAGE"
{ printf '\a' > /dev/tty; } 2>/dev/null

case "$MODE" in
    none) exit 0 ;;
    auto) ;;
    *)    BENCH_NOTIFY_TITLE="$TITLE" BENCH_NOTIFY_MESSAGE="$MESSAGE" sh -c "$MODE" >/dev/null 2>&1 &
          exit 0 ;;
esac

# No desktop under a batch job.
[ -n "${SLURM_JOB_ID:-}" ] && exit 0

# WSL: a Windows notification through PowerShell.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    ps=$(command -v powershell.exe 2>/dev/null)
    [ -z "$ps" ] && [ -x /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe ] &&
        ps=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
    if [ -n "$ps" ]; then
        t=${TITLE//\'/\'\'}; m=${MESSAGE//\'/\'\'}
        # A tray balloon (no PowerShell module needed); the process stays up while the
        # balloon shows, so it is detached.
        nohup "$ps" -NoProfile -NonInteractive -Command "
            Add-Type -AssemblyName System.Windows.Forms;
            \$n = New-Object System.Windows.Forms.NotifyIcon;
            \$n.Icon = [System.Drawing.SystemIcons]::Information;
            \$n.Visible = \$true;
            \$n.ShowBalloonTip(15000, '$t', '$m', [System.Windows.Forms.ToolTipIcon]::Info);
            Start-Sleep -Seconds 15; \$n.Dispose()" >/dev/null 2>&1 &
        exit 0
    fi
fi

if { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; } && command -v notify-send >/dev/null 2>&1; then
    notify-send "$TITLE" "$MESSAGE" >/dev/null 2>&1 &
    exit 0
fi

if [ "$(uname -s)" = Darwin ] && command -v osascript >/dev/null 2>&1; then
    t=${TITLE//\"/\\\"}; m=${MESSAGE//\"/\\\"}
    osascript -e "display notification \"$m\" with title \"$t\"" >/dev/null 2>&1 &
fi
exit 0
