"""CommandSafetyClassifier — allowlist + voice-confirmation policy for shell/AppleScript.

The user-chosen model: allowlisted patterns run autonomously; EVERYTHING else pauses the
task for spoken confirmation; hard destructive patterns always confirm even when an allow
pattern also matches. Compound shell strings (``;``, ``&&``, ``||``, ``|``, newlines) are
DECOMPOSED and auto-allowed only when every segment is itself allowlisted and none is
destructive — ``ls; rm -rf ~`` still confirms (the second segment isn't allowlisted), but
``curl https://x | head`` no longer pays the same tax as a hidden nested command. Backticks
and ``$(`` are never decomposed — they can hide an arbitrary command from the segment scan,
so they always confirm (live de-nagging fix 2026-07, item 2: the blanket compound-confirm
rule made even harmless pipelines like ``curl … | head`` pause every time).

AppleScript gets its own verdict path (``classify_applescript``): multi-line ``tell``
blocks are the *normal* shape of app scripting, so the shell compound-command rule must
not apply (it made every app action pause — live finding 2026-07-03: a correct
"make new note" script stalled on confirmation, the session dropped, and the task was
orphaned). Pragmatic calibration instead: scripts whose every ``tell application`` block
targets a benign-app allowlist run autonomously; destructive patterns (delete, ``do shell
script``, keystroke automation, discard-without-saving, …) always confirm. A script that is
nothing but ``tell application "X" to activate`` always allows (any app), and a script that
addresses ONLY "System Events" with a read-only query (get/count/exists of processes,
windows, UI elements) also allows — UI automation (click/set) on System Events still
confirms.

This guards the delegate's ``run_shell``/``run_applescript`` handlers (they raise
``ConfirmationRequired`` on a ``confirm`` verdict). It is deliberately independent of
Gemini's built-in computer-use ``safety_decision`` handling, which GeminiComputerUsePolicy
already honors on the vision path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["allow", "confirm"]

# Destructive / irreversible / privilege-escalating: ALWAYS confirm, allowlist or not.
DEFAULT_CONFIRM_PATTERNS: tuple[str, ...] = (
    r"\brm\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bgit\s+push\b.*(--force|-f\b)",
    r"\bgit\s+reset\s+--hard\b",
    r"(curl|wget)\b.*\|\s*(ba|z)?sh\b",
    # curl is allowlisted for read-only GETs only. State-changing methods, request
    # bodies (data exfiltration), and file writes (arbitrary output paths) are caught
    # here so they still confirm — a confirm match is checked before the allowlist and
    # wins. Covers -X POST/PUT/DELETE/PATCH, -d/--data*, -F/--form, -T/--upload-file,
    # -o/--output, -O/--remote-name.
    r"\bcurl\b.*(-X|--request)\s*[\"']?(POST|PUT|DELETE|PATCH)",
    r"\bcurl\b.*\s(-d|--data(-\w+)?|-F|--form|-T|--upload-file|-o|--output|-O|--remote-name)\b",
    r"\b(shutdown|reboot|halt)\b",
    r"\bkillall\b",
    r"\bchmod\b.*777",
    r"\blaunchctl\b",
    r"\bsecurity\b",  # macOS keychain CLI
    r">\s*/dev/",
    r"\bdiskutil\b.*\b(erase|partition)\w*",
    # AppleScript: outward-facing / destructive app control.
    r"tell\s+application\s+\"Mail\"[\s\S]*\bsend\b",
    r"\bdelete\b",
    r"\bempty\s+trash\b",
    r"\bmove\b[^\n]*\bto\s+trash\b",
    r"\bdo\s+shell\s+script\b",  # AppleScript's shell escape hatch — never autonomous
    r"\bkeystroke\b",  # System Events key automation (types into whatever is focused)
    r"\bkey\s+code\b",
    r"\bsaving\s+no\b",  # close/quit discarding unsaved changes
)

# Autonomous without confirmation: read-only or trivially reversible one-shots.
DEFAULT_ALLOW_PATTERNS: tuple[str, ...] = (
    r"^(ls|pwd|whoami|date|uname|cat|head|tail|wc|file|stat|du|df|which|env|printenv)\b",
    r"^echo\b",
    r"^(grep|rg|find|fd|locate)\b",
    r"^git\s+(status|log|diff|show|branch|remote)\b",
    r"^(mkdir|touch)\b",
    r"^open\b",
    r"^say\b",
    r"^defaults\s+read\b",
    r"^osascript\b.*display\s+(notification|dialog)",
    # Read-only fetch (bare GET). NOT unconditionally safe: state-changing methods,
    # request bodies, and file writes are pulled back to confirm by the curl entries in
    # DEFAULT_CONFIRM_PATTERNS (checked first, so they win over this allow). Pipe-to-
    # interpreter (`curl … | sh`) is likewise always-confirm.
    r"^curl\b",
    # Live-fix (2c): screencapture is read-only (writes an image file, no privileged access).
    r"^screencapture\b",
    # AppleScript sources (run_applescript passes the script itself).
    r"^display\s+(notification|dialog)\b",
    r"^open\s+location\b",
    r"^tell\s+application\s+\"(Finder|Safari|Google Chrome|Notes|Calendar|Music)\"\s+to\s+"
    r"(get|count|activate|open|reveal|make new (note|event))\b",
)

# Shell control operators that separate a compound command into independently-classifiable
# segments. "&&"/"||" are matched before the single-char alternatives so they split as one
# operator, not two. Backticks and $() are NOT here — see _HIDDEN_NESTING below.
_SPLIT = re.compile(r"\|\||&&|[;&|\n]")

# Backtick / $() command substitution can hide an arbitrary nested command from the
# segment-by-segment allowlist scan — always confirm, never decomposed.
_HIDDEN_NESTING = re.compile(r"`|\$\(")

# Apps whose non-destructive scripting runs autonomously (classify_applescript). Kept
# deliberately small: local, low-stakes data; nothing that sends, spends, or escalates.
DEFAULT_APPLESCRIPT_APPS: tuple[str, ...] = ("Notes", "Calendar", "Reminders", "TextEdit")

_TELL_APP = re.compile(r"tell\s+application\s+\"([^\"]+)\"", re.IGNORECASE)

# Live-fix (2b): a whole script that is nothing but "tell application "X" to activate" is
# always benign (bringing an app to the foreground), regardless of which app or whether it's
# on DEFAULT_APPLESCRIPT_APPS — but ONLY when that is the script's entire content; a script
# that mixes an activate with other tell blocks still goes through the normal apps check.
_SIMPLE_ACTIVATE = re.compile(r'^tell\s+application\s+"([^"]+)"\s+to\s+activate\s*$', re.IGNORECASE)

# Live-fix (2b): read-only System Events queries (get/count/exists of processes, windows, UI
# elements) run autonomously; UI automation (click/set — keystroke/key code are already
# always-confirm destructive patterns) still confirms. Scoped to scripts that ONLY address
# System Events, so it can never leak trust to another app riding along in the same script.
_SYSTEM_EVENTS_UI_AUTOMATION = re.compile(r"\b(click|set)\b", re.IGNORECASE)
_SYSTEM_EVENTS_READ_ONLY_QUERY = re.compile(r"\b(get|count|exists)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SafetyDecision:
    verdict: Verdict
    reason: str


class CommandSafetyClassifier:
    """classify(command) -> allow (run autonomously) | confirm (pause for spoken approval)."""

    def __init__(
        self,
        allow_patterns: tuple[str, ...] | list[str] | None = None,
        confirm_patterns: tuple[str, ...] | list[str] | None = None,
        *,
        extra_allow: tuple[str, ...] = (),
        extra_confirm: tuple[str, ...] = (),
        applescript_apps: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        allow = tuple(allow_patterns if allow_patterns is not None else DEFAULT_ALLOW_PATTERNS)
        confirm = tuple(
            confirm_patterns if confirm_patterns is not None else DEFAULT_CONFIRM_PATTERNS
        )
        self._allow = [re.compile(p, re.IGNORECASE) for p in (*allow, *extra_allow)]
        self._confirm = [re.compile(p, re.IGNORECASE) for p in (*confirm, *extra_confirm)]
        self._applescript_apps = {
            app.lower()
            for app in (
                applescript_apps if applescript_apps is not None else DEFAULT_APPLESCRIPT_APPS
            )
        }

    def classify(self, command: str) -> SafetyDecision:
        cmd = command.strip()
        if not cmd:
            return SafetyDecision("confirm", "empty command")
        for pattern in self._confirm:
            if pattern.search(cmd):
                return SafetyDecision(
                    "confirm", f"matches a destructive pattern ({pattern.pattern})"
                )
        if _HIDDEN_NESTING.search(cmd):
            return SafetyDecision(
                "confirm",
                "compound command — backtick/$() may hide a nested command from the allowlist scan",
            )
        if _SPLIT.search(cmd):
            segments = [s.strip() for s in _SPLIT.split(cmd) if s.strip()]
            if segments and all(self._segment_is_safe(s) for s in segments):
                return SafetyDecision(
                    "allow",
                    f"compound command — every segment is allowlisted ({len(segments)} segments)",
                )
            return SafetyDecision(
                "confirm",
                "compound command — not every segment is allowlisted and non-destructive",
            )
        for pattern in self._allow:
            if pattern.search(cmd):
                return SafetyDecision("allow", "allowlisted")
        return SafetyDecision("confirm", "not on the allowlist")

    def _segment_is_safe(self, segment: str) -> bool:
        """A decomposed piece of a compound command: allowlisted and not destructive."""
        for pattern in self._confirm:
            if pattern.search(segment):
                return False
        return any(pattern.search(segment) for pattern in self._allow)

    def classify_applescript(self, script: str) -> SafetyDecision:
        """AppleScript verdict: benign-app ``tell`` blocks run; destructive always confirms.

        Multi-line ``tell application … end tell`` is the normal shape of app scripting,
        so the shell compound-command rule does NOT apply here. Instead: destructive
        patterns always confirm; otherwise the script runs autonomously iff every ``tell
        application`` block targets the benign-app allowlist. Scripts with no ``tell``
        block fall back to the plain allow patterns (``display notification`` etc.).
        """
        src = script.strip()
        if not src:
            return SafetyDecision("confirm", "empty script")
        for pattern in self._confirm:
            if pattern.search(src):
                return SafetyDecision(
                    "confirm", f"matches a destructive pattern ({pattern.pattern})"
                )
        simple_activate = _SIMPLE_ACTIVATE.match(src)
        if simple_activate:
            return SafetyDecision("allow", f"simple activate of {simple_activate.group(1)}")
        apps = _TELL_APP.findall(src)
        if len(apps) == 1 and apps[0].lower() == "system events":
            if _SYSTEM_EVENTS_UI_AUTOMATION.search(src):
                return SafetyDecision(
                    "confirm",
                    "System Events UI automation (click/set) requires confirmation",
                )
            if _SYSTEM_EVENTS_READ_ONLY_QUERY.search(src):
                return SafetyDecision("allow", "read-only System Events query (get/count/exists)")
        if apps:
            if all(app.lower() in self._applescript_apps for app in apps):
                return SafetyDecision(
                    "allow", f"benign scripting of allowlisted app(s): {', '.join(apps)}"
                )
            outside = ", ".join(a for a in apps if a.lower() not in self._applescript_apps)
            return SafetyDecision("confirm", f"scripts an app outside the allowlist ({outside})")
        for pattern in self._allow:
            if pattern.search(src):
                return SafetyDecision("allow", "allowlisted")
        return SafetyDecision("confirm", "not on the allowlist")
