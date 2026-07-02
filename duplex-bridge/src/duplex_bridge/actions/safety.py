"""CommandSafetyClassifier — allowlist + voice-confirmation policy for shell/AppleScript.

The user-chosen model: allowlisted patterns run autonomously; EVERYTHING else pauses the
task for spoken confirmation; hard destructive patterns always confirm even when an allow
pattern also matches. Compound shell strings (``;``, ``&&``, ``|``, backticks, ``$(``)
are never allowlisted — ``ls; rm -rf ~`` must not ride in on ``^ls``.

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
    r"\b(shutdown|reboot|halt)\b",
    r"\bkillall\b",
    r"\bchmod\b.*777",
    r"\blaunchctl\b",
    r"\bsecurity\b",  # macOS keychain CLI
    r">\s*/dev/",
    r"\bdiskutil\b.*\b(erase|partition)\w*",
    # AppleScript: outward-facing / destructive app control.
    r"tell\s+application\s+\"Mail\".*\bsend\b",
    r"\bdelete\b",
    r"\bempty\s+trash\b",
    r"\bsystem\s+events\b.*\bkeystroke\b",
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
    # AppleScript sources (run_applescript passes the script itself).
    r"^display\s+(notification|dialog)\b",
    r"^open\s+location\b",
    r"^tell\s+application\s+\"(Finder|Safari|Google Chrome|Notes|Calendar|Music)\"\s+to\s+"
    r"(get|count|activate|open|reveal|make new (note|event))\b",
)

# Shell control operators: a command containing these is never allowlisted as a whole.
_COMPOUND = re.compile(r"[;&|`\n]|\$\(")


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
    ) -> None:
        allow = tuple(allow_patterns if allow_patterns is not None else DEFAULT_ALLOW_PATTERNS)
        confirm = tuple(
            confirm_patterns if confirm_patterns is not None else DEFAULT_CONFIRM_PATTERNS
        )
        self._allow = [re.compile(p, re.IGNORECASE) for p in (*allow, *extra_allow)]
        self._confirm = [re.compile(p, re.IGNORECASE) for p in (*confirm, *extra_confirm)]

    def classify(self, command: str) -> SafetyDecision:
        cmd = command.strip()
        if not cmd:
            return SafetyDecision("confirm", "empty command")
        for pattern in self._confirm:
            if pattern.search(cmd):
                return SafetyDecision(
                    "confirm", f"matches a destructive pattern ({pattern.pattern})"
                )
        if _COMPOUND.search(cmd):
            return SafetyDecision(
                "confirm", "compound command — allowlist applies only to simple commands"
            )
        for pattern in self._allow:
            if pattern.search(cmd):
                return SafetyDecision("allow", "allowlisted")
        return SafetyDecision("confirm", "not on the allowlist")
