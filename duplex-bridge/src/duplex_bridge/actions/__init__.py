"""Week-7 host-app actions — the actuators a tool call drives (Chrome, IDE).

These are the "host app actions" half of the loop: a deictic utterance resolves to a tool call
(the model decides *what*), and these handlers carry it out on the host (the *how*). They run on
the Week-6 :class:`~duplex_bridge.worker.BackgroundWorker`, off the audio hot path.

- :func:`rewrite_function_async` — IDE: rewrite a function to async in a real file on disk.
- :func:`compare_products` — Chrome: open a side-by-side comparison of the pointed-at products.

``TOOL_DECLARATIONS`` are provider-neutral function schemas; ``GeminiLiveSession`` converts them
to ``types.Tool`` so the model can emit these calls live.
"""

from __future__ import annotations

from typing import Any

from .browser import BROWSER_TOOL_SPECS, DelegateBrowser
from .chrome import ComparisonResult, compare_products
from .computer import (
    Action,
    Computer,
    ComputerUseExecutor,
    ComputerUseResult,
    FakeComputer,
    MacOSComputer,
    Policy,
    run_computer_use,
)
from .computer_policy import GeminiComputerUsePolicy
from .delegate import (
    ConfirmationRequired,
    DelegateAgent,
    DelegateAgentConfig,
    DelegateResult,
    PendingConfirmation,
)
from .ide import RewriteResult, rewrite_function_async
from .safety import CommandSafetyClassifier, SafetyDecision
from .tasks import DelegateTask, TaskManager

# Provider-neutral tool/function declarations (JSON-schema-ish). The bridge converts these into
# the model provider's tool format so the duplex model can emit the corresponding tool calls.
TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "rewrite_function_async",
        "description": (
            "Rewrite the function the user is pointing at so it is asynchronous (async def, "
            "awaiting blocking calls). Use when the user says e.g. 'rewrite this function async'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to the source file."},
                "function": {"type": "string", "description": "Name of the function to rewrite."},
                "new_source": {
                    "type": "string",
                    "description": "Optional full async rewrite; omit to apply a transform.",
                },
            },
            "required": ["file", "function"],
        },
    },
    {
        "name": "compare_products",
        "description": (
            "Open a side-by-side comparison of two or more products the user is pointing at. "
            "Use when the user says e.g. 'compare these products'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "products": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Product names to compare.",
                }
            },
            "required": ["products"],
        },
    },
    {
        "name": "delegate_task",
        "description": (
            "Delegate a multi-step task on the user's computer to the action agent, which can "
            "run shell commands, AppleScript, a web browser, and full desktop control. Use for "
            "ANY doing-task: 'reply to this email', 'rename those files', 'order this again'. "
            "You will get a started ack immediately — tell the user you're on it and keep "
            "conversing; the outcome arrives later. Ground 'this/that' from the [context] "
            "pointer annotations when phrasing the goal. Multiple tasks may run at once."
        ),
        "behavior": "NON_BLOCKING",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What to accomplish, in natural language, self-contained.",
                }
            },
            "required": ["goal"],
        },
    },
    {
        "name": "check_tasks",
        "description": (
            "Report the status of delegated tasks. Use when the user asks how a task is going "
            "or what's still running."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "confirm_task",
        "description": (
            "Resume a delegated task that is awaiting the user's confirmation, passing their "
            "spoken decision. Call ONLY after the user has clearly approved or declined."
        ),
        "behavior": "NON_BLOCKING",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "The task awaiting confirmation."},
                "approved": {"type": "boolean", "description": "True if the user said yes."},
            },
            "required": ["task_id", "approved"],
        },
    },
    {
        "name": "computer_use",
        "description": (
            "Carry out an arbitrary cross-application action the user describes, by driving the "
            "desktop with screenshot + mouse + keyboard. Use for ANY host action not covered by a "
            "more specific tool (e.g. 'reply to this email', 'add this to my cart', 'fix the "
            "import'). The specific tools are fast paths; this is the general fallback."
        ),
        # Long-running: the model gets a silent "started" ack and keeps conversing; the
        # outcome arrives later as a WHEN_IDLE FunctionResponse (probe-validated).
        "behavior": "NON_BLOCKING",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What to accomplish, in natural language.",
                }
            },
            "required": ["goal"],
        },
    },
]

__all__ = [
    "BROWSER_TOOL_SPECS",
    "TOOL_DECLARATIONS",
    "Action",
    "CommandSafetyClassifier",
    "ComparisonResult",
    "Computer",
    "ConfirmationRequired",
    "SafetyDecision",
    "DelegateAgent",
    "DelegateAgentConfig",
    "DelegateBrowser",
    "DelegateResult",
    "DelegateTask",
    "PendingConfirmation",
    "TaskManager",
    "ComputerUseExecutor",
    "ComputerUseResult",
    "FakeComputer",
    "GeminiComputerUsePolicy",
    "MacOSComputer",
    "Policy",
    "RewriteResult",
    "compare_products",
    "rewrite_function_async",
    "run_computer_use",
]
