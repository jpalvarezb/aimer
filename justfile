# Aimer workspace task runner. Run `just` to list recipes.

# Show available recipes.
default:
    @just --list

# Install/sync the workspace (dev group included).
install:
    uv sync

# Run the full test suite (benchmarks excluded by default config).
test:
    uv run pytest

# Lint + format-check.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Static type-check the workspace packages.
type-check:
    uv run mypy aimer-core/src pointer-agent/src duplex-bridge/src

# Run the duplex bridge. Pass extra args, e.g. `just run-bridge --audio-backend native-vpio`.
run-bridge *ARGS:
    uv run -m duplex_bridge {{ARGS}}

# Build the native macOS VPIO helper (requires Swift/Xcode; macOS only).
build-native:
    cd native && swift build -c release

# Measure end-of-speech -> first-audio latency (needs GEMINI_API_KEY).
measure-latency *ARGS:
    uv run python scripts/bench/measure_ttfb.py --manual-vad --thinking-level minimal {{ARGS}}
