#!/usr/bin/env bash
# Setup script for git hooks that use ruff for code quality

set -e

HOOK_DIR=".git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

echo "Setting up git pre-commit hook with ruff..."

# Create the pre-commit hook
cat > "$HOOK_FILE" << 'HOOK_EOF'
#!/usr/bin/env bash
# Git pre-commit hook that runs ruff for code quality checks
set -e

echo "Running ruff checks before commit..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh  < /dev/null |  sh"
    exit 1
fi

# Run ruff linting on main package files only
echo "→ Running ruff check..."
if ! uv run ruff check anthropic_common/ claude_cli/ gemini_cli/ openai_cli/ vertex_cli/ 2>/dev/null; then
    echo "❌ Ruff linting failed. Please fix the issues above."
    echo "💡 You can auto-fix many issues with: uv run ruff check --fix ."
    exit 1
fi

# Run ruff formatting check on main package files only
echo "→ Running ruff format check..."
if ! uv run ruff format --check anthropic_common/ claude_cli/ gemini_cli/ openai_cli/ vertex_cli/ 2>/dev/null; then
    echo "❌ Ruff formatting check failed. Please format your code."
    echo "💡 You can auto-format with: uv run ruff format ."
    exit 1
fi

# Run mypy type checking on main package files only
echo "→ Running mypy type check..."
if ! uv run mypy anthropic_common/ claude_cli/ gemini_cli/ openai_cli/ vertex_cli/ 2>/dev/null; then
    echo "❌ Mypy type checking failed. Please fix the type issues above."
    exit 1
fi

echo "✅ All checks passed!"
HOOK_EOF

# Make the hook executable
chmod +x "$HOOK_FILE"

echo "✅ Git pre-commit hook installed successfully!"
echo ""
echo "The hook will now run 'uv run ruff check', 'uv run ruff format --check', and 'uv run mypy'"
echo "before each commit to ensure code quality."
echo ""
echo "To skip the hook for a specific commit, use: git commit --no-verify"
