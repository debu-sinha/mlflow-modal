#!/usr/bin/env bash
set -euo pipefail

# Pre-release validation script for mlflow-modal-deploy
# Run this before creating a release tag

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  Pre-Release Validation"
echo "=========================================="
echo ""

ERRORS=0

# Check version consistency
echo -n "Checking version consistency... "
PYPROJECT_VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
INIT_VERSION=$(grep '__version__' src/mlflow_modal/__init__.py | sed 's/__version__ = "\(.*\)"/\1/')

if [ "$PYPROJECT_VERSION" != "$INIT_VERSION" ]; then
    echo -e "${RED}FAIL${NC}"
    echo "  pyproject.toml: $PYPROJECT_VERSION"
    echo "  __init__.py: $INIT_VERSION"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK${NC} (v$PYPROJECT_VERSION)"
fi

# Check CHANGELOG has entry for current version
echo -n "Checking CHANGELOG.md... "
if grep -q "## \[$PYPROJECT_VERSION\]" CHANGELOG.md; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  No entry for v$PYPROJECT_VERSION in CHANGELOG.md"
    ERRORS=$((ERRORS + 1))
fi

# Check for hardcoded versions in tests
echo -n "Checking for hardcoded versions in tests... "
if grep -r "== \"[0-9]\+\.[0-9]\+\.[0-9]\+\"" tests/ --include="*.py" | grep -v "__version__" | grep -q .; then
    echo -e "${RED}FAIL${NC}"
    echo "  Found hardcoded version strings in tests:"
    grep -r "== \"[0-9]\+\.[0-9]\+\.[0-9]\+\"" tests/ --include="*.py" | grep -v "__version__" | head -5
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK${NC}"
fi

# Run pre-commit
echo -n "Running pre-commit hooks... "
if uv run pre-commit run --all-files > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  Run: uv run pre-commit run --all-files"
    ERRORS=$((ERRORS + 1))
fi

# Run tests
echo -n "Running tests... "
if uv run pytest tests/ -q > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC}"
    echo "  Run: uv run pytest tests/ -v"
    ERRORS=$((ERRORS + 1))
fi

# Check README mentions current Modal version requirement
echo -n "Checking README Modal version... "
README_MODAL=$(grep "Modal" README.md | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+" | head -1 || echo "not found")
PYPROJECT_MODAL=$(grep 'modal>=' pyproject.toml | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+" || echo "not found")
if [ "$README_MODAL" = "$PYPROJECT_MODAL" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${YELLOW}WARN${NC}"
    echo "  README: Modal $README_MODAL"
    echo "  pyproject.toml: Modal $PYPROJECT_MODAL"
fi

# Summary
echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo "Ready to create release tag:"
    echo "  git tag -a v$PYPROJECT_VERSION -m \"Release v$PYPROJECT_VERSION\""
    echo "  git push origin v$PYPROJECT_VERSION"
    exit 0
else
    echo -e "${RED}$ERRORS check(s) failed${NC}"
    echo "Fix issues before releasing."
    exit 1
fi
