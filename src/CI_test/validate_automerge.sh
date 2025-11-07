#!/bin/bash
# Script to validate automerge.yml configuration
# This script checks syntax and configuration, but cannot fully test the automerge functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}\n"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

WORKFLOW_FILE=".github/workflows/automerge.yml"

print_header "Validating automerge.yml"

# Check if file exists
if [ ! -f "$WORKFLOW_FILE" ]; then
    print_error "Workflow file not found: $WORKFLOW_FILE"
    exit 1
fi
print_success "Workflow file exists"

# Check YAML syntax (if yamllint is available)
if command -v yamllint &> /dev/null; then
    print_header "Checking YAML syntax"
    if yamllint "$WORKFLOW_FILE" 2>/dev/null; then
        print_success "YAML syntax is valid"
    else
        print_warning "yamllint found issues (may be false positives)"
    fi
else
    print_warning "yamllint not installed, skipping syntax check"
    echo "  Install with: pip install yamllint"
fi

# Check required fields
print_header "Checking required configuration"

# Check workflow name
if grep -q "name:" "$WORKFLOW_FILE"; then
    print_success "Workflow has a name"
else
    print_error "Workflow missing 'name' field"
fi

# Check permissions
if grep -q "contents: write" "$WORKFLOW_FILE" && grep -q "pull-requests: write" "$WORKFLOW_FILE"; then
    print_success "Required permissions are set"
else
    print_error "Missing required permissions (contents: write or pull-requests: write)"
fi

# Check action version
if grep -q "pascalgn/automerge-action@" "$WORKFLOW_FILE"; then
    VERSION=$(grep -o "pascalgn/automerge-action@[^ ]*" "$WORKFLOW_FILE" | head -1)
    print_success "Using automerge action: $VERSION"
else
    print_error "automerge action not found"
fi

# Check required environment variables
print_header "Checking environment variables"

REQUIRED_VARS=("GITHUB_TOKEN" "MERGE_METHOD" "UPDATE_METHOD" "REQUIRED_LABELS" "REQUIRED_STATUS_CHECKS")
for var in "${REQUIRED_VARS[@]}"; do
    if grep -q "$var" "$WORKFLOW_FILE"; then
        VALUE=$(grep "$var" "$WORKFLOW_FILE" | head -1 | sed 's/.*: *//' | tr -d '"' | tr -d "'")
        print_success "$var = $VALUE"
    else
        print_error "Missing environment variable: $var"
    fi
done

# Check trigger events
print_header "Checking trigger events"

TRIGGER_EVENTS=("pull_request" "pull_request_review" "check_suite")
for event in "${TRIGGER_EVENTS[@]}"; do
    if grep -q "$event:" "$WORKFLOW_FILE"; then
        print_success "Trigger event configured: $event"
    else
        print_warning "Trigger event not found: $event"
    fi
done

# Summary
print_header "Validation Summary"
echo "✓ Basic checks completed"
echo ""
echo "Note: This script only validates configuration, not actual functionality."
echo "To fully test automerge.yml:"
echo "  1. Push to GitHub"
echo "  2. Create a test PR"
echo "  3. Add 'automerge' label"
echo "  4. Ensure all CI checks pass"
echo "  5. Monitor the workflow in GitHub Actions"
echo ""
echo "See AUTOMERGE_TESTING.md for detailed testing guide."

