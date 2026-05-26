#!/usr/bin/env bash
set -euo pipefail

# Prepare changelog content - prefer resolved version if available
if [ -n "${RESOLVED_CHANGELOG_B64:-}" ]; then
  echo "$RESOLVED_CHANGELOG_B64" | base64 -d > decoded_changelog.md
  CHANGELOG_CONTENT=$(cat decoded_changelog.md)
elif [ "${HAS_CHANGELOG:-}" == "true" ]; then
  echo "$CHANGELOG_B64" | base64 -d > decoded_changelog.md
  CHANGELOG_CONTENT=$(cat decoded_changelog.md)
else
  CHANGELOG_CONTENT="No significant changes detected in this release."
fi

# Prepare the full release notes in a temporary file
if [ -n "${PREVIOUS_TAG:-}" ]; then
  CHANGELOG_URL="**Full Changelog**: https://github.com/$GITHUB_REPOSITORY/compare/$PREVIOUS_TAG...$RELEASE_TAG"
else
  CHANGELOG_URL=""
fi

# Generate file descriptions table from FILE_DESCRIPTIONS env var
FILE_TABLE="| File | Description |
|------|-------------|"
while IFS='|' read -r filename description; do
  # Skip empty lines
  if [ -n "$filename" ] && [ -n "$description" ]; then
    FILE_TABLE="$FILE_TABLE
| \`$filename\` | $description |"
  fi
done <<< "$FILE_DESCRIPTIONS"

# List archives (add || true to prevent grep exit-on-error)
WINDOWS_ARCHIVE=$(echo "${ASSET_PATHS:-}" | tr ' ' '\n' | grep 'Windows' || true)
LINUX_ARCHIVE=$(echo "${ASSET_PATHS:-}" | tr ' ' '\n' | grep 'Linux' || true)
MACOS_ARCHIVE=$(echo "${ASSET_PATHS:-}" | tr ' ' '\n' | grep 'macOS' || true)
ARCHIVE_LIST="- **Windows**: \`$WINDOWS_ARCHIVE\`
- **Linux**: \`$LINUX_ARCHIVE\`
- **macOS**: \`$MACOS_ARCHIVE\`"

# Build upstream reference line (empty if no upstream remote)
UPSTREAM_REF_LINE=""
if [ -n "${UPSTREAM_REF:-}" ]; then
  UPSTREAM_REF_SHORT=$(echo "$UPSTREAM_REF" | cut -c1-7)
  UPSTREAM_REF_LINE="> Based on upstream \`dev\` @ [\`$UPSTREAM_REF_SHORT\`](https://github.com/Mirrowel/LLM-API-Key-Proxy/commit/$UPSTREAM_REF)"
fi

cat > releasenotes.md <<EOF
## Build Information
| Field | Value |
|-------|-------|
| 📦 **Version** | \`$RELEASE_VERSION\` |
| 💾 **Binary Size** | Win: \`$WIN_SIZE\`, Linux: \`$LINUX_SIZE\`, macOS: \`$MACOS_SIZE\` |
| 🔗 **Commit** | [\`$SHORT_SHA\`](https://github.com/$GITHUB_REPOSITORY/commit/$GITHUB_SHA) |
| 📅 **Build Date** | \`$BUILD_DATE\` |
| ⚡ **Trigger** | \`$EVENT_NAME\` |

## 📋 What's Changed

$UPSTREAM_REF_LINE
$CHANGELOG_CONTENT

### 📁 Included Files
Each OS-specific archive contains the following files:
$FILE_TABLE

### 📦 Archives
$ARCHIVE_LIST

## 🔗 Useful Links
- 📖 [Documentation](https://github.com/$GITHUB_REPOSITORY/wiki)
- 🐛 [Report Issues](https://github.com/$GITHUB_REPOSITORY/issues)
- 💬 [Discussions](https://github.com/$GITHUB_REPOSITORY/discussions)
- 🌟 [Star this repo](https://github.com/$GITHUB_REPOSITORY) if you find it useful!

---

> **Note**: This is an automated build release.

$CHANGELOG_URL
<!-- build_tree: $BUILD_TREE -->
<!-- upstream_base: $UPSTREAM_REF -->
EOF

# Set release flags and notes based on the branch
CURRENT_BRANCH="$GITHUB_REF_NAME"
PRERELEASE_FLAG=""
LATEST_FLAG="--latest"
EXPERIMENTAL_NOTE=""

# Check if the current branch is in the stable branches list
if ! [[ ",${STABLE_BRANCHES:-}," == *",$CURRENT_BRANCH,"* ]]; then
  PRERELEASE_FLAG="--prerelease"
  LATEST_FLAG="" # Do not mark non-stable branches as 'latest'
  
  # Generate experimental warning from template with placeholder substitution
  EXPERIMENTAL_NOTE=$(echo "$EXPERIMENTAL_WARNING" | \
    sed "s|{BRANCH}|$CURRENT_BRANCH|g" | \
    sed "s|{VERSION}|$RELEASE_VERSION|g" | \
    sed "s|{REPO}|$GITHUB_REPOSITORY|g")
fi

# Prepend the experimental note if it exists
if [ -n "$EXPERIMENTAL_NOTE" ]; then
  echo "$EXPERIMENTAL_NOTE" > releasenotes_temp.md
  echo "" >> releasenotes_temp.md
  cat releasenotes.md >> releasenotes_temp.md
  mv releasenotes_temp.md releasenotes.md
fi

# Create the release using the notes file
# Word splitting is intentional for ASSET_PATHS, so we leave it unquoted
gh release create "$RELEASE_TAG" \
  --repo "$GITHUB_REPOSITORY" \
  --target "$GITHUB_SHA" \
  --title "$RELEASE_TITLE" \
  --notes-file releasenotes.md \
  $LATEST_FLAG \
  $PRERELEASE_FLAG \
  $ASSET_PATHS
