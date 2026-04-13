#!/usr/bin/env bash
# =============================================================================
# LLM-Proxy Log Cleanup Script
#
# Cleans up log artifacts that can't be handled by Python's RotatingFileHandler,
# specifically the per-request transaction directories and raw_io directories.
#
# Python's RotatingFileHandler handles:
#   - proxy.log        (50 MB × 3 backups)
#   - proxy_debug.log  (50 MB × 2 backups)
#   - failures.log     (5 MB × 2 backups)
#
# This script handles:
#   - logs/transactions/*   (per-request directories, biggest offender)
#   - logs/raw_io/*         (raw I/O debug directories)
#
# Transaction dirs are named: MMDD_HHMMSS_{format}_{provider}_{model}_{id}
# Since the date is embedded in the name, we parse it from there rather than
# relying on filesystem mtime (which can be unreliable across docker mounts).
#
# Install via cron.d:
#   cp cleanup-logs.sh /opt/llm-proxy/scripts/
#   echo '0 3 * * * root /opt/llm-proxy/scripts/cleanup-logs.sh >> /var/log/llm-proxy-cleanup.log 2>&1' > /etc/cron.d/llm-proxy-cleanup
#
# =============================================================================
set -u

# --- Configuration ---
LOG_BASE="${LLM_PROXY_LOG_DIR:-/opt/llm-proxy/logs}"
TRANSACTIONS_DIR="${LOG_BASE}/transactions"
RAW_IO_DIR="${LOG_BASE}/raw_io"

# Retention: delete transaction dirs older than this many days
TRANSACTION_RETENTION_DAYS="${TRANSACTION_RETENTION_DAYS:-7}"

# Retention for raw I/O debug logs (uses mtime since names are UUID-based)
RAW_IO_RETENTION_DAYS="${RAW_IO_RETENTION_DAYS:-3}"

# Maximum number of transaction dirs to keep (safety cap even for recent ones)
TRANSACTION_MAX_COUNT="${TRANSACTION_MAX_COUNT:-10000}"

# --- Functions ---
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup_transactions_by_name() {
    # Transaction dirs are named: MMDD_HHMMSS_...
    # We parse the MMDD prefix to determine age, inferring the year from
    # the current date (handles year rollover for Jan dirs viewed in Jan+).
    local target_dir="$1"
    local retention_days="$2"

    if [[ ! -d "$target_dir" ]]; then
        log_msg "SKIP: transactions directory does not exist: ${target_dir}"
        return 0
    fi

    local current_year
    current_year=$(date +%Y)
    local current_mmdd
    current_mmdd=$(date +%m%d)

    # Calculate the cutoff date
    local cutoff_epoch
    cutoff_epoch=$(date -d "-${retention_days} days" +%s)
    local cutoff_display
    cutoff_display=$(date -d "-${retention_days} days" +%Y-%m-%d)

    log_msg "CLEAN: transactions — scanning for dirs older than ${cutoff_display} ..."

    # Build list of dirs to delete using find + basename parsing
    # This avoids `ls` buffering issues with 50k+ entries
    local to_delete_file
    to_delete_file=$(mktemp)
    local total=0
    local marked=0

    find "$target_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | while IFS= read -r dir_name; do
        total=$((total + 1))

        # Extract MMDD from directory name (first 4 chars)
        mmdd="${dir_name:0:4}"

        # Validate it looks like a date (month 01-12, day 01-31)
        case "$mmdd" in
            0[1-9][0-3][0-9]|1[0-2][0-3][0-9]) ;;
            *) continue ;;
        esac

        month="${mmdd:0:2}"
        day="${mmdd:2:2}"

        # Infer year: if the MMDD is greater than current MMDD, it's likely
        # from last year (e.g., dir from December viewed in January)
        inferred_year="$current_year"
        if [[ "$mmdd" > "$current_mmdd" ]]; then
            inferred_year=$((current_year - 1))
        fi

        # Build a full date and compare against cutoff
        dir_epoch=$(date -d "${inferred_year}-${month}-${day}" +%s 2>/dev/null) || continue

        if [[ "$dir_epoch" -lt "$cutoff_epoch" ]]; then
            echo "${target_dir}/${dir_name}"
            marked=$((marked + 1))
        fi
    done > "$to_delete_file"

    local count
    count=$(wc -l < "$to_delete_file")
    log_msg "CLEAN: transactions — found ${count} dirs to delete"

    if [[ "$count" -gt 0 ]]; then
        # Use xargs for efficient bulk deletion
        xargs -d '\n' -P 4 -n 100 rm -rf < "$to_delete_file"
        log_msg "CLEAN: transactions — deleted ${count} dirs"
    fi

    rm -f "$to_delete_file"
}

cleanup_old_dirs_by_mtime() {
    local target_dir="$1"
    local retention_days="$2"
    local label="$3"

    if [[ ! -d "$target_dir" ]]; then
        log_msg "SKIP: ${label} directory does not exist: ${target_dir}"
        return 0
    fi

    local before_count
    before_count=$(find "$target_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

    if [[ "$before_count" -eq 0 ]]; then
        log_msg "SKIP: ${label} directory is empty"
        return 0
    fi

    log_msg "CLEAN: ${label} — removing dirs older than ${retention_days} days (current count: ${before_count})"

    local deleted
    deleted=$(find "$target_dir" -mindepth 1 -maxdepth 1 -type d -mtime "+${retention_days}" -print0 2>/dev/null \
        | xargs -0 -P 4 -n 50 rm -rf 2>/dev/null; \
        find "$target_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

    log_msg "CLEAN: ${label} — ${deleted} dirs remaining after cleanup"
}

enforce_max_count() {
    local target_dir="$1"
    local max_count="$2"
    local label="$3"

    if [[ ! -d "$target_dir" ]]; then
        return 0
    fi

    local current_count
    current_count=$(find "$target_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | wc -l)

    if [[ "$current_count" -le "$max_count" ]]; then
        log_msg "CAP: ${label} — ${current_count} dirs within max ${max_count}, no action needed"
        return 0
    fi

    local excess=$((current_count - max_count))
    log_msg "CAP: ${label} — ${current_count} dirs exceeds max ${max_count}, removing ${excess} oldest"

    # Transaction dirs sort chronologically by name (MMDD_HHMMSS prefix)
    # Use find + sort instead of ls for reliability with large directories
    find "$target_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | sort \
        | head -n "$excess" \
        | sed "s|^|${target_dir}/|" \
        | xargs -d '\n' -P 4 -n 100 rm -rf

    log_msg "CAP: ${label} — trimmed to ~${max_count} directories"
}

# --- Main ---
log_msg "========== LLM-Proxy Log Cleanup Starting =========="
log_msg "Config: TRANSACTION_RETENTION_DAYS=${TRANSACTION_RETENTION_DAYS}, RAW_IO_RETENTION_DAYS=${RAW_IO_RETENTION_DAYS}, TRANSACTION_MAX_COUNT=${TRANSACTION_MAX_COUNT}"

# Show disk usage before cleanup
if command -v du &>/dev/null; then
    log_msg "Disk usage before: $(du -sh "$LOG_BASE" 2>/dev/null | cut -f1)"
fi

# 1. Clean old transaction directories (by name-embedded date)
cleanup_transactions_by_name "$TRANSACTIONS_DIR" "$TRANSACTION_RETENTION_DAYS"

# 2. Enforce max count on transactions (sorted by name = chronological)
enforce_max_count "$TRANSACTIONS_DIR" "$TRANSACTION_MAX_COUNT" "transactions"

# 3. Clean old raw_io directories (by mtime, names are UUID-based)
cleanup_old_dirs_by_mtime "$RAW_IO_DIR" "$RAW_IO_RETENTION_DAYS" "raw_io"

# Show disk usage after cleanup
if command -v du &>/dev/null; then
    log_msg "Disk usage after:  $(du -sh "$LOG_BASE" 2>/dev/null | cut -f1)"
fi

log_msg "========== LLM-Proxy Log Cleanup Complete =========="
