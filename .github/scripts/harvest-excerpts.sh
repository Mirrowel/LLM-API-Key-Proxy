#!/usr/bin/env bash
# harvest-excerpts.sh — build docs/excerpts.json (the landing page's auto-card pool)
#
# Finds posts authored by the deployment's bot identities across the home
# repos (issue comments, PR comments, PR review summaries), drops hidden
# (minimized) content, applies a quality gate, and writes a capped,
# newest-first JSON pool the page deals random cards from.
#
# Contract:
#   env in : EXCERPT_REPOS  space-separated "owner/name" list (default: this
#                             deployment's home repos)
#            EXCERPT_BOTS   comma-separated bot logins, account AND app forms
#            EXCERPT_MAX    pool cap (default 80)
#            EXCERPT_DAYS   recency tier: newer-than-days posts fill first (180; 0 = off)
#            EXCERPT_MIN    backfill floor: older posts top up while recent tier is under this (40)
#            EXCERPT_MIN    backfill floor for older posts (default 40)
#            EXCERPT_OUT    output path (default docs/excerpts.json)
#   needs  : gh (authenticated; search + GraphQL + REST), jq
#   out    : EXCERPT_OUT = {"generated": iso, "days": n, "items": [{k,t,ti,r,n,u,w}...]}
#            k = comment|review, t = FULL text (the page clips for display), ti = thread title,
#            r = repo, n = number, u = source url, w = authored-at iso
#   exit   : 0 = pool written (possibly unchanged), 1 = config/tooling failure
#   notes  : authenticated gh only — never run against untrusted input; the
#            script is repo-local tooling for the informational page, not part
#            of the agent's trust boundary.
set -euo pipefail

EXCERPT_REPOS="${EXCERPT_REPOS:-Mirrowel/Mirrobot-agent Mirrowel/LLM-API-Key-Proxy}"
EXCERPT_BOTS="${EXCERPT_BOTS:-Mirrobot-Agent,mirrobot-agent[bot]}"
EXCERPT_MAX="${EXCERPT_MAX:-80}"
# recency tiering: posts newer than EXCERPT_DAYS fill the pool first; older
# material tops up only while the recent tier is under EXCERPT_MIN items
EXCERPT_DAYS="${EXCERPT_DAYS:-180}"
EXCERPT_MIN="${EXCERPT_MIN:-40}"
EXCERPT_OUT="${EXCERPT_OUT:-docs/excerpts.json}"

command -v gh >/dev/null 2>&1 || { echo "harvest: gh not found" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "harvest: jq not found" >&2; exit 1; }

# lowercase bot set for matching (GitHub logins are case-insensitive)
BOTS_LOWER=$(printf '%s' "$EXCERPT_BOTS" | tr ',' '\n' | tr -d ' "' | awk 'NF{print tolower($0)}' | paste -sd, -)
[ -n "$BOTS_LOWER" ] || { echo "harvest: no bot identities configured" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
CANDS="$WORK/candidates.tsv"   # repo|number|isPR|title
RAW="$WORK/raw.jsonl"          # one JSON object per harvested post
: > "$CANDS"; : > "$RAW"

# ---- 1. candidate threads: search per repo (commenter ORs + reviewed-by ORs) ----
# One search per kind per repo — repeated qualifiers are OR'ed by GitHub
# search, so all bot identities ride one query. This keeps search pressure
# low: the endpoint secondary-rate-limits aggressively (observed live: HTTP
# 403 after a few rapid harvests), which silently starves the pool.
# Every search is logged; failures retry once (60s backoff on secondary
# rate limits) and are counted loudly — they never abort the harvest.
SEARCH_FAILS=0
search_candidates() { # $1 = full q= query, $2 = label for logs, $3 = repo (tsv column 1)
  local out n tries=0 err="$WORK/search.err"
  while :; do
    tries=$((tries + 1))
    if out=$(gh api -X GET search/issues --paginate \
        -f q="$1" -f sort=created -f order=desc -f per_page=30 \
        --jq ".items[] | [\"$3\", .number, (if has(\"pull_request\") then \"pr\" else \"issue\" end), (.title // \"\")] | @tsv" 2>"$err"); then
      n=$(printf '%s' "$out" | grep -c .) || true
      echo "harvest: search [$2] ok, ${n:-0} thread(s)" >&2
      printf '%s\n' "$out"
      return 0
    fi
    echo "harvest: search [$2] FAILED (attempt $tries): $(tail -2 "$err" 2>/dev/null | tr '\n' ' ')" >&2
    [ "$tries" -ge 2 ] && { SEARCH_FAILS=$((SEARCH_FAILS + 1)); return 1; }
    if grep -qi "secondary rate" "$err" 2>/dev/null; then sleep 65; else sleep 4; fi
  done
}

for REPO in $EXCERPT_REPOS; do
  COMMENT_Q="repo:$REPO"; REVIEW_Q="repo:$REPO"
  for BOT in $(printf '%s' "$EXCERPT_BOTS" | tr ',' ' '); do
    COMMENT_Q="$COMMENT_Q commenter:$BOT"
    REVIEW_Q="$REVIEW_Q reviewed-by:$BOT"
  done
  search_candidates "$COMMENT_Q" "$REPO commenter" "$REPO" >> "$CANDS" || true
  sleep 2
  search_candidates "$REVIEW_Q" "$REPO reviewed-by" "$REPO" >> "$CANDS" || true
  sleep 2
done
sort -u -t$'\t' -k1,2 "$CANDS" -o "$CANDS"
TOTAL_CAND=$(wc -l < "$CANDS")
echo "harvest: $TOTAL_CAND candidate thread(s) across $EXCERPT_REPOS"

# ---- 2. per-thread GraphQL: bot-authored, non-minimized posts ----
fetch_thread() { # $1 = repo, $2 = number, $3 = issue|pr
  local OWNER NAME Q
  OWNER=${1%%/*}; NAME=${1##*/}
  if [ "$3" = "pr" ]; then
    Q='query($o:String!,$n:String!,$num:Int!){
      repository(owner:$o,name:$n){ pullRequest(number:$num){
        title
        comments(last:50){
          nodes{ author{login} isMinimized body url createdAt } }
        reviews(last:25){
          nodes{ author{login} body url state submittedAt } } } } }'
  else
    Q='query($o:String!,$n:String!,$num:Int!){
      repository(owner:$o,name:$n){ issue(number:$num){
        title
        comments(last:50){
          nodes{ author{login} isMinimized body url createdAt } } } } }'
  fi
  gh api graphql -f query="$Q" -f o="$OWNER" -f n="$NAME" -F num="$2" --jq "
    .data.repository.pullRequest // .data.repository.issue // empty |
    .title as \$ti |
    ( (.comments.nodes // [])
      | map(select(.isMinimized != true)
        | select(((.author.login // \"\") | ascii_downcase) as \$a | (\"$BOTS_LOWER\" | split(\",\") | index(\$a)))
        | {k:\"comment\", t:(.body // \"\"), ti:\$ti, w:(.createdAt // \"\"), u:(.url // \"\")}) )
    + ( if .reviews then (.reviews.nodes // [])
        | map(select(((.author.login // \"\") | ascii_downcase) as \$a | (\"$BOTS_LOWER\" | split(\",\") | index(\$a)))
          | select((.body // \"\") | length > 0)
          | {k:\"review\", t:(.body // \"\"), ti:\$ti, w:(.submittedAt // \"\"), u:(.url // \"\")})
        else [] end)" 2>/dev/null || { echo "harvest: thread $1#$2 fetch FAILED" >&2; FETCH_FAILS=$((FETCH_FAILS + 1)); true; }
}
FETCH_FAILS=0

while IFS=$'\t' read -r REPO NUM KIND TITLE; do
  [ -n "${REPO:-}" ] || continue
  TITLE=${TITLE%$'\r'}
  fetch_thread "$REPO" "$NUM" "$KIND" \
    | jq -c --arg r "$REPO" --argjson n "$NUM" '.[] | .r=$r | .n=$n' >> "$RAW" || true
done < "$CANDS"
TOTAL_RAW=$(wc -l < "$RAW")
echo "harvest: $TOTAL_RAW bot-authored post(s) collected"

# ---- 3. quality gate + dedupe + cap ----
# comment gate: 150..900 chars, prose-dominant (backtick fraction < .15),
#               and not a conversational ack (old-era "Thanks for the great
#               report" openers — the voice the platform moved away from)
# review gate:  80..2400 chars, backtick fraction < .35 (reviews carry code spans)
# both: strip AI-footer lines, skip pure-code or empty results (no server-side
# clipping — full text ships; the page clips for display)
jq -s --argjson max "$EXCERPT_MAX" --argjson days "$EXCERPT_DAYS" --argjson floor "$EXCERPT_MIN" '
  def recent: ((now - (.w | fromdateiso8601)) / 86400) <= $days;
  map(
    (.t // "") as $body0
    | ($body0 | split("\n") | map(select((ascii_downcase | test("generated by an ai")) | not)) | join("\n") | sub("^\\s+|\\s+$";"")) as $body
    | .t = $body
    | select($body | length >= 80)
    | select(([$body | scan("`")] | length) as $bt
      | if .k == "review"
        then ($body | length <= 2400) and (($bt * 3) < ($body | length))
        else ($body | length >= 150) and ($body | length <= 900) and (($bt * 7) < ($body | length))
          and ((($body | test("^@\\S+[,:]?\\s+(thanks|on it|i.?m on it|hi\\b|hello|acknowledg)";"i"))
             or ($body | test("^(you.?re (absolutely )?right|good catch|great (catch|report)|looks good to merge|time to review my own work)";"i"))) | not) end)
    | select((.u | length) > 0)
  )
  | unique_by(.u)
  | sort_by(.w) | reverse
  # two-tier recency: posts newer than $days fill the pool first; older ones
  # top up only while the recent tier sits under $floor — recency leads, the
  # pool never starves. ($days = 0 disables tiering: plain newest-first cap.)
  | (if $days > 0 then
      (map(select(recent))) as $R
      | (map(select(recent | not))) as $O
      | $R[0:$max] + (if ($R|length) < $floor then $O[0:($floor - ($R|length))] else [] end)
    else . end)
  | .[0:$max]
  | {generated: (now | todateiso8601), days: $days, items: .}
' "$RAW" > "$EXCERPT_OUT"

POOL=$(jq '.items | length' "$EXCERPT_OUT")
echo "harvest: pool = $POOL item(s) -> $EXCERPT_OUT"
echo "harvest: health — $SEARCH_FAILS search failure(s), $FETCH_FAILS thread-fetch failure(s)"
[ "$SEARCH_FAILS" -eq 0 ] || echo "harvest: WARNING searches failed — pool may be starved; the page shows only recent items and degrades to a notice when there are none" >&2
