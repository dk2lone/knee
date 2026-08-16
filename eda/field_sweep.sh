#!/bin/bash
# Every session, before anything else: Code, Models, Discussion. All three, every time.
#
# Code alone is not enough. The single-model finding that reframed the whole strategy was
# in Discussion, the normalisation bug that would have wasted an 8-hour run was in
# Discussion, and neither is discoverable from a notebook diff.
#
# Prints only what is new since the last run. Everything it has already shown you lives in
# eda/.field_seen, so delete that file to replay the whole field.
set -u
C=rsna-knee-abnormality-detection
cd "$(dirname "$0")/.." || exit 1
SEEN=eda/.field_seen
touch "$SEEN"

echo "=== CODE: by score, then by votes ==="
for sort in scoreDescending voteCount; do
  kaggle kernels list --competition $C --sort-by $sort --page-size 20 2>/dev/null |
    tail -n +3 | awk '{print $1}'
done | sort -u | while read -r ref; do
  grep -qxF "k $ref" "$SEEN" && continue
  echo "NEW  $ref"
  d="nb/$(basename "$ref")"
  mkdir -p "$d" && kaggle kernels pull "$ref" -p "$d" -m >/dev/null 2>&1
  python3 - "$d" <<'PY'
import json, sys, pathlib
m = next(pathlib.Path(sys.argv[1]).glob("kernel-metadata.json"), None)
if m:
    j = json.loads(m.read_text())
    # Titles lie. The mounts do not.
    for k in ("dataset_sources", "model_sources", "kernel_sources"):
        if j.get(k):
            print(f"     {k:16} {j[k]}")
PY
  echo "k $ref" >> "$SEEN"
done

# Kaggle serves both of these from JavaScript, so the API returns nothing and only a real
# browser sees them. safari.sh is the automation tab; it gets closed at the end.
SAFARI=~/.claude/scripts/safari.sh
for tab in models discussion; do
  echo
  # macOS ships bash 3.2, which has no ${var^^}
  echo "=== $(echo "$tab" | tr '[:lower:]' '[:upper:]') ==="
  $SAFARI open "https://www.kaggle.com/competitions/$C/$tab?sort=votes" >/dev/null 2>&1
  sleep 4
  # No new/seen diff here on purpose. There are ~20 threads, they fit on one screen, and
  # a thread that was read once can grow the comment that matters. Read the list.
  $SAFARI text 2>/dev/null |
    sed -n '/^Most Usage/,/^1$/p;/^Pinned topics/,/^1$/p' |
    grep -vE '^(more_horiz|push_pin|arrow_drop_(up|down)|reply|Reply|React|add_reaction)$'
done
$SAFARI close >/dev/null 2>&1

echo
echo "Read every NEW thread in full before deciding it is irrelevant. Two of the three"
echo "findings of 16 Aug were in threads with 3 votes."
