#!/usr/bin/env bash
# BUILD step: regenerate the committed artefacts that are derived rather than measured.
#
#   bash docs/regenerate.sh
#
# This writes to the package: the pinned environment lock (from requirements.txt), the
# SHA-256 manifest, the execution flowchart and the function reference.  It is
# deliberately separate from docs/run_verification.sh so that a verification run cannot
# rewrite the environment record it is meant to be checking.
#
# Run after any change to requirements.txt, to the code, to draft.md or to the figures,
# then run docs/run_verification.sh to confirm the result.

set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"

fail=0
run() {
  echo "\$ $*"
  "$@"
  [ $? -ne 0 ] && fail=1
  return 0
}

run "$PY" docs/make_lock.py
run "$PY" docs/make_flowchart.py
run "$PY" docs/make_function_reference.py
run "$PY" docs/make_manifest.py
run "$PY" docs/make_manifest.py --lock
run "$PY" docs/make_manifest.py --check

if [ "$fail" -eq 0 ]; then
  echo "regenerated: lock, flowchart, function reference, manifest"
else
  echo "AT LEAST ONE REGENERATION STEP FAILED"
fi
exit "$fail"
