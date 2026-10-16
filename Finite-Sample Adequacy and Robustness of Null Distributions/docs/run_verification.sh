#!/usr/bin/env bash
# READ-ONLY verification: run every check quoted in vignette.md and write
# docs/verification_log.txt.  No command here writes to the package, so the log is an
# independent check of the committed state rather than a record of it being rebuilt.
#
#   bash docs/run_verification.sh
#
# Writers that regenerate committed artefacts (the lock file, the manifest, the
# flowchart, the function reference) live in docs/regenerate.sh, which is a BUILD step
# and must be run before this one, not as part of it.  Mixing the two would let a
# verification run silently rewrite the environment record it is supposed to be checking.
#
# The archive run (`python3 code/experiments.py`, about 3.3 minutes, overwrites the
# committed results) is also excluded; Section 6 of the vignette describes running it in
# a scratch directory instead.
#
# Author: t4-docs (task-4); read-only split for reviewer comment 3.

set -u
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"
LOG="docs/verification_log.txt"

: > "$LOG"
{
  echo "mc-tk verification log (read-only checks)"
  echo "date       : $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "directory  : $(pwd)"
  echo "interpreter: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
  echo
} >> "$LOG"

run() {
  echo "\$ $*" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1
  status=$?
  echo "[exit $status]" >> "$LOG"
  echo >> "$LOG"
  return $status
}

fail=0
run "$PY" -V                        || fail=1
run "$PY" -m pip --version          || fail=1
run "$PY" docs/probe_experiments.py || fail=1
run "$PY" docs/probe_docstring_pass.py || fail=1
run "$PY" docs/verify_bartlett.py   || fail=1
run "$PY" code/verify_manuscript.py || fail=1
run "$PY" code/verify_dcor_against_R.py || fail=1
run "$PY" code/verify_letter_claims.py || fail=1
run "$PY" code/verify_crossrefs.py || fail=1
run "$PY" code/verify_references.py || fail=1
run "$PY" code/rev_study1_ext.py --validate || fail=1
run "$PY" docs/make_manifest.py --check || fail=1
run "$PY" -c "import sys; sys.path.insert(0, 'code'); import rev_power; print('rev_power imports cleanly')" || fail=1

echo "===============================================================" | tee -a "$LOG"
if [ "$fail" -eq 0 ]; then
  echo "ALL COMMANDS EXITED 0 -- log written to $LOG" | tee -a "$LOG"
else
  echo "AT LEAST ONE COMMAND FAILED -- see $LOG" | tee -a "$LOG"
fi
exit "$fail"
