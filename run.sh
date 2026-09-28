#!/bin/sh
# Entry point for the container. Usage:
#   docker run --rm fl-aggregation-evidence            # Track 1 (seconds)
#   docker run --rm fl-aggregation-evidence track2     # Track 2 (about ten minutes)
#   docker run --rm fl-aggregation-evidence all        # both
set -e

track1() {
  echo "== Track 1: controlled counterexample and blind verifier =="
  python evidence.py
  python check_ab.py
  python bit_exact_check.py
  python separability.py
}

track2() {
  echo "== Track 2: empirical reachability study =="
  python m1_run.py
  python m1_analyse.py
  python mkfig_m1.py
  python m1_krumfloor.py
}

case "${1:-track1}" in
  track1) track1 ;;
  track2) track2 ;;
  all)    track1; track2 ;;
  *)      exec "$@" ;;
esac
