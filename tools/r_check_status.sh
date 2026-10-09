#!/bin/sh
# Passes an R CMD check log whose only finding is the compiled-code warning for
# the stderr and abort references inside the vendored dependency archives.
# Any other note, warning or error fails, and so does that warning when it names
# the package's own object.
set -eu
log=$1
[ -r "$log" ] || { echo "R CMD check: cannot read $log" >&2; exit 1; }
count=$(grep -c '^Status: ' "$log" || true)
if [ "$count" -ne 1 ]; then
    echo "R CMD check: missing or repeated Status line" >&2
    exit 1
fi
status=$(sed -n 's/^Status: //p' "$log")
others=$(grep -E '\.\.\. (NOTE|WARNING|ERROR)$' "$log" | grep -v 'checking compiled code' || true)
if [ "$status" = OK ] && [ -z "$others" ]; then
    echo "R CMD check: OK"
    exit 0
fi
objects=$(grep -E 'Objects?:' "$log" || true)
bad_archives=$(printf '%s\n' "$objects" |
    grep -Eo 'lib/lib[[:alnum:]_.+-]+\.a' |
    grep -Ev '^lib/lib(openzl|zstd|lz4)\.a$' || true)
if [ "$status" != "1 WARNING" ] || [ -n "$others" ] ||
   ! grep -q 'checking compiled code .* WARNING' "$log" ||
   ! grep -Eq 'lib/lib(openzl|zstd|lz4)\.a' "$log" ||
   [ -z "$objects" ] || [ -n "$bad_archives" ] ||
   grep -q 'geozl_r\.o' "$log"; then
    echo "R CMD check: $status" >&2
    [ -n "$others" ] && echo "$others" >&2
    exit 1
fi
echo "R CMD check: $status, only the known compiled-code warning"
