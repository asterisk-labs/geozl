#!/bin/sh
# Every benchmark on one machine: BACKEND=metal|cuda, REAL=path to 24 real cells.
set -e
cd "$(dirname "$0")"
B=${BACKEND:-cuda}
EXT=$([ "$B" = metal ] && echo metal || echo ptx)
REAL=${REAL:-build/cells24.u16}
run() {
  label=$1; shift
  echo "### $label"
  "$@" | grep -E "^(data|blocks|lanes|cpu|plan|device|passes|gpu|check)"
}
for input in synthetic "$REAL"; do
  name=$([ "$input" = synthetic ] && echo synthetic || echo "real S2 L1C")
  for codec in pfor pivco; do
    m=$([ $codec = pfor ] && echo pzp || echo pzv)
    run "$codec uint16, rumi tiles 176, $name" ./build/${codec}_$B --input "$input" --cells 24 --module build/${m}2.$EXT --reps 30
    run "$codec uint16, whole cells 1056, $name" ./build/${codec}_$B --input "$input" --cells 24 --mode cell --module build/${m}2.$EXT --reps 30
    run "$codec uint8, rumi tiles 176, $name" ./build/${codec}_$B --input "$input" --cells 24 --u8 --module build/${m}1.$EXT --reps 30
  done
done
