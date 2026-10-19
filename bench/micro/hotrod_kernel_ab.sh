#!/usr/bin/env bash
# bench/micro/hotrod_kernel_ab.sh - bit-exactness of the hot-rod kernel switches, each setting in its own process.
#
# The switches are read once per process (static caches), so every (case, setting, seed) is a separate run of
# hotrod_kernel_ab. The raw outputs of each non-baseline setting are compared byte for byte (cmp) with the baseline
# setting's output for the same seed:
#   gr      baseline STRATA_GR_DOWN_WPC=8 (default)     vs 4, 2
#   router  baseline two launches (STRATA_ROUTER_FUSED unset) vs STRATA_ROUTER_FUSED=1
#   topk    baseline default (STRATA_TOPK_SPLIT unset)  vs STRATA_TOPK_SPLIT=1
# Outputs and logs go to $OUT (default /root/.claude/jobs/743230e4/tmp/kab). Usage: bench/micro/hotrod_kernel_ab.sh
set -u

HERE="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$HERE/build/hotrod_kernel_ab}"
OUT="${OUT:-/root/.claude/jobs/743230e4/tmp/kab}"
SEEDS="${SEEDS:-1 2 3}"
mkdir -p "$OUT" || exit 2
if [ ! -x "$BIN" ]; then
    echo "missing $BIN (cmake --build <build> --target hotrod_kernel_ab)" >&2
    exit 2
fi

# case  setting  [VAR=value ...]   (the first setting of each case is its baseline)
SETTINGS=(
    "gr wpc8 STRATA_GR_DOWN_WPC=8"
    "gr wpc4 STRATA_GR_DOWN_WPC=4"
    "gr wpc2 STRATA_GR_DOWN_WPC=2"
    "router twolaunch"
    "router fused STRATA_ROUTER_FUSED=1"
    "topk default"
    "topk split STRATA_TOPK_SPLIT=1"
)

# the baseline setting name of each case
case_base() {
    case "$1" in
        gr) echo wpc8 ;;
        router) echo twolaunch ;;
        topk) echo default ;;
    esac
}

# run one (case, setting, seed) with only the setting's variables set; prints the exit code
run_one() {
    local case="$1" name="$2" seed="$3"
    shift 3
    local base="$OUT/$case.$name.s$seed"
    env -u STRATA_GR_DOWN_WPC -u STRATA_ROUTER_FUSED -u STRATA_TOPK_SPLIT -u STRATA_HC_SPLIT "$@" \
        timeout 300 "$BIN" --case "$case" --seed "$seed" --out "$base.bin" > "$base.log" 2>&1
    echo $?
}

printf '%-7s %-10s %-5s %-4s %-10s %s\n' case setting seed rc bytes verdict-vs-baseline
any_fail=0
for seed in $SEEDS; do
    for entry in "${SETTINGS[@]}"; do
        read -r case name rest <<< "$entry"
        # shellcheck disable=SC2086
        rc=$(run_one "$case" "$name" "$seed" $rest)
        f="$OUT/$case.$name.s$seed.bin"
        bytes=$(stat -c %s "$f" 2>/dev/null || echo 0)
        base_f="$OUT/$case.$(case_base "$case").s$seed.bin"
        if [ "$rc" != 0 ]; then
            verdict="RUN FAILED (see $OUT/$case.$name.s$seed.log)"
            any_fail=1
        elif [ "$name" = "$(case_base "$case")" ]; then
            verdict="baseline"
        elif [ ! -s "$base_f" ]; then
            verdict="NO BASELINE OUTPUT"
            any_fail=1
        elif cmp -s "$base_f" "$f"; then
            verdict="IDENTICAL"
        else
            verdict="DIFFER: $(cmp "$base_f" "$f" 2>&1 | head -1)"
            any_fail=1
        fi
        printf '%-7s %-10s %-5s %-4s %-10s %s\n' "$case" "$name" "$seed" "$rc" "$bytes" "$verdict"
    done
done

echo
echo "variant lines (from the logs):"
for f in "$OUT"/*.log; do
    printf '%s: %s\n' "$(basename "$f")" "$(grep -E 'variant|path|differ|FAIL' "$f" | head -3 | tr '\n' ' ')"
done

if [ "$any_fail" != 0 ]; then
    echo "RESULT: at least one run failed or differed"
    exit 1
fi
echo "RESULT: every setting is byte-identical to its baseline for every seed"
