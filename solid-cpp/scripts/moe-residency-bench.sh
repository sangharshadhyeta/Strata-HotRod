#!/usr/bin/env bash
# Repeated-measurement harness for moe-cache residency policies.
#
# Exists because single-run comparisons on this rig proved unusable: nominally
# identical configurations ranged 7.59-13.11 and 10.30-33.37 tok/s, i.e. the
# run-to-run spread was larger than the effects being claimed.
#
# Three design points, each learned the hard way:
#
#  - Configurations are INTERLEAVED (A,B,C,A,B,C...), not blocked (AAA,BBB,CCC).
#    A blocked run charges any drift over the session - thermal, memory
#    fragmentation, disk state - entirely to whichever configuration happened to
#    run last, which is indistinguishable from a real effect. This matters more
#    than the wall-clock cost.
#
#  - Each server start yields ONE cold sample and several warm ones. Startup
#    under a memory cap dominates the cost (~90s of ~130s), so taking a single
#    warm sample per start throws away most of what was paid for. Cold samples
#    still need their own restart - that is irreducible, and cold is the metric
#    the pre-warm policies exist to move.
#
#  - Page cache dropped and server restarted before every repetition, so a cold
#    sample is genuinely cold rather than a measure of how warm the last test
#    left the machine.
set -u

MODEL=${MODEL:-/home/Projects/llama.cpp/models/moe-test2/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}
BIN=${BIN:-/home/Projects/llama.cpp/build/bin/llama-server}
PORT=${PORT:-8095}
CAP=${CAP:-5G}
LOAD_CAP=${LOAD_CAP:-24G}
CAP_BYTES=${CAP_BYTES:-5368709120}
REPS=${REPS:-5}
WARM=${WARM:-5}
NCMOE=${NCMOE:-23}
CTX=${CTX:-8192}
HIST=${HIST:-${BENCH_TMP:-/tmp/moe-bench}/moe-history.txt}
OUT=${OUT:-${BENCH_TMP:-/tmp/moe-bench}/bench}
PROMPT='{"messages":[{"role":"user","content":"Write three sentences about the ocean and three about mountains."}],"max_tokens":80}'

mkdir -p "$OUT"; rm -f "$OUT"/*.samples

CONFIGS=("baseline" "demote-soft" "demote-hard")
env_for() {
    case "$1" in
        baseline)           echo "GGML_CUDA_MOE_CACHE_HISTORY=/nonexistent GGML_CUDA_MOE_CACHE_COLD_AFTER_EPOCHS=0" ;;
        history)            echo "GGML_CUDA_MOE_CACHE_HISTORY=$HIST GGML_CUDA_MOE_CACHE_COLD_AFTER_EPOCHS=0" ;;
        demote-soft)        echo "GGML_CUDA_MOE_CACHE_HISTORY=$HIST GGML_CUDA_MOE_CACHE_COLD_AFTER_EPOCHS=4600 GGML_CUDA_MOE_CACHE_DEMOTE=cold" ;;
        demote-hard)        echo "GGML_CUDA_MOE_CACHE_HISTORY=$HIST GGML_CUDA_MOE_CACHE_COLD_AFTER_EPOCHS=4600 GGML_CUDA_MOE_CACHE_DEMOTE=pageout" ;;
    esac
}

stop_server() {
    systemctl stop "llama-bench-$PORT.scope" >/dev/null 2>&1
    pkill -9 -f "port $PORT" >/dev/null 2>&1
    sleep 2
}

measure() { # $1 label, $2 rep
    local label=$1 rep=$2
    stop_server
    # Evict only the model's pages, not the whole page cache. Global
    # drop_caches also flushes libraries and the CUDA runtime, which is most of
    # the startup cost and none of what the test needs cold.
    sync
    ${BENCH_TMP:-/tmp/moe-bench}/dropfile "$MODEL" 2>/dev/null

    # shellcheck disable=SC2046
    # Load with a loose cap, then tighten. Loading *under* pressure thrashes and
    # dominated the runtime (~2 of every 2.4 min), and it is not what we set out
    # to measure - inference under pressure is. Lowering memory.max on a live
    # cgroup forces immediate reclaim, which is also the more realistic shape:
    # a server that was fine and then came under pressure, rather than one
    # starved from birth.
    systemd-run --scope --quiet -p MemoryMax="$LOAD_CAP" -p MemorySwapMax=0 \
        --unit="llama-bench-$PORT" env $(env_for "$label") \
        "$BIN" -m "$MODEL" -ncmoe "$NCMOE" --jinja --moe-cache auto \
        -c "$CTX" -fit off --host 127.0.0.1 --port "$PORT" \
        > "$OUT/$label-$rep.log" 2>&1 &
    disown

    local up=0
    for _ in $(seq 1 100); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
        sleep 3
    done
    [ "$up" = 1 ] || { echo "  $label rep$rep: server did not start"; return; }

    # Apply the real cap now that the weights are in. Reclaim happens here, once,
    # instead of fighting the loader page by page.
    local cg="/sys/fs/cgroup/system.slice/llama-bench-$PORT.scope"
    if [ -w "$cg/memory.max" ]; then
        echo "$CAP_BYTES" > "$cg/memory.max"
        sleep 3   # let the kernel settle the forced reclaim before timing anything
    else
        echo "  WARNING: could not tighten memory.max - sample not under pressure"
    fi

    local out
    for i in $(seq 0 "$WARM"); do
        curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
            -d "$PROMPT" -o "$OUT/tmp.json" --max-time 900
        out=$(python3 -c "
import json
try:    print('%.2f' % json.load(open('$OUT/tmp.json'))['timings']['predicted_per_second'])
except Exception: print('nan')")
        [ "$out" = "nan" ] && continue
        if [ "$i" = 0 ]; then
            echo "$out" >> "$OUT/$label.cold.samples"
            echo "  $label rep$rep: cold $out tok/s"
        else
            echo "$out" >> "$OUT/$label.warm.samples"
        fi
    done
}

echo "=== moe-cache residency benchmark ==="
echo "cap=$CAP reps=$REPS warm-per-start=$WARM ncmoe=$NCMOE ctx=$CTX (interleaved)"
for rep in $(seq 1 "$REPS"); do
    echo "--- round $rep/$REPS ---"
    for cfg in "${CONFIGS[@]}"; do
        measure "$cfg" "$rep"
    done
done
stop_server

echo
echo "=== RESULTS ==="
for cfg in "${CONFIGS[@]}"; do
    python3 - "$cfg" "$OUT" <<'PY'
import sys, statistics, os
cfg, out = sys.argv[1], sys.argv[2]
def load(kind):
    p = os.path.join(out, f"{cfg}.{kind}.samples")
    if not os.path.exists(p): return []
    return [float(l) for l in open(p) if l.strip()]
def fmt(xs):
    xs = sorted(xs)
    med = statistics.median(xs)
    iqr = (statistics.quantiles(xs, n=4)[0], statistics.quantiles(xs, n=4)[2]) if len(xs) >= 4 else (xs[0], xs[-1])
    return "n=%2d  median %6.2f  IQR %6.2f-%-6.2f  min %6.2f  max %6.2f" % (
        len(xs), med, iqr[0], iqr[1], xs[0], xs[-1])
c, w = load("cold"), load("warm")
print("%-20s" % cfg)
if c: print("    cold  " + fmt(c))
if w: print("    warm  " + fmt(w))
PY
done
