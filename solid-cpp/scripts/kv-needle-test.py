# Needle-in-a-haystack across context depths.
#
# KV quantization error accumulates over stored tokens: every key/value carries
# it and attention reads them all, so damage shows up at distance, not on short
# prompts. Planting a fact at several depths and asking for it back tests
# precisely that - and it works where llama-perplexity does not, since this model
# (Gemma-4 + MTP/iSWA) segfaults that tool over its ctx_other requirement.
import json, sys, urllib.request, uuid

PORT = sys.argv[1]
TARGET_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
NONCE = uuid.uuid4().hex[:6]  # unique per run, so the prompt cache cannot serve a previous run
NEEDLE = "The secret access code for the Meridian project is 74-BRAVO-91."
QUESTION = "What is the secret access code for the Meridian project? Reply with only the code."
FILLER = ("The archives of the northern trading company record shipments of grain, "
          "timber and salt across the inland routes during the long summer seasons. ")

def build(depth):
    # assemble, then splice the needle in at depth
    n = max(1, TARGET_TOKENS // 32)  # ~32 tokens per filler+record sentence
    body = [FILLER + f"Record {NONCE}-{i}. " for i in range(n)]
    body.insert(int(len(body) * depth), NEEDLE + " ")
    return "".join(body)

def ask(text):
    body = json.dumps({"messages": [{"role": "user", "content": text + "\n\n" + QUESTION}],
                       "max_tokens": 300, "temperature": 0, "seed": 42}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=1200)
    d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or m.get("reasoning_content") or "").strip(), d["timings"]["prompt_n"]

results = {}
for depth in (0.1, 0.5, 0.9):
    ans, ntok = ask(build(depth))
    ok = "74-BRAVO-91" in ans.replace(" ", "")
    results[depth] = ok
    print(f"  depth {int(depth*100):>2}%  ctx={ntok:>6} tokens  {'FOUND' if ok else 'MISSED'}  | {ans[:60]}")
print("  score:", sum(results.values()), "/", len(results))
