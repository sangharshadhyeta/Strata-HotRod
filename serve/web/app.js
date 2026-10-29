// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
function showTab(name) {
  tab = ["chat", "images", "monitor", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "images", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (tab === "images") $("img-prompt").focus();
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadConfig();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
function headers(json = false) {
  const h = {};
  const key = store.get("apikey", "");
  if (key) h.Authorization = "Bearer " + key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = store.get("apikey", "");
$("api-key").onchange = () => { store.set("apikey", $("api-key").value.trim()); toast("success", "API key saved", "Kept in this browser only."); };

let health = {model: "strata", images: false, max_context: 0};
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                          : "Attach a text file (or drop it here)";
    $("chat-empty-sub").textContent = `${health.model} runs on this PC. Nothing leaves it.`;
    $("tab-btn-images").hidden = !health.image_generation;
    if (location.hash === "#images" && health.image_generation) showTab("images");
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

// the poll: the requests table's span, and the timeline rows since the page's newest (tl_after) of this run (tl_base)
function metricsUrl() {
  const q = [reqShowAll ? "requests=all" : "", tl.after != null && tl.base != null ? `tl_after=${tl.after}&tl_base=${tl.base}` : ""];
  const s = q.filter(Boolean).join("&");
  return s ? `metrics?${s}` : "metrics";
}
let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
async function poll() {
  try {
    const r = await fetch(metricsUrl(), {headers: headers()});
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About > Settings.", 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else {
    setPill("idle", "Idle");
  }
  if (live.queued > 0) setPill("queued", `${live.queued} queued`);
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "monitor") renderConvCache(m.conversation_cache);
  if (tab === "monitor") renderResidency(m.residency);
  if (tab === "monitor") renderTimeline(m.timeline);
  if (tab === "about") renderAbout(eng, hw, st);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
// the experts held in VRAM, from the engine's swap round: rows[layer][expert] is '0' not resident, '1' resident and idle, '2' used, '3' hot
function renderResidency(r) {
  const card = $("res-card");
  card.hidden = !r;
  if (!r) return;                                  // an older engine, or no swap round yet
  const canvas = $("res-canvas"), n = r.experts, rows = r.rows, L = rows.length;
  // one layer = a block of bw x bh cells (512 experts: 32 x 16), the blocks tiled bcols across with a 1-cell gap: every expert is a square
  const bw = n % 32 === 0 ? 32 : Math.ceil(Math.sqrt(n)), bh = Math.ceil(n / bw);
  const bcols = Math.min(L, 8), brows = Math.ceil(L / bcols);
  const cw = bcols * bw + (bcols - 1), ch = brows * bh + (brows - 1);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  const css = getComputedStyle(document.documentElement);
  const tone = (name, fallback) => {                // "#rrggbb" of a token, drawn into an ImageData
    const v = css.getPropertyValue(name).trim();
    const m = /^#([0-9a-f]{6})$/i.exec(v) || /^#([0-9a-f]{6})$/i.exec(fallback);
    const x = parseInt(m[1], 16);
    return [(x >> 16) & 255, (x >> 8) & 255, x & 255];
  };
  const bg = tone("--st-surface-2", "#f7f7f7"), acc = tone("--st-accent", "#10b981"), hot = tone("--st-accent-hover", "#0c8a60");
  const surf = tone("--st-surface", "#ffffff");
  const idle = acc.map((c, i) => Math.round(0.28 * c + 0.72 * surf[i]));
  const colors = [bg, idle, acc, hot];
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cw, ch);
  const gap = tone("--st-line", "#e5e7eb");           // the 1-cell gaps between the layers' blocks
  for (let i = 0; i < cw * ch; i++) {
    img.data[i * 4] = gap[0]; img.data[i * 4 + 1] = gap[1]; img.data[i * 4 + 2] = gap[2]; img.data[i * 4 + 3] = 255;
  }
  rows.forEach((row, l) => {
    const x0 = (l % bcols) * (bw + 1), y0 = Math.floor(l / bcols) * (bh + 1);
    for (let e = 0; e < n; e++) {
      const c = colors[row.charCodeAt(e) - 48] || bg, o = ((y0 + Math.floor(e / bw)) * cw + x0 + (e % bw)) * 4;
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  });
  ctx.putImageData(img, 0, 0);
  const total = r.layers * r.experts;
  $("res-sum").textContent = `${fmt(r.resident)} of ${fmt(total)} experts · ${fmt(r.swaps)} swaps in ${fmt(r.round)} rounds${r.quality_swaps ? ` (${fmt(r.quality_swaps)} for quality)` : ""}`;
}
// the decode timeline: a scrolling view of the engine's host times, one buffer a second from /metrics (layers: [window, layer, tokens,
// demand reads, CPU jobs, doorbell, plan published, quantise, reads start, reads done, jobs ready, CPU done] in us after the run's
// base; reads: [I/O thread, layer, demand, start, end]; gaps: [at, windows lost]).  The server keeps the last 15 s.  While following,
// the right edge is the engine's clock (set by each poll, moved every frame); a drag, a wheel or Pause stops following, Live resumes.
const TL_KINDS = [
  ["gpu", "GPU: experts in VRAM, then waiting for the CPU's rows", "--st-info", 0.45],
  ["gpu2", "GPU: combine + next layer (the host waits)", "--st-info", 1],
  ["gpuw", "GPU: commit + draft (window end)", "--st-info", 0.7],
  ["plan", "host: plan + quantise", "--st-ink-muted", 0.55],
  ["read", "host: demand reads (drive)", "--st-danger", 1],
  ["bwait", "host: waits on a read already in flight", "--st-warn", 1],
  ["cpu", "CPU experts", "--st-accent", 1],
  ["guess", "I/O: read-ahead guess", "--st-warn", 0.55],
  ["dread", "I/O: demand read", "--st-danger", 0.7],
];
const TL_IX = Object.fromEntries(TL_KINDS.map((k, i) => [k[0], i]));
const TL_LW = 92, TL_LANE = 22, TL_SPAN = 4000, TL_BUCKET = 250, TL_LAG = 1000, TL_KEEP_US = 15000000;   // ms: the visible span, the index's buckets, and how far
// behind the engine's clock the right edge sits (a window reaches the page at the next 0.5 s write and the next poll)
const tl = {v0: 0, v1: TL_SPAN, width: TL_SPAN, follow: true, anchor: null, drag: null, on: false, busy: false, dirty: true, raf: 0, last: 0};
tlClear();
function renderTimeline(t) {
  if (!t) { $("tl-card").hidden = true; return; }        // an older server
  $("tl-card").hidden = false;
  tl.on = !!t.on;
  if (!tl.busy) { const b = $("tl-engine"); b.setAttribute("aria-pressed", String(tl.on)); b.textContent = tl.on ? "Stop" : "Record"; }
  const live = tl.on && !!t.data;
  $("tl-off").hidden = live;
  if (tl.on && !t.data) $("tl-off").textContent = "Recording: the first decode windows show here within a few seconds of a request.";
  $("tl-body").hidden = !live;
  tlButtons();
  if (!live) { tlClear(); $("tl-sum").textContent = ""; return; }
  tlApply(t.data);
  tl.dirty = true;
  tlKick();
}
function tlButtons() {
  $("tl-btn").textContent = tl.follow ? "Pause" : "Resume";
  $("tl-btn").setAttribute("aria-pressed", String(!tl.follow));
  $("tl-reset").hidden = tl.follow;
}
// the page's own buffer: the server sends the rows since its newest (tl_after) and keeps only what it has; the page keeps the last
// TL_KEEP_US of them, indexed by start time, and appends each poll's rows.  A reset (a first poll, or another run) starts it over.
function tlClear() {
  tl.data = null; tl.layers = []; tl.prev = null; tl.buckets = new Map(); tl.maxLen = 0; tl.gaps = []; tl.ios = new Set();
  tl.used = new Set(); tl.after = null; tl.base = null; tl.bmin = 0; tl.lanes = [["gpu", "GPU (inferred)"], ["host", "host thread"], ["cpu", "CPU pool"]];
  tl.laneIx = Object.fromEntries(tl.lanes.map((l, i) => [l[0], i]));
}
function tlAdd(lane, kind, a, b, label) {
  const s = {l: lane, k: TL_IX[kind], a: a / 1000, b: b / 1000, t: label};
  const key = Math.floor(s.a / TL_BUCKET);
  if (!tl.buckets.has(key)) tl.buckets.set(key, []);
  tl.buckets.get(key).push(s);
  tl.maxLen = Math.max(tl.maxLen, s.b - s.a);
  tl.used.add(s.k);
}
// one layer's segments; the GPU's gap after the previous layer is drawn when this one comes (its window ends there)
function tlAddLayer(r) {
  const [w, l, nt, mis, jobs, t0, tp, t1, t2, tpf, t3, t4] = r;
  if (tl.prev) {
    const p = tl.prev, kind = p[0] !== w ? "gpuw" : "gpu2";
    tlAdd("gpu", kind, p[11], t0, kind === "gpuw" ? `window end after layer ${p[1]}: commit, draft, next window's first layer`
                                                  : `layer ${p[1]} combine, layer ${l} attention + router (host waits)`);
  }
  tl.prev = r;
  tlAdd("host", "plan", t0, Math.max(tp, t2), `layer ${l}: plan + quantise`);
  if (tpf > t2) {   // the CPU experts' blobs: read from the drive (mis of them), the rest from RAM or a landed guess
    tlAdd("host", mis > 0 ? "read" : "plan", t2, tpf, mis > 0 ? `layer ${l}: ${mis} demand read${mis > 1 ? "s" : ""} from the drive`
                                                            : `layer ${l}: CPU experts' blobs from RAM`);
  }
  if (t3 - tpf > 20) tlAdd("host", "bwait", tpf, t3, `layer ${l}: waiting on reads already in flight`);
  if (jobs > 0 && t4 > t3) tlAdd("cpu", "cpu", t3, t4, `layer ${l}: ${jobs} expert${jobs > 1 ? "s" : ""} on the CPU`);
  tlAdd("gpu", "gpu", tp, t4, `layer ${l}: VRAM experts, then the CPU's rows`);
}
function tlApply(d) {
  if (d.reset || d.base_ns !== tl.base) tlClear();
  tl.base = d.base_ns;
  tl.data = d;
  tl.anchor = {eng: d.now_us / 1000, perf: performance.now()};    // the engine's clock at this poll, ms
  let after = tl.after ?? -Infinity;
  for (const r of d.layers) { tl.layers.push(r); tlAddLayer(r); after = Math.max(after, r[11]); }
  for (const r of d.reads) {
    tlAdd(`io${r[0]}`, r[2] ? "dread" : "guess", r[3], r[4], `I/O thread ${r[0]}: ${r[2] ? "demand read" : "guess"} for layer ${r[1]}`);
    after = Math.max(after, r[4]);
  }
  for (const g of d.gaps) tl.gaps.push({at: g[0] / 1000, lost: g[1]});
  if (isFinite(after)) tl.after = after;
  // keep TL_KEEP_US behind the newest row: the rows and the buckets (by start) older than that go
  const cut = (tl.after ?? 0) - TL_KEEP_US;
  let k = 0;
  while (k < tl.layers.length && tl.layers[k][11] < cut) k++;
  if (k) tl.layers.splice(0, k);
  for (const key of [...tl.buckets.keys()]) if (key < Math.floor(cut / 1000 / TL_BUCKET)) tl.buckets.delete(key);
  tl.gaps = tl.gaps.filter(g => g.at * 1000 >= cut);
  tl.bmin = tl.layers.length ? tl.layers[0][5] / 1000 : 0;
  const ids = [...tl.ios, ...((d.reads || []).map(r => r[0]))];
  tl.ios = new Set(ids);
  const ios = [...tl.ios].sort((a, b) => a - b);
  tl.lanes = [["gpu", "GPU (inferred)"], ["host", "host thread"], ["cpu", "CPU pool"], ...ios.map(i => [`io${i}`, `I/O ${i}`])];
  tl.laneIx = Object.fromEntries(tl.lanes.map((l, i) => [l[0], i]));
  tlSummary();
}
// the summary and the legend over the rows the buffer holds
function tlSummary() {
  const L = tl.layers;
  let reads = 0, bwait = 0, cpu = 0, gpuOnly = 0;
  const wins = new Map();
  const ms = (a, b) => (b - a) / 1000;
  L.forEach((r, i) => {
    const [w, , nt, mis, jobs, t0, tp, , t2, tpf, t3, t4] = r;
    if (!wins.has(w)) wins.set(w, nt);
    if (tpf > t2 && mis > 0) reads += ms(t2, tpf);
    if (t3 - tpf > 20) bwait += ms(tpf, t3);
    if (jobs > 0 && t4 > t3) cpu += ms(t3, t4);
    const nx = L[i + 1];
    if (nx && nx[0] === w) gpuOnly += ms(t4, nx[5]);
  });
  const nw = wins.size || 1, tot = (L.length ? (L[L.length - 1][11] - L[0][5]) / 1000 : 0) || 1;
  const toks = [...wins.values()].reduce((a, n) => a + n, 0);
  const part = (v, label) => `<span><b>${fmt(v / nw, 1)} ms</b> ${label} (${fmt((100 * v) / tot)}%)</span>`;
  $("tl-why").innerHTML = part(gpuOnly, "GPU alone, host waiting") + part(reads, "demand reads") + part(bwait, "waits on reads in flight") +
                          part(cpu, "CPU experts");
  $("tl-sum").textContent = `${nw} window${nw > 1 ? "s" : ""} · ${fmt(tot / nw, 1)} ms each · ${fmt(toks / nw, 1)} rows a window`;
  $("tl-legend").innerHTML = TL_KINDS.filter((k, i) => tl.used.has(i))
    .map(k => `<span><i class="res-sw" style="background:${tlColor(k)}"></i>${esc(k[1])}</span>`).join("");
}
function tlColor(k) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(k[2]).trim() || "#888";
  return k[3] >= 1 ? v : `color-mix(in srgb, ${v} ${Math.round(k[3] * 100)}%, transparent)`;
}
// the x of a time (ms) in a canvas W px wide, for the view as it is now
const tlPx = (t, W) => TL_LW + ((t - tl.v0) / (tl.v1 - tl.v0)) * (W - TL_LW - 4);
function drawTimeline() {
  const cv = $("tl-canvas"), ctx = cv.getContext("2d"), LH = TL_LANE, LW = TL_LW;
  const dpr = devicePixelRatio || 1, W = cv.clientWidth || 600, H = tl.lanes.length * LH + 20;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement), mut = css.getPropertyValue("--st-ink-muted"), line = css.getPropertyValue("--st-line");
  const colors = TL_KINDS.map(k => tlColor(k));
  ctx.clearRect(0, 0, W, H);
  ctx.font = "11px " + (css.getPropertyValue("--st-font") || "system-ui");
  const span = tl.v1 - tl.v0, x = t => tlPx(t, W);
  tl.lanes.forEach(([id, name], i) => {
    ctx.fillStyle = mut; ctx.fillText(name, 4, i * LH + 15);
    ctx.fillStyle = line; ctx.fillRect(LW, i * LH + LH - 1, W - LW, 1);
  });
  // the short segments (under 2 px) are drawn once per lane, kind and pixel: a 4 s view of 30 ms windows has more of them than pixels
  const drawn = new Set();
  for (let b = Math.floor((tl.v0 - tl.maxLen) / TL_BUCKET), e = Math.floor(tl.v1 / TL_BUCKET); b <= e; b++) {
    for (const s of tl.buckets.get(b) || []) {
      if (s.b < tl.v0 || s.a > tl.v1) continue;
      const i = tl.laneIx[s.l], a = Math.max(LW, x(s.a)), z = Math.max(a + 1, Math.min(W - 4, x(s.b)));
      if (z - a < 2) {
        const key = (Math.floor(a) * 64 + i) * 16 + s.k;
        if (drawn.has(key)) continue;
        drawn.add(key);
      }
      ctx.fillStyle = colors[s.k];
      ctx.fillRect(a, i * LH + 4, z - a, LH - 9);
    }
  }
  // a gap: the engine's ring overwrote windows before a chunk carried them (a thin line and the count)
  ctx.fillStyle = css.getPropertyValue("--st-warn") || "#c80";
  for (const g of tl.gaps) {
    if (g.at < tl.v0 || g.at > tl.v1) continue;
    const gx = x(g.at);
    ctx.fillRect(gx, 0, 1, H - 20);
    ctx.fillText(`${g.lost} lost`, gx + 3, 11);
  }
  // ticks at 1, 2 or 5 x 10^n ms, at least 70 px apart; the labels count back from the right edge (now)
  const want = span / Math.max(1, Math.floor((W - LW) / 70));
  const mag = Math.pow(10, Math.floor(Math.log10(want))), step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= want);
  ctx.fillStyle = mut;
  for (let t = Math.ceil(tl.v0 / step) * step; t <= tl.v1; t += step) {
    const a = x(t);
    ctx.fillRect(a, tl.lanes.length * LH, 1, 4);
    ctx.fillText(`${fmt(t - tl.v1, step < 1 ? 1 : 0)} ms`, a + 2, tl.lanes.length * LH + 15);
  }
  tl.dirty = false;
}
// the segment of a lane under a time (ms), or null
function tlSegAt(lane, t) {
  const b = Math.floor(t / TL_BUCKET);
  for (let k = Math.floor((t - tl.maxLen) / TL_BUCKET); k <= b; k++)
    for (const s of tl.buckets.get(k) || []) if (s.l === lane && s.a <= t && s.b >= t) return s;
  return null;
}
function tlAt(e) {
  const cv = $("tl-canvas"), R = cv.getBoundingClientRect();
  return {t: tl.v0 + ((e.clientX - R.left - TL_LW) / (cv.clientWidth - TL_LW - 4)) * (tl.v1 - tl.v0), lane: Math.floor((e.clientY - R.top) / TL_LANE)};
}
// the engine's clock now (ms): the last poll's, moved on by the time since, less the lag the right edge keeps
function tlNow() { return tl.anchor ? tl.anchor.eng + (performance.now() - tl.anchor.perf) - TL_LAG : tl.v1; }
function tlKick() { if (!tl.raf) tl.raf = requestAnimationFrame(tlFrame); }
// every frame while the view follows the clock; drawn at most 30 times a second, and a view that has stopped moving is drawn once
function tlFrame(ts) {
  tl.raf = 0;
  if (!tl.data || !tl.on || tab !== "monitor") return;
  if (tl.follow) { const now = tlNow(); tl.v1 = now; tl.v0 = now - tl.width; }
  if ((tl.follow || tl.dirty) && ts - tl.last >= 33) { tl.last = ts; drawTimeline(); }
  if (tl.follow || tl.dirty) tlKick();
}
(function initTimeline() {
  const cv = $("tl-canvas"), tip = $("tl-tip");
  cv.addEventListener("mousemove", e => {
    const p = tlAt(e), R = cv.getBoundingClientRect(), W = cv.clientWidth, lane = tl.lanes[p.lane];
    const g = !tl.drag && tl.gaps.find(g => Math.abs(tlPx(g.at, W) - (e.clientX - R.left)) < 4);
    const s = !g && !tl.drag && lane && tlSegAt(lane[0], p.t);
    const text = g ? `${g.lost} window${g.lost > 1 ? "s" : ""} lost here: the engine's ring overwrote them before a chunk carried them`
                   : s ? `${s.t} · ${fmt(s.b - s.a, 2)} ms` : "";
    tip.hidden = !text;
    if (!text) return;
    tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY + 12 + "px";
    tip.textContent = text;
  });
  cv.addEventListener("mouseleave", () => (tip.hidden = true));
  // the view: double-click or the wheel zooms around the pointer, a drag pans; a view moved this way stops following the clock
  const setView = (v0, v1) => {
    const max = Math.max(tl.bmin + 20, tlNow()), w = Math.min(15000, Math.min(max - tl.bmin, Math.max(20, v1 - v0)));
    tl.v0 = Math.min(Math.max(tl.bmin, v0), max - w);
    tl.v1 = tl.v0 + w;
    tl.width = w;
    tl.follow = false;
    tlButtons();
    drawTimeline();
  };
  const zoomAt = (e, factor) => {
    const t = tlAt(e).t, w = (tl.v1 - tl.v0) / factor, f = (t - tl.v0) / (tl.v1 - tl.v0);
    setView(t - f * w, t - f * w + w);
  };
  cv.addEventListener("dblclick", e => { e.preventDefault(); zoomAt(e, e.shiftKey ? 0.5 : 2); });
  cv.addEventListener("wheel", e => {
    if (!tl.data) return;
    e.preventDefault();                               // the wheel zooms the timeline, not the page, while over it
    zoomAt(e, Math.exp(-e.deltaY * 0.0015));
  }, {passive: false});
  cv.addEventListener("mousedown", e => {
    if (e.button !== 0) return;
    e.preventDefault();                               // no text selection while dragging or double-clicking
    tl.drag = {x: e.clientX, v0: tl.v0, v1: tl.v1};
    cv.style.cursor = "grabbing";
    tip.hidden = true;
  });
  addEventListener("mousemove", e => {
    if (!tl.drag) return;
    const dt = ((e.clientX - tl.drag.x) / (cv.clientWidth - TL_LW)) * (tl.drag.v1 - tl.drag.v0);
    setView(tl.drag.v0 - dt, tl.drag.v1 - dt);
  });
  addEventListener("mouseup", () => { if (tl.drag) { tl.drag = null; cv.style.cursor = ""; } });
  // Pause freezes the view (the engine keeps recording and the buffer keeps filling); Resume follows the clock again
  $("tl-btn").addEventListener("click", () => {
    tl.follow = !tl.follow;
    tlButtons();
    tl.dirty = true;
    tlKick();
  });
  // Live: back to the clock at the default span
  $("tl-reset").addEventListener("click", () => {
    tl.follow = true;
    tl.width = TL_SPAN;
    tlButtons();
    tl.dirty = true;
    tlKick();
  });
  addEventListener("resize", () => tl.data && drawTimeline());
  // the engine's switch (POST /v1/monitor/timeline): off stops the stamps, on starts them (the engine starts with the switch on)
  $("tl-engine").addEventListener("click", async () => {
    const want = !tl.on, btn = $("tl-engine");
    tl.busy = true; btn.disabled = true;
    try {
      const r = await fetch("v1/monitor/timeline", {method: "POST", headers: headers(true), body: JSON.stringify({on: want})});
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error?.message || `HTTP ${r.status}`);
      tl.on = want;
      if (!want) tlClear();
      btn.setAttribute("aria-pressed", String(want)); btn.textContent = want ? "Stop" : "Record";
      $("tl-off").hidden = false;
      $("tl-off").textContent = want ? "Recording: the first decode windows show here within a few seconds of a request." : $("tl-off").dataset.text;
      $("tl-body").hidden = true;
    } catch (e) {
      toast("warn", "Timeline", String(e.message || e), 5000);
    } finally {
      tl.busy = false; btn.disabled = false;
    }
  });
  $("tl-off").dataset.text = $("tl-off").textContent;
})();
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).";
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : live.state === "generating" ? "Prefill this request" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || (st.gpu_note ? "not available" : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : (st.gpu_note ? "not available on Windows AMD yet" : ""));
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
            (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  // context fill: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * frac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * frac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(frac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="9" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      // Reused tokens are not prefetched again. Match the Monitor's existing Prefill metric.
      const fresh = r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0));
      const prefillRate = fresh > 0 && r.prompt_ms > 0 ? fresh / (r.prompt_ms / 1000) : null;
      return `<tr><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(prefillRate)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : (st.gpu_note || "not readable (NVML)")],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return `<label for="${id}" title="${esc(k.help)}">${esc(k.help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Start the model again to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
});

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
// The sampling shown until the user changes it is the model card's thinking set. Untouched, the page sends NO sampling
// fields: the server picks the card's set for the request's thinking state (thinking 1.0 / 0.95 / 20, not thinking
// 0.7 / 0.8 / 20 with a presence penalty). Before 0.1.45 the page sent 0.6 itself; a stored 0.6 with the other two
// values untouched counts as "not set" too.
const DEFAULTS = {thinking: "high", temperature: 1.0, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true};
const OLD_DEFAULT_TEMPERATURE = 0.6;
function samplingUntouched(s) {
  const t = +s.temperature;
  return (t === DEFAULTS.temperature || t === OLD_DEFAULT_TEMPERATURE) && +s.top_p === DEFAULTS.top_p && +s.top_k === DEFAULTS.top_k;
}
let settings = {...DEFAULTS, ...store.get("sampling", {})};
if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
let messages = store.get("chat", []);
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

function saveChat() {
  store.set("chat", messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                           files: (m.files || []).map((f) => ({name: f.name}))})));
}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}${m.meta ? ` · ${m.meta}` : ""}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? (m.liveMeta || "") : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages(draft) {       // draft (20 Oct): a user message not sent yet, read as the last one (the draft warm)
  const out = [];
  for (const m of draft ? [...messages, draft] : messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

// The body Send posts for the conversation (draft: a user message not sent yet). One function, so a warm (POST /v1/prefill)
// reads the same prompt prefix as the real request.
function requestBody(draft) {
  const body = {model: health.model, messages: apiMessages(draft), stream: true,
                reasoning_effort: settings.thinking};
  if (!(settings.temperature > 0)) {
    body.temperature = 0;
  } else if (!samplingUntouched(settings)) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it
  return body;
}

// ------------------------------------------------------------------ reading the draft while you type (POST /v1/prefill)
// The engine reads the conversation with the draft while you type, so Send reads only what changed. Nothing is shown:
// the answer and the errors of a warm go to the console only.
let warmTimer = null, warmCtl = null, warmKey = null, warmPasted = false;
function cancelWarm() {
  clearTimeout(warmTimer);
  warmTimer = null;
  if (warmCtl) warmCtl.abort();
  warmCtl = null;
}
function warmDraft() {
  warmTimer = null;
  const text = $("input").value.trim();
  if (!text || busy || attachments.some((a) => a.kind !== "file")) return;   // pictures are not warmed: vision encodes are not prefix-reusable
  const files = attachments.filter((a) => a.kind === "file");
  const key = JSON.stringify([text, files.map((f) => f.name)]);
  if (key === warmKey) return;
  warmKey = key;
  if (warmCtl) warmCtl.abort();
  const controller = new AbortController();
  warmCtl = controller;
  const draft = {role: "user", text, images: [], files, time: Date.now()};
  fetch("v1/prefill", {method: "POST", headers: headers(true), body: JSON.stringify(requestBody(draft)), signal: controller.signal})
    .then((r) => r.json())
    .then((j) => console.debug("prefill", j))
    .catch((e) => console.debug("prefill", e && e.name))
    .finally(() => { if (warmCtl === controller) warmCtl = null; });
}
function scheduleWarm(ms) {
  clearTimeout(warmTimer);
  warmTimer = setTimeout(warmDraft, ms);
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  cancelWarm();                    // the server keeps a warm read that the request continues (Service.warm)
  warmKey = null;
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  // while it runs: under your message the prompt being read (then its read speed), under the answer the tokens so far and the live decode speed
  const userEl = el.previousElementSibling, um0 = messages[messages.indexOf(m) - 1];
  const liveTick = setInterval(() => {
    const lv = (lastMetrics && lastMetrics.live) || {};
    const ue = userEl && userEl.querySelector(".st-msg__meta");
    if (lv.state === "reading") {
      const tot = lv.prompt_total || lv.prompt_tokens;
      if (ue && um0) ue.textContent = `You · ${timeStr(um0.time)} · reading the prompt` +
        (lv.prompt_read != null && tot ? `: ${fmt(lv.prompt_read)} of ${fmt(tot)} tokens` : tot ? ` (${fmt(tot)} tokens)` : "") +
        (lv.elapsed_s ? ` · ${fmt(lv.elapsed_s, 0)} s` : "");
    } else if (lv.state === "generating") {
      if (ue && um0 && lv.prompt_tokens)
        ue.textContent = `You · ${timeStr(um0.time)} · prompt ${fmt(lv.prompt_tokens)} tokens${lv.prefill_tok_s_mean ? ` read at ${fmt(lv.prefill_tok_s_mean, 1)} tok/s` : ""}`;
      const sp = lv.tok_s != null ? lv.tok_s : lv.tok_s_mean;
      m.liveMeta = `${fmt(lv.generated || 0)} tokens${sp != null ? ` · decode ${fmt(sp, 1)} tok/s` : ""}`;
      const ae = el.querySelector(".meta-text");
      if (ae && !m.meta) ae.textContent = m.liveMeta;
    }
  }, 400);
  const body = requestBody();

  let firstAt = null, thinkStart = null, usage = null, timings = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the engine reported an error");
        if (j.usage) usage = j.usage;
        if (j.timings) timings = j.timings;   // the engine's own speeds: the prompt read and the decode
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  clearInterval(liveTick);
  m.liveMeta = "";
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    // under the answer: the speed the engine measured for reading the prompt and for decoding (as llama.cpp's own page shows them),
    // else this page's own average from the first token
    const wr = timings && timings.predicted_per_second;
    const speeds = wr ? ` · decode ${fmt(wr, 1)} tok/s` : secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : "";
    m.meta = `${fmt(n)} tokens${speeds}${m.stopped ? " · stopped" : ""}` +
             (projectionLoaded() ? (settings.esp ? " · projection on" : " · projection off") : "");
  } else if (m.stopped) {
    m.meta = "Stopped";
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`;
  if (m.limit) m.meta = `${m.meta || ""} · stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`;
  // under YOUR message: the prompt the engine read for it and how fast (the conversation cache's share is not read again)
  const um = messages[messages.indexOf(m) - 1];
  if (timings && um && um.role === "user" && (timings.prompt_n || timings.cache_n)) {
    const read = timings.prompt_n || 0, held = timings.cache_n || 0;
    um.meta = (read ? `prompt ${fmt(read)} tokens read${timings.prompt_per_second ? ` at ${fmt(timings.prompt_per_second, 1)} tok/s` : ""}` : "prompt not re-read") +
              (held ? ` · ${fmt(held)} tokens reused` : "");
    const ue = el.previousElementSibling && el.previousElementSibling.querySelector(".st-msg__meta");
    if (ue) ue.textContent = `You · ${timeStr(um.time)} · ${um.meta}`;
  }
  busy = null;
  setBusy(false);
  // a draft typed while the answer ran was not read (warmDraft skips while busy): read it now, the answer just ended
  if ($("input").value.trim()) scheduleWarm(0);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);
// the draft is read while it is typed: 1.2 s after the last keystroke, 0.3 s after a paste (its input event comes in the same task,
// so it does not restart the timer)
$("input").addEventListener("input", () => { if (!warmPasted) scheduleWarm(1200); });
$("input").addEventListener("paste", () => {
  warmPasted = true;
  setTimeout(() => { warmPasted = false; }, 0);
  scheduleWarm(300);
});

$("new-btn").onclick = () => {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (!messages.length) return;
  const backup = messages;
  messages = [];
  saveChat();
  renderChat();
  toast("info", "New chat", "The last one was cleared.", 6000, {label: "Undo", run: () => { messages = backup; saveChat(); renderChat(); }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${health.images ? " or pictures" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the sampling drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking};
  if (!(+s.temperature > 0)) d.temperature = 0;
  else if (!samplingUntouched(s)) Object.assign(d, {temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  $("o-thinking").textContent = sel ? {none: "answers right away", low: "short", medium: "medium", high: "thorough (default)"}[sel.dataset.v] : "";
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-esp").onclick = () => $("s-esp").setAttribute("aria-checked", String($("s-esp").getAttribute("aria-checked") !== "true"));
$("s-mcp").onclick = () => $("s-mcp").setAttribute("aria-checked", String($("s-mcp").getAttribute("aria-checked") !== "true"));
$("s-share").onclick = () => $("s-share").setAttribute("aria-checked", String($("s-share").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true",
              esp: $("s-esp").getAttribute("aria-checked") === "true",
              mcp: $("s-mcp").getAttribute("aria-checked") === "true"};
  store.set("sampling", settings);
  const share = $("s-share").getAttribute("aria-checked") === "true";
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", "Sampling saved", share ? "Other apps (omp, API clients) use these settings from their next request."
                                               : "Other apps use their own settings again.");
    } catch (e) {
      toast("error", "Saved here, but not for other apps", e.message, 6000);
    }
    return;
  }
  toast("success", "Sampling saved", settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ start
setBusy(false);
renderChat();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, size: $("img-size").value, response_format: "b64_json"};
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  $("img-go").disabled = true;
  const t0 = performance.now();
  $("img-hint").textContent = "Generating\u2026 (a minute or more on a small GPU)";
  try {
    const r = await fetch("v1/images/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      cap.textContent = `${j.size}, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = "Done. Right-click an image to save it.";
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", "No image", String(e.message || e), 8000);
  } finally {
    $("img-go").disabled = false;
  }
};
