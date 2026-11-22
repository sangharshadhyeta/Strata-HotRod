#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    return llama_server(params, argc, argv);
}

int llama_server(common_params & params, int argc, char ** argv) {
    bool is_run_by_cli = (argv == nullptr);

    common_models_handler models_handler;

    // note: router mode also accepts -hf remote-preset, so we need to check that first
    if (!is_run_by_cli && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty();

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // The dynamic atlas should work out of the box, not only for someone who
    // knows to pass two paths by hand. Both default to per-model files under
    // the cache directory (per-model, not one shared name, so switching
    // models doesn't feed one model's co-activation graph into another's
    // atlas). An explicit --expert-atlas-file still wins, same rule as
    // everywhere else here.
    // Same "should work out of the box" rule for the MTP draft model. It is
    // the single largest thing a bare launch was still missing: measured on
    // gemma-4, adding it took decode from 24 to 34 tok/s, and the only reason
    // it needed a flag is that the file has a different name. Unsloth ships it
    // beside the model as mtp-<model>.gguf, so look there and use it if it
    // exists. An explicit -md still wins, and a model with no such sibling is
    // unaffected - this only fills in a path the user would otherwise have to
    // type themselves.
    if (!params.speculative.has_dft() && !params.model.path.empty()) {
        // The draft is NOT named after the target: unsloth ships gemma-4's as
        // mtp-gemma-4-26B-A4B-it-Q8_0.gguf beside a target quantised
        // UD-Q4_K_M, so matching on "mtp-<target filename>" finds nothing
        // (checked - that was this code's first version). Scan the directory
        // for mtp-*.gguf instead and take it only when there is exactly one:
        // with a single candidate the intent is unambiguous, and with several
        // there is a real choice to make that belongs to the user rather than
        // to a guess here.
        const std::filesystem::path mp(params.model.path);
        std::error_code ec;
        std::filesystem::path found;
        int n_found = 0;
        for (const auto & de : std::filesystem::directory_iterator(mp.parent_path(), ec)) {
            const std::string fn = de.path().filename().string();
            if (fn.rfind("mtp-", 0) == 0 && de.path().extension() == ".gguf") {
                found = de.path();
                n_found++;
            }
        }
        if (n_found == 1) {
            params.speculative.draft.mparams.path = found.string();
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            SRV_INF("found an MTP draft model beside the target and enabled speculative decoding: %s "
                    "(pass -md to override, or --spec-type none to disable)\n", found.string().c_str());
        } else if (n_found > 1) {
            SRV_INF("%d mtp-*.gguf files sit beside the target - pass -md to choose one\n", n_found);
        }
    }

    // an -md that names an MTP head, with no --spec-type: use it as one (it was run as a plain draft model: -25%)
    common_speculative_default_type_from_draft(params);

    {
        std::string model_tag = std::filesystem::path(params.model.path).stem().string();
        if (model_tag.empty()) {
            model_tag = "model";
        }
        const std::string cache_dir = fs_get_cache_directory();
        fs_create_directory_with_parents(cache_dir);
        if (params.expert_atlas_file.empty()) {
            params.expert_atlas_file = cache_dir + "expert-atlas-" + model_tag + ".json";
        }
        if (!getenv("GGML_CUDA_MOE_CACHE_COACT_FILE")) {
            const std::string coact = cache_dir + "coact-" + model_tag + ".jsonl";
#if defined(_WIN32)
            _putenv_s("GGML_CUDA_MOE_CACHE_COACT_FILE", coact.c_str());
#else
            setenv("GGML_CUDA_MOE_CACHE_COACT_FILE", coact.c_str(), 1);
#endif
        }
    }

    // Must run after the n_parallel auto-resolution above -
    // common_moe_calibrate() builds a real llama_context_params from
    // params.n_parallel, and -1 ("auto", never resolved before this point)
    // silently wraps to a huge unsigned n_seq_max and crashes the no-alloc
    // probe with "n_seq_max must be <= 256" - found by testing --moe-calibrate
    // together with --model-draft/--spec-type for real, not by inspection.
    //
    // A plain launch with no cached calibration for this model+hardware+
    // context combination, and no explicit -ncmoe of the user's own, should
    // not silently fall back to a safe-but-unoptimized placement - it should
    // calibrate automatically, the same way passing --moe-calibrate would,
    // before serving. This reuses the exact same branch below (temporary
    // status server, calibrate, fall through to normal serving) - the only
    // difference from an explicit --moe-calibrate launch is who set the flag.
    if (!params.moe_calibrate && common_moe_should_auto_calibrate(params)) {
        SRV_WRN("%s", "no cached MoE calibration found for this model+hardware+context+parallelism "
                "combination - running calibration automatically before serving "
                "(pass -ncmoe explicitly, or run --moe-calibrate yourself, to control this)\n");
        params.moe_calibrate = true;
    }

    if (params.moe_calibrate) {
        // Stand up a minimal status server on the normal host:port for the
        // duration of calibration. Without this, the webui's poll of /props
        // gets a bare connection refused - indistinguishable from "the
        // server crashed" - for however long calibration takes (this fork's
        // own first GLM-5.3-Flash calibration took several minutes just for
        // its first candidate, and a full run can take hours). Returning a
        // real HTTP 503 with a status message instead makes the webui show
        // its existing "loading" state (server.svelte.ts already treats any
        // 503 from /props that way) with real progress in the description,
        // rather than the scarier "server unavailable" it shows for a
        // genuine connection failure.
        httplib::Server calib_srv;
        // The real server sets this (server-http.cpp) and the status server did not,
        // which is the entire reason it could not take the port: a server that just
        // exited leaves 8099 in TIME_WAIT, bind fails, and retrying bind_to_port on
        // the same httplib::Server does not recover - so the background retry added
        // earlier spun forever and the progress page never appeared. With
        // SO_REUSEADDR the first bind succeeds and the retry is a fallback rather
        // than the mechanism.
        calib_srv.set_socket_options([](const socket_t sock) {
            httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
        });
        auto status_handler = [](const httplib::Request &, httplib::Response & res) {
            const auto st = common_moe_calibration_status_get_struct();
            nlohmann::json body = {
                {"error", {
                    {"message", common_moe_calibration_status_get()},
                    {"type",    "calibrating"},
                    {"code",    503},
                }},
                // Structured fields for the status page's progress bar - the
                // "error.message" string above carries the same info for any
                // plain client that only reads that (e.g. a future webui
                // integration), this is for one that wants to render it.
                {"calibration", {
                    {"stage",     st.stage},
                    {"elapsed_s", st.elapsed_s},
                    {"done",      st.done},
                    {"total",     st.total}, // 0 = no estimate yet
                    {"eta_s",     st.eta_s}, // -1 = not enough data yet
                    // What each finished candidate settled on. The bar says
                    // the run is alive; this says what it is learning, and
                    // makes a rejection visible as a rejection rather than as
                    // a candidate that merely scored badly.
                    {"decisions", [&] {
                        json arr = json::array();
                        for (const auto & d : st.decisions) {
                            arr.push_back({
                                {"lever",    d.lever},
                                {"value",    d.value},
                                {"result",   d.result},
                                {"accepted", d.accepted},
                                {"chosen",   d.chosen},
                            });
                        }
                        return arr;
                    }()},
                }},
            };
            res.status = 503;
            res.set_content(body.dump(), "application/json; charset=utf-8");
        };
        calib_srv.Get("/props",  status_handler);
        calib_srv.Get("/health", status_handler);
        // The real webui shell (index.html + JS bundle) is served by
        // server_http_context, which isn't stood up in calibrate mode - so a
        // browser hitting "/" here would otherwise get nothing at all to
        // render, not even the code that would poll /props. A small
        // self-contained page instead of nothing: polls /props itself and
        // shows the same status text, no build/asset dependency.
        calib_srv.Get("/", [](const httplib::Request &, httplib::Response & res) {
            static const char * const html =
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Calibrating - llama.cpp</title>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<style>"
                "body{font-family:system-ui,sans-serif;background:#0b0b0c;color:#e5e5e5;"
                "display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0;padding:1.5rem}"
                "main{max-width:38rem;width:100%}"
                "h1{font-size:1.1rem;font-weight:600;display:flex;align-items:center;gap:.6rem;margin:0 0 .9rem}"
                ".spin{width:1rem;height:1rem;border:2px solid #555;border-top-color:#e5e5e5;"
                "border-radius:50%;animation:spin 1s linear infinite;flex:none}"
                "@keyframes spin{to{transform:rotate(360deg)}}"
                "p{color:#a3a3a3;line-height:1.5;font-size:.9rem;margin:.4rem 0}"
                "#stage{color:#e5e5e5;font-weight:500}"
                "#bar-track{background:#232326;border-radius:999px;height:.5rem;overflow:hidden;margin:.7rem 0}"
                "#bar-fill{background:#4ed6a5;height:100%;width:0%;transition:width .4s ease;border-radius:999px}"
                "#stats{display:flex;justify-content:space-between;font-size:.78rem;color:#8a8a8f}"
                "#dec{margin-top:1.2rem;width:100%;border-collapse:collapse;font-size:.78rem}"
                "#dec th{text-align:left;color:#8a8a8f;font-weight:500;padding:.3rem .5rem .3rem 0;"
                "border-bottom:1px solid #232326}"
                "#dec td{padding:.3rem .5rem .3rem 0;border-bottom:1px solid #1b1b1e;color:#c8c8cc}"
                "#dec td.r{color:#4ed6a5}"
                // The middle column carries the value being tried ("-ub 2048",
                // "next layer + prev picks") - the column a reader scans to compare
                // candidates. Left to auto it collapsed to the width of the shortest
                // label and wrapped the longer ones.
                "#dec th:nth-child(1),#dec td:nth-child(1){width:20%}"
                "#dec th:nth-child(2),#dec td:nth-child(2){width:36%;white-space:nowrap}"
                "#dec tr.no td{color:#8a8a8f}"
                "#dec tr.sel td{color:#e9fbf4;background:#12241d;font-weight:600}"
                "#dec tr.sel td:first-child::before{content:'\\2713 ';color:#4ed6a5}"
                "#dec tr.no td.r{color:#d67a7a}"
                "#dec-wrap{display:none}"
                "#dec{margin-top:1.2rem;width:100%;border-collapse:collapse;font-size:.78rem}"
                "#dec th{text-align:left;color:#8a8a8f;font-weight:500;padding:.3rem .5rem .3rem 0;"
                "border-bottom:1px solid #232326}"
                "#dec td{padding:.3rem .5rem .3rem 0;border-bottom:1px solid #1b1b1e;color:#c8c8cc}"
                "#dec tr.no td{color:#8a8a8f}"
                "#dec td.r{color:#4ed6a5}"
                // The middle column carries the value being tried ("-ub 2048",
                // "next layer + prev picks") - the column a reader scans to compare
                // candidates. Left to auto it collapsed to the width of the shortest
                // label and wrapped the longer ones.
                "#dec th:nth-child(1),#dec td:nth-child(1){width:20%}"
                "#dec th:nth-child(2),#dec td:nth-child(2){width:36%;white-space:nowrap}"
                "#dec tr.no td.r{color:#d67a7a}"
                "#dec-wrap{display:none}"
                "</style></head><body><main>"
                "<h1><span class='spin'></span>Calibrating MoE placement</h1>"
                "<p id='stage'>Contacting status endpoint ...</p>"
                "<div id='bar-track'><div id='bar-fill'></div></div>"
                "<div id='stats'><span id='count'></span><span id='eta'></span></div>"
                "<div id='dec-wrap'><table id='dec'><thead><tr><th>Decided</th><th>Tried</th>"
                "<th>Result</th></tr></thead><tbody id='dec-body'></tbody></table></div>"
                "<div id='dec-wrap'><table id='dec'><thead><tr><th>Decided</th><th>Tried</th>"
                "<th>Result</th></tr><tbody id='dec-body'></tbody></table></div>"
                "<p style='font-size:.78rem'>This page refreshes itself every 3 seconds - no action needed. "
                "The progress bar is a rough estimate, not exact - some search stages only get sized once "
                "earlier ones finish.</p>"
                "</main><script>"
                "async function poll(){"
                "try{"
                "const r=await fetch('/props');const j=await r.json();"
                "const c=j.calibration;"
                "if(c){"
                "document.getElementById('stage').textContent=c.stage||'Calibrating ...';"
                "const pct=c.total>0?Math.min(100,Math.round(100*c.done/c.total)):0;"
                "document.getElementById('bar-fill').style.width=pct+'%';"
                "document.getElementById('count').textContent=c.total>0?"
                "('candidate '+c.done+' / ~'+c.total+' ('+pct+'%)'):'';"
                "const es=c.elapsed_s,em=Math.floor(es/60),ss=es%60;"
                "let etaTxt='elapsed '+em+'m'+String(ss).padStart(2,'0')+'s';"
                "if(c.eta_s>=0){const tm=Math.floor(c.eta_s/60),ts=c.eta_s%60;"
                "etaTxt+=' - ~'+tm+'m'+String(ts).padStart(2,'0')+'s left';}"
                "document.getElementById('eta').textContent=etaTxt;"
                "var ds=c.decisions||[];"
                "if(ds.length){"
                "document.getElementById('dec-wrap').style.display='block';"
                "var h='';"
                "for(var i=ds.length-1;i>=0;i--){var d=ds[i];"
                "h+=\"<tr class='\"+(d.chosen?'sel':(d.accepted?'yes':'no'))+\"'><td>\"+d.lever+\"</td><td>\"+d.value+\"</td>\"+"
                "\"<td class='r'>\"+d.result+\"</td></tr>\";}"
                "document.getElementById('dec-body').innerHTML=h;"
                "}"
                "}else if(j.default_generation_settings||j.model_path){"
                // Calibration finished: this same process stopped the status
                // server and started real serving on the same port, so /props
                // is now the ordinary server's. Without this the page kept
                // polling a perfectly healthy server, found no `calibration`
                // key, and sat on "Calibrating ..." forever - the run was done
                // and the tab never showed it.
                "document.getElementById('stage').textContent='Calibration complete - loading the UI ...';"
                "document.getElementById('bar-fill').style.width='100%';"
                "setTimeout(function(){location.reload();},600);"
                "return;"
                "}else{"
                "document.getElementById('stage').textContent="
                "(j.error&&j.error.message)||'Calibrating ...';"
                "}"
                "}catch(e){"
                "document.getElementById('stage').textContent="
                "'Waiting for the status endpoint to come up ...';"
                // Clear the stale bar/count/eta from the last successful poll -
                // otherwise a fetch failure (calibration finished and this
                // process exited, or the page was left open across a restart)
                // leaves an old progress snapshot on screen next to text that
                // contradicts it, reading as "still running" when it is not.
                "document.getElementById('bar-fill').style.width='0%';"
                "document.getElementById('count').textContent='';"
                "document.getElementById('eta').textContent='';"
                "}"
                "}"
                "poll();setInterval(poll,3000);"
                "</script></body></html>";
            res.set_content(html, "text/html; charset=utf-8");
        });

        std::thread calib_http_thread;
        std::atomic<bool> calib_bind_stop{false};
        // Keep retrying in the background for as long as calibration runs,
        // rather than giving up after a fixed window. A restart leaves the
        // previous server's socket held for a few seconds, and a bounded retry
        // loses the race whenever that takes a moment longer than the window -
        // observed directly: a 15s loop expired, and the page then stayed blank
        // for the whole 10-minute run, which is indistinguishable from a hang
        // and is exactly what the retry existed to prevent. A background
        // retry has no such cliff: the page comes up the moment the port frees,
        // whenever that is, and calibration is never blocked waiting for it.
        calib_http_thread = std::thread([&] {
            bool announced_wait = false;
            while (!calib_bind_stop.load(std::memory_order_relaxed)) {
                if (calib_srv.bind_to_port(params.hostname, params.port)) {
                    SRV_INF("--moe-calibrate: status server listening on http://%s:%d/ while calibrating "
                            "(this same process starts real serving on the same port once calibration "
                            "completes)\n", params.hostname.c_str(), params.port);
                    calib_srv.listen_after_bind();
                    return;
                }
                if (!announced_wait) {
                    announced_wait = true;
                    SRV_INF("--moe-calibrate: %s:%d is still held (a previous server shutting down?) - "
                            "retrying in the background; the progress page appears as soon as it frees\n",
                            params.hostname.c_str(), params.port);
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });

        common_moe_calibrate(params);

        calib_bind_stop.store(true, std::memory_order_relaxed);
        calib_srv.stop();
        if (calib_http_thread.joinable()) {
            calib_http_thread.join();
        }

        // Fall through into normal serving instead of exiting - the cache
        // entry common_moe_calibrate() just wrote is picked up automatically
        // a few hundred lines down by common_maybe_autoplace_moe_cpu() (via
        // common_moe_calibration_lookup()), the same path any later launch
        // without --moe-calibrate would use. Clearing the flag here only
        // matters for router mode, which re-reads params per spawned
        // instance - it must not re-enter calibration on every one.
        params.moe_calibrate = false;

        // Build the atlas from the traffic calibration just generated, before
        // serving starts. The candidate subprocesses inherit
        // GGML_CUDA_MOE_CACHE_COACT_FILE (defaulted above, so this needs no
        // flags) and each writes real routing co-activation while it is
        // benchmarked - by the time calibration finishes there is a genuine,
        // model-specific graph sitting there. Without this the first serving
        // session runs with no atlas at all and has to wait for the
        // background regen loop's first interval, which is precisely the
        // "everything is ready except the one thing that needed traffic"
        // gap. Bounded and best-effort: a failure here costs the initial
        // atlas, not the server, and the regen loop will try again anyway.
        if (!params.expert_atlas_file.empty()) {
            const char * coact = getenv("GGML_CUDA_MOE_CACHE_COACT_FILE");
            std::error_code ec;
            const bool have_traffic = coact && std::filesystem::exists(coact, ec) &&
                                      !ec && std::filesystem::file_size(coact, ec) > 0 && !ec;
            if (have_traffic) {
                const std::string script_rel = "scripts/moe-atlas-evolve.py";
                std::string script_path;
                std::vector<std::filesystem::path> candidates = { std::filesystem::path(script_rel) };
                std::error_code exe_ec;
                const auto exe = std::filesystem::read_symlink("/proc/self/exe", exe_ec);
                if (!exe_ec) {
                    const auto bin_dir = exe.parent_path();
                    candidates.push_back(bin_dir / ".." / ".." / script_rel);
                    candidates.push_back(bin_dir / ".." / script_rel);
                    candidates.push_back(bin_dir / script_rel);
                }
                for (const auto & cand : candidates) {
                    std::error_code cec;
                    if (std::filesystem::exists(cand, cec) && !cec) {
                        script_path = cand.string();
                        break;
                    }
                }
                if (!script_path.empty()) {
                    const std::filesystem::path atlas_path(params.expert_atlas_file);
                    const auto state_path = atlas_path.parent_path() / "atlas-evolve-state.npz";
                    SRV_INF("%s", "building the initial expert atlas from the traffic calibration just "
                            "generated ...\n");
                    const std::string cmd = "python3 '" + script_path + "'"
                        " --coact '" + std::string(coact) + "'"
                        " --state '" + state_path.string() + "'"
                        " --out '"   + params.expert_atlas_file + "' >/dev/null 2>&1";
                    const int rc = std::system(cmd.c_str());
                    if (rc == 0) {
                        SRV_INF("initial expert atlas written to '%s'\n", params.expert_atlas_file.c_str());
                    } else {
                        SRV_WRN("initial atlas build exited %d - the background regen loop will retry\n", rc);
                    }
                }
            }
        }

        SRV_INF("%s", "--moe-calibrate: calibration complete, starting real serving now on the same port "
                "with the placement just measured\n");
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    // For a local file get_name() is the path. Inserting that as an alias made it
    // an "explicit" alias, so the server never looked at the model's own
    // general.name - a split model was named after its first shard. Leave the
    // alias empty then and let server_context name the model after loading it.
    if (params.model_alias.empty() && !model_name.empty() && model_name != params.model.path) {
        params.model_alias.insert(model_name);
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_child child; // only used in non-router mode
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/experts",                  ex_wrapper(routes.get_experts));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.post("/moe-tuning",               ex_wrapper(routes.post_moe_tuning));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child, see server-stream.h
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = server_stream_make_get_handler();
        streams_lookup_h = server_stream_make_lookup_handler();
        stream_delete_h  = server_stream_make_delete_handler();
    }
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get_h));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS is set to allow all origins ('*') and no API key is set\n");
        SRV_WRN("%s", "this can be a security risk (cross-origin attacks)\n");
        SRV_WRN("%s", "more info: https://github.com/ggml-org/llama.cpp/pull/25655\n");
        SRV_WRN("%s", "-----------------\n");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    std::vector<std::string> warn_names;
    if (is_router_server) {
        warn_names.push_back("router mode");
    }

    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return 1;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("built-in tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (warn_names.size() > 0) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "the following feature(s) are enabled:\n");
        for (const auto & name : warn_names) {
            SRV_WRN("    %s\n", name.c_str());
        }
        SRV_WRN("%s", "do not expose the server to untrusted environments\n");
        SRV_WRN("%s", "-----------------\n");
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server && !is_run_by_cli) {
        // single-model mode (NOT spawned by router)
        // if this is invoked by CLI, model downloading should be already handled
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

        clean_up = [&models_routes, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            mcp_mgr.shutdown();
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            mcp_mgr.shutdown();
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            ctx_http.stop();
            // terminate() itself doesn't persist the token-frequency
            // histogram (see server-token-freq.h) - without this, every
            // restart loses any counts recorded since the last periodic
            // flush, which for the "restart to regenerate the FR-Spec
            // sidecar" workflow this exists for could be most or all of a
            // session's traffic.
            ctx_server.flush_token_freq();
            ctx_server.terminate();
            mcp_mgr.shutdown();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            mcp_mgr.shutdown();
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // register signal handler if not running by CLI
    if (!is_run_by_cli) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = signal_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
        sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    SRV_INF("listening on %s\n", ctx_http.listening_address.c_str());

    // TODO: remove this in the future
    // check the string to also handle the .sock case
    if (string_ends_with(ctx_http.listening_address, ":8080")) {
        SRV_WRN("%s", "NOTICE: server default port will be changed to :9931 in a future release\n");
        SRV_WRN("%s", "        ref: https://github.com/ggml-org/llama.cpp/pull/26508\n");
    }

    if (is_router_server) {
        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
