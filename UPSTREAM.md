# What comes from whom

This repository contains only files written or changed for solid.cpp and Strata-HotRod. Everything else in those projects is
someone else's work and is not copied here.

| Base project | Owner | Licence | Used for |
|---|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) | Niko1221 and contributors | MIT | The engine, server and installer that HotRod modifies (`hotrod-qwen/`, `hotrod-deepseek/`) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | ggml-org and contributors | MIT | The base of solid.cpp (`solid-cpp/`) |
| maya-ds4 (Project Maya) | Project Maya | see its repository | The DeepSeek engine (`strata-dsv41`) that the DeepSeek port builds on |

## Rebuilding

Each folder keeps the paths its files have in the base project. To rebuild, check out the base project at the version the
notes name (Strata 0.2.7.1 for the DeepSeek port; the Strata commit named in `hotrod-qwen/SPEC.md` for the Qwen work; llama.cpp
`1f368f354`, 13 August 2026, for solid.cpp) and copy these files over it.

## Files here that are not original work

`hotrod-qwen/` and `hotrod-deepseek/` include some upstream files that were edited by the HotRod commits (for example
`setup.py`, `serve/server.py`). They appear because they were changed, and they still contain upstream's code.
