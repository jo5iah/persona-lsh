# experiments: persona-lsh on the rig

The runs beyond the published demo: more models, seven traits, drawn questions and framings, local judges,
digest sizes and bit codes. Moved here from the agent-harness repo on 2026-11-19; their earlier history is in
that repo's `notebooks/persona-lsh*.ipynb` and `docs/persona-lsh-notes.md`.

| File | What |
|---|---|
| `persona-lsh.ipynb` | Capture, judging and classification, on the rig (gpu kernel). Resumable |
| `qwen2_5-lsh.ipynb` | Hardware test: Qwen2.5-7B alone, in bf16 then fp16. Before capture it times and profiles the batched pass with the first layers on card 2, 1, then 0 (section 3a). Why: agent-harness `ops/gpurig2/history.md` |
| `persona-lsh-results.ipynb` | Interpretation and figures; section 11 is the write-up of each run. Needs torch for sections 7 and 9c (CPU is enough) |
| `notes.md` | Where things stand, what goes into the README, the rig runbook, copying data off the rig |

The research plan is [`../PLAN.md`](../PLAN.md). The personality series is in [`../personality/`](../personality/).

## What stays in agent-harness

The notebooks import [agent-harness](https://github.com/jo5iah/agent-harness) as the installed `harness` package
(`pip install -e ~/agent-harness` in the kernel's env) and find its clone through `harness.__file__`:

| There | Used for |
|---|---|
| `harness.load_model_profile`, `harness.structured` | the local judges |
| `configs/rig.yaml` | the rig's Ollama instances and model settings |
| `ops/gpurig/gpu-telemetry.sh`, `gpu_step_load.py`, `pull-hf-models.py` | telemetry, step loads, model downloads |
| `ops/gpurig2/` | the rig's setup, power cap (`gpu-power.sh`) and history |
| `lsh-models.md` | which models, and why they fit the cards |

## Running

On the rig: `git pull` in `~/persona-lsh` and `~/agent-harness`, open `experiments/persona-lsh.ipynb` with the gpu
kernel and run all, then `experiments/persona-lsh-results.ipynb`. `notes.md` section 3 has the full runbook.
Outputs go to `~/persona-lsh-output/<experiment>/` and telemetry to `~/gpu-telemetry/`, outside both repos.
