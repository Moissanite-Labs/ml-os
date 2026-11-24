# Per-request LoRA, 8 slots, 2 base-shaped outputs: root-cause notes

Read-only investigation. No server or GPU process was started. Paths: `S` = `m1/llama.cpp-src/tools/server`, `L` = `m1/llama.cpp-src/src` (tree HEAD `8af68d486a94...`, matches the build).

**Bottom line:** this is not batching semantics, not a harness payload error, and not a slot-level LoRA race. Slots 1, 2 and 3 were decoded together in the same batch at every step, so they had the same adapter state. Why slot 1 took the adapter-shaped branch and slots 2 and 3 the base-shaped one is undetermined: two explanations below the server remain live.

## 1. What happened
- **Payload:** identical for all 8 requests. `gpu_workload_v1.py:41` defines `THREE_ACTIVE`, `:105-109` builds the body, `:193` maps it over `range(8)`; only `id_slot` differs. The file's sha256 `7e8ca193...` matches the run packet.
- **Slot mapping:** each request landed in the slot it asked for, e.g. `0.05.249.901 I slot select: id  2 | task -1 | selected slot by id (2)`.
- **Event order:**
  - `0.05.249.954 ... id  2 | task 134 | processing task`, then `.967` id 1 (task 133), `.975` id 3 (task 135).
  - `0.05.268.986 ... id  0 ... selected slot by id (0)`, launched at `0.05.329.335` (task 137); slots 4-7 (tasks 138-141) by `.416`.
  - Id 136 is the internal NEXT_RESPONSE task (`S/server-context.cpp:3541-3543`).
- **Lockstep:** slots 1-3 advance together throughout: `0.08.512.953 ... id  1 | task 133 | n_gen =    106`, `0.08.513.421 ... id  2 | task 134 | n_gen =    106`. All eight end with `graphs reused =        250`.
- **"2 of 8, slots 2 and 3":** supported only for the stored 80-character `text_head`. Slots 2 and 3 equal `w0_single`; the other six equal `triple_lora_single`. The heads diverge at character 2, right after `"\n\n"`.
- **"Byte-identical output":** not supported. v1 kept 80 characters, and the baseline was 64 tokens against 128.
- **"Prefilled as their own batch":** true only for the first 55 of 59 prompt tokens (inferred). With default checkpoints on this hybrid model the last 4 tokens are split off (`S/server-context.cpp:4332-4345, 4461-4475`) and were batched with the other five slots' prompts. Every tc5 8-slot round shows the same early group of 1-6 slots.

## 2. Mechanism
- **Storage:** the `lora` field is parsed to `task.params.lora` (`S/server-schema.cpp:231-239`, `S/server-common.cpp:318-329`, `S/server-task.h:97`). An empty list falls back to the global scales (`S/server-context.cpp:1894-1897`). Launch copies it to `slot.lora` and `slot.prompt.lora` (`:1899-1912, 1962`).
- **Comparison:** `can_batch_with` requires `are_lora_equal` (`:540-557`, `S/server-common.cpp:331-344`). The first generating slot becomes `slot_batched`; mismatching generating slots (`:3814-3819`) and prompt slots (`:3957-3959`) are skipped that iteration.
- **Different configs:** they are serialized, lowest slot id first, never merged or downgraded. This is documented at `S/README.md:618`.
- **Application:** once per batch, from `slot_batched->lora` (`:3576-3583`), via `common/common.cpp:1679-1689` and `L/llama-context.cpp:1767-1783`. An unchanged set is a no-op (`:1785-1809`). A change forces a re-reserve that resets cached graphs (`:797-824`, `:2259`), so a stale graph cannot be reused (`:1840`, `L/llama-graph.h:968`).
- **In the graph:** the delta is added for every row (`L/llama-graph.cpp:1826-1842`).
- **Arrival mid-decode:** the task is declined while a decode runs (`S/server-context.cpp:2949-2954`, `:4644-4649`), requeued at the front (`S/server-queue.cpp:258-262`), and launched before the next batch.
- **Slot reuse and prompt cache:**
  - A LoRA change clears the slot's tokens (`S/server-context.cpp:1903-1907`).
  - Cache load and slot selection skip states made under another config (`S/server-slot-sched.cpp:169, 182-185, 376-380`).
  - `--no-cache-idle-slots` only disables `S/server-context.cpp:3030-3046`.
  - `cache_prompt:false` forces a full re-evaluation (`:4148-4151`); the log shows `/    59 tokens`.
- **These adapters:** each is one rank-8 pair on `blk.31.ffn_up.weight` (GGUF headers). The last layer's FFN runs only on output rows (`L/models/qwen35.cpp:182-185, 199`), so no cached state depends on LoRA; only each step's logits do.

## 3. Which explanation
- **(a) Batching semantics: no.** All 8 configs were equal, and mismatched configs are serialized, not served base output.
- **(c) Harness payload: no.** The harness's evidence is weak, though (80 characters, no tokens, no echo).
- **(b) Slot-level race or stale state: not supported** (medium-high confidence; I did not read all 6.6k lines).
  - Slots 1-3 are consecutive sequence ids, so the sequential split puts them in one micro-batch (`L/llama-batch.cpp:692-695`, `L/llama-memory-hybrid.cpp:83-90`).
  - The reuse count of 125 for the round fits exactly one unchanged adapter set during generation.
- **(d) Still live, roughly even:**
  - **d1:** adapters applied on all rows, and a near-tied second token flipped through row-dependent GPU numerics.
  - **d2:** a defect below the server (ggml-cuda kernel, graph, or hybrid memory) drops or alters the delta for output rows 1 and 2 of the 3-output mixed micro-batch.
- **Separator:** per-token logprobs at the divergent token.

## 4. tc5
- **How adapters were turned on:** per-request field only (`tc5_replay.py:40, 109-112`). The W0 arm sends `lora: []`; the server starts with scales 0.0 (`:185-190`); there is no POST `/lora-adapters`.
- **8-slot runs:** arms never overlap and every request in an arm carries the same config, so each stack batch ran with the adapter set applied (inferred).
  - Log support: the first batch after an arm change is about 3 ms slower, e.g. q8_0 8-slot 30.1-32.1 ms against 27.2-27.4 ms.
  - Per-row effect is unverified: the 8-slot trials in `q8_0.json` and `q4_k_m.json` hold no content hashes. The hashing code in the script postdates the runs (script 16:43, results 16:33 and 16:34).
  - Throughput cannot reveal it either: measured overhead is -0.66% and +0.48%, which is noise for adapters this small.
- **Single-slot:** not affected. Stack and W0 hashes differ in all trials, one distinct output per arm. The first 120 characters are identical between arms, so classifiers that compare heads would miss it.

## 5. Experiment (not run; about 80 s of GPU)
- **Server:** the exact v1 flags, one start.
- **Every request:** the v1 body plus `"n_predict":16, "n_probs":5, "post_sampling_probs":false, "return_tokens":true`.
- **Record per request:** requested slot, response `id_slot`, `generation_settings.lora`, `tokens`, sha256 of `content`, `completion_probabilities`, `timings`, send and receive times, and the matching task id from the log.
- **Sequence:**
  1. References: one W0 and one stack request on slot 0. Take t* as the first differing token, and L0 and L1 as the top-5 logprobs there.
  2. Controls: 8 sequential stack requests (slots 0-7); 8 concurrent W0; 4 W0 plus 4 stack concurrent. A payload echo mismatch or a wrong class in the sequential or W0 controls indicates a harness or parse fault.
  3. Main: 60 rounds of 8 concurrent stack requests, with slot order permuted per round.
  4. Probes: 3 concurrent stack requests on slots 1-3 only; staggered rounds (k requests, 30 ms pause, the rest, for k = 1 to 4).
- **Decision rule:** set ε to 10 times the largest logprob deviation seen in the controls.
  - Base-branch row with logprobs at t* within ε of L1, and a top-2 margin in L1 no larger than ε: d1.
  - Within ε of L0 while a row in the same batch matches L1: d2.
  - Neither: state corruption.
- **Rate:** report failures over 480 requests with a Wilson 95% interval, and failing rounds over 60 with an exact interval (0 of 60 gives an upper bound of about 5%).
- **Readback:** GET `/lora-adapters` shows global scales only. GET `/slots` gives `params.lora` and `id_task` (`S/server-context.cpp:823-840`). The response's `generation_settings.lora` is the parsed request (`S/server-task.cpp:37-40, 342`). Nothing exposes the set applied in the context.

## Verified, inferred, unknown
- **Verified:** payload code; slot mapping; the 80-character matches; the log timeline; all source citations; adapter tensor lists; tc5 fields and timings.
- **Inferred:** micro-batch composition (default verbosity logs no batch contents); divergence at the second token (the vocabulary has `"\n\n"` as token 271, but no tokens were stored); constant adapter state during generation.
- **Unknown:** text beyond 80 characters; d1 versus d2; the failure rate (one observation); ggml-cuda kernels were not audited.

Scratch files in this folder: `parse_server_log.py`, `gguf_meta_peek.py`, `tc5_q8_0_timeline.txt`.
