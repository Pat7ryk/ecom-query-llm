# N1: Qwen2.5-14B-Instruct QLoRA results

> Status: N1 three-seed training and standard full-test evaluation are complete. The final N1-vs-E2a claim remains pending because the three E2a adapters have not yet been rescored with the same double-quantization protocol, and clean/merged, bootstrap, and ranking analyses are not included in the returned artifact.

## 1. Experiment identity

| Item | Recorded value |
| --- | --- |
| Git commit | `b6db723e7547a769385e297318b4de4214573628` |
| Base model | `Qwen/Qwen2.5-14B-Instruct` |
| Model commit | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` |
| Data manifest SHA256 | `52dd97a7cd55301fdf5ee77039144724d59e183e7384c8a105daea6e4ace3fbb` |
| Source archive SHA256 | `e11cf0a7f0d69e3a01d898a5bd875bd3edef4688e5654828a2ba9fe49d25420e` |
| LlamaFactory | `0.9.5` |
| PyTorch / CUDA / driver | `2.15.0 / 13.2 / 595.71.05` |
| GPU | 1 x NVIDIA GeForce RTX 5090, 32,607 MiB |
| Evaluation quantization | bitsandbytes 4-bit NF4, bf16 compute, double quantization enabled |

The source archive was `N1_results.tar.gz`. It was checked for path traversal, obvious credentials, row counts, completion signals, and protocol consistency before these compact artifacts were imported.

## 2. Configuration and deviations

The intended research variable is the base-model scale change from Qwen2.5-7B-Instruct to Qwen2.5-14B-Instruct. The actual server used an RTX 5090 32 GB rather than the planned A100 80 GB. To fit memory, all three runs used micro-batch 1 and gradient accumulation 32, preserving the registered effective batch size of 32. This hardware and micro-batch deviation must be disclosed in the final report.

The returned archive does not contain the smoke-test completion marker or the N0 gate/waiver artifact. The orchestration script requires those checks before a fresh formal run, but their original evidence should be retained separately for a complete audit bundle.

## 3. Training record

| Seed | Best checkpoint | Eval loss | Elapsed training wrapper time | Peak GPU memory | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 625 | 0.061404 | 8,126 s (2.26 h) | 31,478 MiB | completed |
| 43 | 400 | 0.061902 | 8,480 s (2.36 h) | 28,924 MiB | completed |
| 44 | 625 | 0.061828 | 8,816 s (2.45 h) | 27,514 MiB | completed |

Total recorded wrapper time was 25,422 seconds (7.06 GPU-hours). Each training log completed 625 optimization steps and evaluated 1,000 validation examples. Warnings were limited to the LlamaFactory upcast-layernorm recommendation, tokenizer special-token alignment, Triton default-fusion notice, and the absence of an `eval_accuracy` plot; no traceback, OOM, NaN, or failed run was found.

## 4. N1 full-test results

The prior bias was selected independently on the 6,000-row tune split for each seed, then applied once to the 10,000-row test split.

| Seed | Tune Macro F1 raw -> adjusted | Test raw accuracy | Test raw Macro F1 | Test adjusted accuracy | Test adjusted Macro F1 | Selected bias `(E,S,C,I)` |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.532988 -> 0.568915 | 0.7464 | 0.542281 | 0.7330 | 0.564195 | `(0.00, 0.50, 1.00, 0.75)` |
| 43 | 0.500103 -> 0.557519 | 0.7457 | 0.523131 | 0.7426 | 0.564639 | `(0.00, 0.25, 1.25, 0.00)` |
| 44 | 0.549229 -> 0.563956 | 0.7489 | 0.554430 | 0.7491 | 0.561737 | `(0.00, 0.00, 0.25, 0.00)` |
| Mean | - | **0.7470** | **0.539947** | **0.741567** | **0.563523** | - |

The adjusted Macro F1 range is **0.561737-0.564639**. All format-valid rates are 1.0. Full per-class precision/recall/F1, class distributions, confusion matrices, and bias curves are stored in the three `n1-seed*-prior.json` files beside this report.

## 5. Comparison boundary

For orientation only, the repository's historical E2a reports have adjusted Macro F1 values 0.549287, 0.553800, and 0.540541 (range 0.540541-0.553800). The current N1 range is higher and non-overlapping, with a mean advantage of about 0.01565. This is **not yet the formal fair comparison** because those E2a scores were generated under the older evaluation setup without the newly required double-quantization alignment.

Do not report "stable improvement" as the final conclusion until all of the following are complete:

1. Rescore the three E2a adapters with NF4, bf16 compute, and double quantization enabled.
2. Produce the full/clean/merged comparison and document the independent clean-label source.
3. Run query-level bootstrap confidence intervals and ranking analyses.
4. Archive the N0 gate result or the team-approved waiver and the N1 smoke-test evidence.

Current conclusion: **N1 full-test results are complete and promising; the confirmatory N1-vs-E2a comparison remains incomplete.**
