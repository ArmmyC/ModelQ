# ADR 0040: The Evaluation Keeps One Model on the Device at a Time

- Status: Accepted (Task 62). The numbers below ran on Modal. The record is [`adr-0040-evaluation-offload.json`](../validation/adr-0040-evaluation-offload.json).
- Date: 2026-10-11
- Scope: `modelq eval` runs the original model and each container's model over the same windows. Each model now waits on the CPU except while it runs, so only one copy of the weights occupies the device at a time. The numbers do not change.
- Builds on: [ADR 0035](0035-evaluating-every-output.md) (evaluating every output) and [ADR 0038](0038-awq-sharded-checkpoints.md), whose 3B evaluation ran out of GPU memory.

## Context

`modelq eval` placed two full F32 copies of the model on the device at once: the original and the quantized one. For Qwen2.5-3B that is about 24.8 GB, more than the 22 GiB of an L4. The 3B evaluation in ADR 0038 failed while moving the second copy onto the card.

## Decisions

1. **Offloading.** The original and each quantized model move to the device only for their own forward pass on each window, and back to the CPU afterwards. Only one model's weights occupy the device at a time. The per-window logits stay on the device.
2. **Substitution is unchanged.** Each container is substituted into its model on the device, as before, and the model then moves back to the CPU.
3. **The metrics are unchanged.** Each forward pass runs on the same values. A move between devices copies F32 values exactly.
4. **Peak memory is recorded.** The report records `peak_device_memory_bytes` for CUDA runs.
5. **Only CUDA offloads.** On the CPU nothing moves.

## Verification

- **0.5B, AWQ INT4, full split, Modal L4.** The quantized perplexity is 16.05708635670797, exactly ADR 0031's anchor (difference 0.0). The original perplexity is 13.069886129532039. Peak device memory is 5.78 GB, and the evaluation took 494 s.
- **Qwen2.5-3B, AWQ INT4, Modal L4.** On 20 windows: original 7.834003499036884, quantized 11.701691859138526, peak device memory 16.26 GB, 634 s. On the full split of 146 windows: original 8.030300985667864, quantized 12.065450350922525, a relative increase of +50.2%, peak device memory 16.26 GB, 3206 s.
- **Local.** The 34 quality-evaluation tests pass on the CPU.

## What this does not show

- **Time.** Moving the models dominates the cost. The 0.5B evaluation took 494 s, and the 3B full split took 3206 s.
- **Host memory.** Both F32 copies stay in host RAM, about 25 GB for the 3B model. The Modal container had 48 GB.
- **Larger models.** A model much larger than 3B needs more host RAM and more time.
- **The 3B quality result.** The 3B quantized model loses 50% perplexity on the full split. The 0.5B model loses 23%. The evaluation itself is consistent, since the 0.5B anchor reproduces exactly. The cause is not investigated. The 3B validation metrics (a maximum MSE of 5.7e-1, against 2.7e-4 for 0.5B) suggest large rescaled weights, but that is a guess until it is checked.

## Consequences

- `modelq eval` runs a 3B model on a 22 GiB card, with time as the cost.
- The evaluation holds two F32 copies of the model in host RAM.
