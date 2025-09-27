# Initial GRPO Roadmap

## Objectives
- Build a minimal PyTorch-based codebase to experiment with Group Relative Policy Optimization (GRPO).
- Keep the stack dependency-light to understand each training component.
- Provide hooks for evaluating language-model reward signals and KL-control heuristics.

## Proposed Project Layout
- `src/grpo/` – core algorithm implementation (policy, value, sampler, loss).
- `src/grpo/data/` – prompt and response datasets plus sampling utilities.
- `src/grpo/models/` – model wrappers for policy and reference models.
- `src/grpo/training/` – training loop orchestration, gradient accumulation, logging.
- `src/grpo/utils/` – shared helpers (configuration, masking, scheduling, metrics).
- `experiments/` – notebooks or scripts for running specific training setups.
- `tests/` – unit and integration tests.

## Near-Term Milestones
1. Define configuration schema and CLI entry point for experiments.
2. Implement token masking and reward aggregation utilities required by GRPO.
3. Write a reference policy/value model wrapper with minimal transformer backbone.
4. Draft the GRPO loss computation with KL penalty and group-relative advantages.
5. Assemble a training loop supporting batched rollouts and logging/metrics.

## Open Questions
- Which tokenizer/model family should the first implementation target?
- What evaluation harness (if any) should accompany the base project?
- How will rewards be generated for early experiments (rule-based vs model-based)?

