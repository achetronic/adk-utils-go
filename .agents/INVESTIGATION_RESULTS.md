# Investigation Results: Compaction Behaviour Analysis

**Date**: February 2026  
**Test files**: `compaction_simulation_test.go` in `plugin/contextguard`  
**Context**: Full analysis of the compaction system, covering the original split-with-recent-tail approach, the migration to Crush-style full-summary mode, and the calibrated heuristic that closes the timing gap.

---

## Part 1: Original approach — split with recent tail (historical)

> **Note**: This section documents the original behaviour before Proposal 5. The threshold strategy no longer keeps a recent tail — it always summarizes everything. These results explain *why* the recent tail approach was abandoned.

### Retry Rounds vs `kube-3rounds / 8k ctx`

| Scenario | Attempts | Tokens Before | Tokens After | Reduction | Contents | Fits? |
|---|---|---|---|---|---|---|
| kube-3r/8k/attempts=1 | 1 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=3 | 3 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=10 | 10 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=20 | 20 | 24,126 | 8,068 | 66.6% | 24→9 | NO |

**Conclusion**: More retry rounds provided zero benefit. The 9 remaining recent messages (~8k tokens from 3 tool_call/tool_response pairs) were untouched by compaction. This was the fundamental flaw: the recent tail is irreducible.

### Kube conversations across context sizes (with recent tail)

| Scenario | Tokens After | Fits? |
|---|---|---|
| kube-3r / 8k | 8,068 | **NO** |
| kube-5r / 8k | 8,068 | **NO** |
| kube-10r / 8k | 8,068 | **NO** |
| kube-20r / 8k | 8,068 | **NO** |
| kube-3r / 32k | 24,126 (no compaction) | YES |
| kube-10r / 32k | 8,068 | YES |
| kube-20r / 128k | 24,223 | YES |

**Key insight**: 8k context was structurally insufficient for kube conversations because a single round's tool responses (~8k tokens) saturated the entire window. The convergence to exactly 8,068 tokens across all 8k scenarios proved the bottleneck was in the recent tail, not in the summarization.

### Why the recent tail was the wrong approach

The original design kept ~20% of the context window as verbatim recent messages. This meant:
1. A single large tool response in the recent window could exceed the budget alone.
2. Retry loops only re-compacted already-compacted older content — the recent window was never touched.
3. The system failed deterministically for any context window where one round of tool responses exceeded the threshold.

---

## Part 2: Current approach — full summary (Crush-style)

The threshold strategy now summarizes the **entire** conversation. After compaction, `req.Contents` is exactly:

```
[summary message] + [continuation message]
```

### Why there's no need for retry

The summary is generated with `maxOutputTokens = buffer × 0.50`:

| Context window | Buffer | maxOutputTokens | Summary ~tokens | Threshold |
|---|---|---|---|---|
| 200k | 20k | 10k | ~10k | 180k |
| 128k | 25.6k | 12.8k | ~12.8k | 102.4k |
| 32k | 6.4k | 3.2k | ~3.2k | 25.6k |
| 4k | 800 | 400 | ~400 | 3.2k |

Post-compaction result is always `summary (~maxOutputTokens) + continuation (~100 tokens)`. This is orders of magnitude below the threshold in all cases. Even if the LLM slightly exceeds `maxOutputTokens` (some providers don't enforce it strictly), the result is still tiny compared to the threshold. A retry loop would never fire.

### Summarization input is safe too

The summarization prompt renders tool results as `[tool X returned a result]` — raw payloads are not included. So even a conversation with 500k of JSON tool responses produces a summarization input of manageable size (tool calls listed as one-liners).

---

## Part 3: Timing gap — calibrated heuristic

### The problem

ContextGuard uses `BeforeModelCallback` (before the LLM call) to check tokens, but gets real token counts via `AfterModelCallback` (after the LLM responds). Between callbacks, tool results may be appended to the conversation:

```
Step N:
  BeforeModelCallback → check tokens → LLM call → AfterModelCallback (persist PromptTokenCount)
  → Tool executes → result appended to session

Step N+1:
  preprocess (builds req with new tool results) → BeforeModelCallback → reads stale PromptTokenCount from N
```

If the tool returned a massive response, `req` at step N+1 is larger than what was measured at step N. Using the stale `PromptTokenCount` directly would miss the growth.

Crush CLI doesn't have this problem because it checks *after* each step (with fresh tokens) and can stop *before* the next call. ADK only offers before/after callbacks.

### The solution

Both a real token count and a heuristic estimate are persisted at each step:

| Callback | Persisted | Key |
|---|---|---|
| `BeforeModelCallback` | `estimateTokens(req)` after compaction | `__context_guard_last_heuristic_{agent}` |
| `AfterModelCallback` | `PromptTokenCount` from provider | `__context_guard_real_tokens_{agent}` |

On the next `BeforeModelCallback`, `tokenCount()` computes:

```
currentHeuristic = estimateTokens(req)          // on the current, possibly-grown request
correction = max(1.0, real / lastHeuristic)     // how much len/4 underestimated last time
calibrated = currentHeuristic × correction      // scale current heuristic to real-token space
return max(real, calibrated)
```

**Why this works**:
- `currentHeuristic` tracks growth (tool results added → more text → higher heuristic).
- `correction` translates heuristic tokens into real tokens using the last known ratio.
- `max(real, calibrated)` ensures we never undercount — if the request didn't grow, `real` dominates; if it grew, `calibrated` dominates.
- The correction factor is floored at 1.0 to prevent the calibrated value from being smaller than the raw heuristic.
- If no real tokens exist (first turn), a conservative default factor of 1.5 is applied.

### Proof

**`TestTimingGap_CalibratedHeuristicPreventsOverflow`** (200k context window, 180k threshold):

| | Value | Triggers compaction? |
|---|---|---|
| Stale real tokens (old approach) | 140,000 | NO (140k < 180k) |
| Calibrated estimate (new approach) | 180,018 | YES (180k ≥ 180k) |

The tool response added ~40k real tokens between steps. The old approach missed it entirely. The calibrated estimate detected it.

**`TestTimingGap_MassiveToolResponse`** (200k window, 400k-char tool response):

| | Value |
|---|---|
| Stale real tokens | 100,000 |
| Heuristic on current req | 150,008 |
| Correction factor | 2.0 |
| Calibrated estimate | 300,016 |

Compaction fires correctly at 300k, well above the 180k threshold.

---

## Overall Architecture Summary

```
BeforeModelCallback (step N)
├── tokenCount(ctx, req)
│   ├── currentHeuristic = estimateTokens(req)      ← on the FULL current request
│   ├── real = loadRealTokens(ctx)                   ← from step N-1's AfterModel
│   ├── lastHeuristic = loadLastHeuristic(ctx)       ← from step N-1's BeforeModel
│   ├── correction = max(1.0, real / lastHeuristic)
│   ├── calibrated = currentHeuristic × correction
│   └── return max(real, calibrated)
├── if totalTokens < threshold → pass through
├── if totalTokens ≥ threshold:
│   ├── summarize(entire conversation)               ← LLM call with structured prompt
│   ├── replaceSummary(req, summary, nil)             ← req = [summary]
│   ├── injectContinuation(req, userContent)          ← req = [summary, continuation]
│   └── persistSummary(ctx, summary, totalTokens)
└── persistLastHeuristic(ctx, estimateTokens(req))    ← for next step's calibration

LLM call (step N) → response with PromptTokenCount

AfterModelCallback (step N)
└── persistRealTokens(ctx, PromptTokenCount)          ← for next step's calibration

Tool execution → results appended to session → loop back to step N+1
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Store only `PromptTokenCount`, not sum with `CandidatesTokenCount` | `PromptTokenCount` already includes the full conversation. The output tokens become part of the next prompt automatically. |
| Filter `resp.Partial` in `AfterModelCallback` | ADK calls the callback for every streaming chunk. Only the final response carries `UsageMetadata`. |
| Always full summary (no recent tail) | Eliminates the irreducible recent tail problem. Post-compaction size is always tiny and predictable. |
| Calibrated heuristic instead of raw `max(real, heuristic)` | Raw heuristic underestimates by 2-3× for structured content. The correction factor from the previous call bridges this gap. |
| Correction factor floored at 1.0 | Prevents the heuristic from being *reduced* when real tokens are lower (e.g., after compaction). |
| Default factor 1.5 for first turn | Conservative — triggers compaction slightly early rather than risking overflow on the very first turn without calibration data. |
| Continuation message includes original user request | Prevents the agent from asking the user to repeat themselves after compaction. |
| Todo preservation in summary prompt | Maintains task tracking state across compaction boundaries. |
