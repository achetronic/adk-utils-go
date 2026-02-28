# Investigation Results: Compaction Behaviour Analysis

**Date**: February 2026  
**Test files**: `compaction_strategy_singleshot_test.go`, `compaction_strategy_multiturn_test.go` in `plugin/contextguard`  
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
- If no real tokens exist (first turn), a conservative default factor of 2.0 is applied.

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
| Correction factor capped at 5.0 | Prevents a single anomalous turn (JSON-heavy tool schemas) from producing a disproportionate correction that persists. |
| Default factor 1.5 for first turn | Conservative — triggers compaction slightly early rather than risking overflow on the very first turn without calibration data. |
| Continuation message includes original user request | Prevents the agent from asking the user to repeat themselves after compaction. |
| Todo preservation in summary prompt | Maintains task tracking state across compaction boundaries. |
| Truncate conversation before summarization | Prevents the summarization prompt itself from exceeding the summarizer LLM's context window. |
| Fallback summary on LLM failure | If summarization fails, a mechanical summary (first 200 chars of each message) is used instead of passing through the bloated request. |

---

## Part 4: Hardening — production-readiness improvements

### Problem: three failure modes identified in production testing

After deploying the initial Crush-style compaction (Proposal 5), three architectural weaknesses were identified through both production observation and exhaustive stress testing:

1. **Uncapped correction factor**: In a JSON-heavy tool schema turn, `correction = realTokens / lastHeuristic` could reach 8-10x. This persisted across turns, causing premature compaction on every subsequent message until the next reset.

2. **Summarizer context overflow**: When a conversation accumulated 300k+ chars of tool responses (e.g., `kubectl get pods -o json` on a large cluster), the `summarize()` call sent the entire conversation as a prompt to the summarizer LLM. The summarization prompt itself exceeded the LLM's context window, causing an API error.

3. **Pass-through on failure**: When `summarize()` failed, `Compact()` returned an error. `beforeModel` logged a warning and passed through the original bloated request to the provider, which then rejected it for exceeding the context window. The user saw an error instead of a degraded-but-working experience.

### Solutions implemented

#### Correction factor cap (5.0x)

```go
const maxCorrectionFactor = 5.0

correction = float64(realTokens) / float64(lastHeuristic)
if correction < 1.0 { correction = 1.0 }
if correction > maxCorrectionFactor { correction = maxCorrectionFactor }
```

The `len(text)/4` heuristic typically underestimates by 1.5-3x. A 5x cap handles extreme cases while preventing runaway corrections. Combined with `resetCalibration()` after compaction, the system is robust against both anomalous turns and stale post-compaction factors.

#### Conversation truncation for summarizer

```go
func truncateForSummarizer(contents []*genai.Content, contextWindow int) []*genai.Content {
    budget := int(float64(contextWindow) * 0.80)  // 80% leaves room for system prompt + output
    for len(contents) > 2 && estimateContentTokens(contents) > budget {
        contents = contents[1:]  // drop oldest messages first
    }
    return contents
}
```

Called before `summarize()`. Trims the conversation to 80% of the context window. The 20% headroom is for the summarization system prompt (~1k tokens), previous summary context, and output tokens. Oldest messages are dropped first because recent context is most valuable for a useful summary.

#### Fallback summary on failure

```go
summary, err := summarize(ctx, s.llm, contentsForSummary, existingSummary, buffer, todos)
if err != nil {
    slog.Warn("ContextGuard [threshold]: summarization failed, using fallback", ...)
    summary = buildFallbackSummary(contentsForSummary, existingSummary)
}
```

`buildFallbackSummary()` concatenates the first 200 chars of each message — mechanical but guaranteed to succeed and be small. The session continues with degraded summary quality rather than crashing.

### Stress test validation

25 multi-turn session simulations covering:
- Both 200k and 8k context windows
- Token ratios from 1.5x to 4.0x
- With and without UsageMetadata (pure heuristic mode)
- Normal conversation, tool-heavy, tool bursts, massive tool responses (up to 750k chars)
- Long-running sessions (40-100 turns with multiple compactions)
- Large system prompts (up to 12k chars in an 8k window)
- Compaction loop detection
- Alternating tool/text patterns

All 25 tests pass with zero overflows and zero compaction loops. See TODOS.md for the complete test matrix.
