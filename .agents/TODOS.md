# TODOs

## ~~contextguard: `safeSplitIndex` can regress splitIdx to 0, nullifying compaction~~

**Status**: Fixed  
**Package**: `plugin/contextguard`  
**Files**: `compaction_utils.go` (`safeSplitIndex`, `findSplitIndex`)

### What was done

1. **`safeSplitIndex` floor at `minIdx=1`**: the backwards loop now stops at 1 instead of 0, guaranteeing at least 1 message is always compacted. A `slog.Warn` fires when the floor is hit.
2. **`findSplitIndex` counts all token types**: the budget loop now counts `FunctionCall.Args` and `FunctionResponse.Response` tokens, not just `part.Text`. This makes the split point realistic for agentic conversations.
3. **7 new tests** covering all-tool conversations for both `safeSplitIndex`, `findSplitIndex`, and end-to-end threshold/sliding-window strategies.

---

## ~~Proposal 1: Cut at tool_call/tool_response pair boundaries~~

**Status**: Implemented  
**Package**: `plugin/contextguard`  
**Files**: `compaction_utils.go` (`safeSplitIndex`, `walkBackToPairBoundary`, `walkForwardToPairBoundary`)

### What was done

Rewrote `safeSplitIndex` with a two-phase approach:
1. **Backward walk** (`walkBackToPairBoundary`): tries to find a clean text boundary before the candidate index (preserves original behavior for mixed conversations).
2. **Forward walk** (`walkForwardToPairBoundary`): if backward walk reaches 0, walks forward from the original candidate to the nearest pair boundary — i.e., right after a `tool_response`, which is the seam between two complete `[tool_call, tool_response]` pairs.

This means pure-tool conversations now split between pairs instead of collapsing to index 0/1.

### Simulation results (before → after)

| Scenario | Before | After |
|---|---|---|
| tool-storm-20x2k / 8k ctx | old=1, -0.1%, NO fit | old=39, 94.9%, YES fit |
| tool-storm-50x5k / 32k ctx | old=1, -0.0%, NO fit | old=99, 98.0%, YES fit |
| tool-storm-50x5k / sw=10 | old=1, -0.0% | old=99, 98.0%, YES fit |
| tool-storm-50x5k / sw=30 | old=1, -0.0% | old=99, 98.0%, YES fit |

---

## ~~Proposal 2: Iterative compaction (retry loop)~~

**Status**: Implemented  
**Package**: `plugin/contextguard`  
**Files**: `compaction_threshold.go`, `compaction_sliding_window.go`, `contextguard.go`

### What was done

Both `thresholdStrategy.Compact` and `slidingWindowStrategy.Compact` now retry compaction up to `maxCompactionAttempts` (3) if tokens still exceed the threshold after the first pass:

- **Threshold**: halves `recentBudget` each retry, forcing the split point earlier.
- **Sliding window**: halves `recentKeep` each retry (min 3), keeping fewer recent messages.
- Each retry calls the LLM for summarization, accumulating the summary.
- Aborts early if `oldContents` is empty (nothing left to compact).

Added `maxCompactionAttempts = 3` constant to `contextguard.go`.

---

## ~~Proposal 4: Align `estimateContentTokens` with `estimateTokens`~~

**Status**: Implemented  
**Package**: `plugin/contextguard`  
**Files**: `compaction_utils.go`

### What was done

Extracted `estimatePartTokens(part *genai.Part) int` helper that counts Text + FunctionCall (name + args) + FunctionResponse (name + response). Both `estimateTokens` and `estimateContentTokens` now use it. The sliding window's diagnostic token count is now accurate for agentic conversations.

---

## Proposal 3: Truncate giant tool_responses before summarization and in recent window

**Priority**: Medium (deferred — will implement in a future version)  
**Investigation**: See [INVESTIGATION_RESULTS.md](./INVESTIGATION_RESULTS.md) for full data  
**Package**: `plugin/contextguard`  
**Files**: `compaction_utils.go` (new helper + changes to `summarize` and `replaceSummary`)

### Problem

A single `kubectl get pods -A -o json` can produce 200k+ chars (~50k tokens). This causes two problems:

1. **Summarization input is too large**: the LLM called for summarization also has a context limit and can't process 200k chars of JSON.
2. **Recent messages are untouched**: even after compaction, the "recent" portion keeps the full giant responses, defeating the purpose.

Simulation shows `kube-3rounds / 8k ctx` compacts 66.4% but still doesn't fit (8101 > 6400 threshold) because the remaining 8 recent messages include huge tool responses.

### Proposed solution

Two truncation points:

**A) Before summarization** — truncate tool_responses in `oldContents` before passing to `summarize()`:

```go
func truncateForSummarization(contents []*genai.Content, maxCharsPerResponse int) []*genai.Content {
    // Deep copy, truncate FunctionResponse.Response values > maxCharsPerResponse
    // Append "[truncated from X chars]" indicator
}
```

**B) In recent window** — optionally truncate oversized tool_responses in `recentContents`. Opt-in via `WithMaxToolResponseSize(n)`.

### Suggested defaults

| Parameter | Default | Rationale |
|---|---|---|
| Summarization truncation | 2,000 chars | Enough for tool name, key fields, errors |
| Recent window truncation | disabled | Preserve full context by default; user opts in |

### Remaining simulation gap

| Scenario | Current result | With Proposal 3 (estimated) |
|---|---|---|
| kube-3rounds / 8k ctx | 8101 tokens, NO fit | ~2k tokens, YES fit |
