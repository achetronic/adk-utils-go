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

---

## Proposal 5: Crush-style token threshold compaction

**Priority**: High  
**Package**: `plugin/contextguard`  
**Files**: `contextguard.go`, `compaction_threshold.go`, `compaction_utils.go`

### Problem

The current threshold strategy uses a `len(text)/4` heuristic to estimate tokens. This significantly underestimates token usage for structured content (JSON tool calls, markdown, etc.) — the real ratio is often ~3 chars/token or less. As a result, the LLM rejects requests for exceeding its context window **before** ContextGuard's estimate reaches the compaction threshold, so compaction never fires.

Crush CLI (github.com/charmbracelet/crush) solves this by using **real token counts** reported by the provider after each LLM call, which are always accurate.

### Proposed changes

#### 1. Use real token counts instead of `len/4` heuristic

ADK's `AfterModelCallback` receives `LLMResponse.UsageMetadata` with `PromptTokenCount` and `CandidatesTokenCount`, populated by both the OpenAI and Anthropic adapters in adk-utils-go. The plan:

- Add an `AfterModelCallback` to the plugin that persists the latest `PromptTokenCount + CandidatesTokenCount` into session state after each LLM call.
- In `BeforeModelCallback`, read the accumulated token count from state instead of calling `estimateTokens()`.
- Fall back to `len/4` only if `UsageMetadata` is nil (e.g. first turn, or provider doesn't report usage).

Note: `PromptTokenCount` already includes the full conversation, so the last turn's `PromptTokenCount` is sufficient — no need to accumulate across turns.

#### 2. Summarize everything instead of keeping a recent tail

Crush summarizes the **entire** conversation and restarts with just `[summary]`. The current ContextGuard keeps ~20% of the context window as verbatim recent messages. Consider offering a mode (opt-in or default) that summarizes all contents, since the recent tail can itself contain oversized tool responses that defeat compaction (see also Proposal 3).

#### 3. Inject continuation context after compaction

After compaction, Crush injects the user's original prompt wrapped with `"The previous session was interrupted because it got too long, the initial user request was: ..."` and appends a continuation instruction. ContextGuard could do the same by appending a user message to `req.Contents` after `replaceSummary()`. `CallbackContext.UserContent()` provides the latest user message.

#### 4. Preserve todos in the summary prompt

Crush's `buildSummaryPrompt()` reads the todo list and includes it in the summarization request with an instruction to restore it via the `todos` tool. ContextGuard could read todos from `CallbackContext.State()` and inject them into the summarization prompt, ensuring task continuity across compaction boundaries.

### Impact assessment

| Change | Effort | Impact |
|---|---|---|
| Real token counts via `AfterModelCallback` | Medium | **Critical** — fixes compaction never firing |
| Summarize everything (no recent tail) | Low | Prevents oversized recent messages from defeating compaction |
| Continuation context injection | Low | Better task continuity after compaction |
| Todo preservation | Low | Prevents loss of task tracking state |

### Reference

- Crush CLI compaction: `internal/agent/agent.go` — `Summarize()`, `getSessionMessages()`, `buildSummaryPrompt()`
- ADK `AfterModelCallback` signature: `func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)`
- ADK `UsageMetadata`: `genai.GenerateContentResponseUsageMetadata.PromptTokenCount`, `.CandidatesTokenCount`
