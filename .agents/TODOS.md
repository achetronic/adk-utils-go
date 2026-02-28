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

---

## ~~Proposal 2: Iterative compaction (retry loop)~~

**Status**: Implemented then removed  
**Package**: `plugin/contextguard`

### What was done

Originally added a retry loop that halved `recentBudget` each pass. **Removed** when the threshold strategy moved to full-summary mode (Proposal 5) — with no recent tail, there is nothing to shrink, so retry is unnecessary. The sliding window strategy still retries with a progressively smaller `recentKeep`.

---

## ~~Proposal 4: Align `estimateContentTokens` with `estimateTokens`~~

**Status**: Implemented  
**Package**: `plugin/contextguard`  
**Files**: `compaction_utils.go`

### What was done

Extracted `estimatePartTokens(part *genai.Part) int` helper that counts Text + FunctionCall (name + args) + FunctionResponse (name + response). Both `estimateTokens` and `estimateContentTokens` now use it. The sliding window's diagnostic token count is now accurate for agentic conversations.

---

## ~~Proposal 3: Truncate giant tool_responses before summarization and in recent window~~

**Status**: No longer needed  
**Priority**: Was Medium (deferred)  
**Investigation**: See [INVESTIGATION_RESULTS.md](./INVESTIGATION_RESULTS.md) for the original data

### Why it's no longer needed

Proposal 3 was designed to solve the problem of oversized tool responses living in the **recent tail** after compaction. With full-summary mode (Proposal 5), there is no recent tail — everything is summarized. The summarization prompt itself doesn't include raw tool response payloads either (tool results are rendered as `[tool X returned a result]`). So the problem this proposal addressed no longer exists.

---

## ~~Proposal 5: Crush-style compaction~~

**Status**: Implemented  
**Branch**: `feat/crush-style-compaction`  
**Package**: `plugin/contextguard`  
**Files**: `contextguard.go`, `compaction_threshold.go`, `compaction_utils.go`, `contextguard_test.go`, `compaction_simulation_test.go`

### Problem

The original threshold strategy had two critical flaws:

1. **`len(text)/4` heuristic drastically underestimates real token counts** for structured content (JSON, markdown, non-ASCII). The compaction threshold was never reached before Anthropic rejected the request for exceeding its context window.
2. **Keeping a "recent tail"** after compaction meant oversized tool responses in the tail could still overflow the context window. The retry loop couldn't help because it only re-compacted already-compacted content — the recent window was untouched.

Crush CLI (github.com/charmbracelet/crush) solves both by using real provider token counts and summarizing everything.

### What was implemented

#### 5.1: Real token counts via `AfterModelCallback`

ADK's `AfterModelCallback` receives `LLMResponse.UsageMetadata.PromptTokenCount`, populated by both the OpenAI and Anthropic adapters in adk-utils-go. The implementation:

- `AfterModelCallback` persists `PromptTokenCount` to session state after each LLM call (filtering out streaming partials via `resp.Partial` and nil `UsageMetadata`).
- `BeforeModelCallback` persists the heuristic estimate of the final request (after any compaction) so the next call can compute a calibration factor.
- `tokenCount()` uses a calibrated heuristic (see "Timing gap" section below).
- Falls back to `heuristic × 1.5` if no real tokens are available (first turn).

Only `PromptTokenCount` is stored — not the sum with `CandidatesTokenCount` — because `PromptTokenCount` already represents the full conversation size the LLM received.

#### 5.2: Full summary mode (always, not opt-in)

The threshold strategy always summarizes the **entire** conversation. No recent tail is preserved. After compaction, `req.Contents` is exactly `[summary] + [continuation]` — 2 messages. This matches Crush CLI behavior and eliminates the problem of oversized recent messages.

`WithFullSummary()` was initially implemented as an opt-in option, then made the only behavior. The option, the `fullSummary` field, `splitContents()`, `initialRecentBudget()`, `recentWindowRatio`, and the retry loop were all removed from the threshold strategy.

#### 5.3: Continuation context injection

After compaction, a continuation message is appended to `req.Contents`:

```
[System: The conversation was compacted because it exceeded the context window.
The summary above contains all prior context. The user's current request is:
`<original user message>`. Continue working on this request without asking
the user to repeat anything.]
```

`CallbackContext.UserContent()` provides the latest user message. If unavailable, a generic continuation instruction is used.

#### 5.4: Todo preservation in summarization prompt

When compaction fires, `loadTodos()` reads the todo list from session state (key `"todos"`) and appends it to the summarization prompt:

```
[Current todo list]
- [in_progress] Analyze timing gap
- [completed] Implement real token counts
[End todo list]

Include these tasks and their statuses in your summary under a dedicated
"## Todo List" section. Instruct the resuming assistant to restore them
using the `todos` tool to continue tracking progress.
```

Supports both `[]TodoItem` and `[]any` (from JSON deserialization).

### Timing gap analysis and calibrated heuristic

#### The problem

ContextGuard checks tokens in `BeforeModelCallback` but gets real token counts in `AfterModelCallback` — one step behind. Between the two callbacks, tool results are appended to the conversation. If a tool returns a massive response, the request sent to the LLM will be larger than the last measured `PromptTokenCount`.

Crush CLI doesn't have this problem because it checks *after* each step completes (with fresh token counts) and stops *before* the next LLM call. ADK only offers before/after callbacks — there is no `StopWhen` mechanism.

#### The solution: calibrated heuristic

Both the real token count and the heuristic estimate are persisted at each step:

| Callback | What is persisted | Purpose |
|----------|------------------|---------|
| `BeforeModelCallback` | `lastHeuristic` = `estimateTokens(req)` after compaction | Calibration baseline |
| `AfterModelCallback` | `realTokens` = `PromptTokenCount` from provider | Ground truth |

On the next `BeforeModelCallback`, `tokenCount()` computes:

```
currentHeuristic = estimateTokens(req)            // reflects current request including new tool results
correction = max(1.0, realTokens / lastHeuristic)  // how much len/4 underestimated last time
calibrated = currentHeuristic × correction          // apply correction to current estimate
return max(realTokens, calibrated)                  // never undercount
```

This means:
- If tools added tokens, `currentHeuristic` grows, and the correction factor scales it to real-token space.
- If the request didn't grow, `realTokens` from the last call is still the dominant value.
- The correction factor is floored at 1.0 — the calibrated estimate is never less than the raw heuristic.
- If no real tokens are available (first turn), the default factor of 1.5 is used.

#### Proof via simulation tests

`TestTimingGap_CalibratedHeuristicPreventsOverflow`:
- Step N: heuristic=70k, real=140k (correction=2.0)
- Tool adds 80k chars → Step N+1: heuristic=90k
- Old `tokenCount()`: returns 140k (stale) → below 180k threshold → **no compaction → overflow**
- New `tokenCount()`: returns 180k (calibrated = 90k × 2.0) → above threshold → **compaction fires**

`TestTimingGap_MassiveToolResponse`:
- 400k-char tool response between steps
- Calibrated estimate = 300k → correctly triggers compaction on a 200k window

### What the summarization flow looks like

**Input to summarization LLM** (separate call from the agent's normal LLM):

System prompt: structured instructions for producing a summary with sections (Current State, Key Information, Context & Decisions, Exact Next Steps), plus a dynamic word limit based on the buffer size.

User prompt: the entire conversation rendered as text:
```
user: Revisa los pods
model: [called tool: kubectl_get_pods]
user: [tool kubectl_get_pods returned a result]
model: Hay 40 pods running...
```

Note: tool results are rendered as `[tool X returned a result]` — raw payloads are not included. This keeps the summarization input manageable regardless of tool response size.

**Output**: a structured summary of ~buffer×0.50 tokens max.

**Final `req.Contents` sent to the agent's LLM**:
```
[user: "[Previous conversation summary]\n<summary>\n[End of summary — conversation continues below]"]
[user: "[System: The conversation was compacted... The user's current request is: `...`. Continue working...]"]
```

From ~140k tokens down to ~10k. The agent continues seamlessly.

### Test coverage

93 tests total, including:
- `TestPersistAndLoadRealTokens`, `TestLoadRealTokens_Float64Conversion`
- `TestTokenCount_PrefersRealTokens`, `TestTokenCount_CalibratedHeuristic`, `TestTokenCount_CorrectionFloorAtOne`, `TestTokenCount_RealTokensWinWhenLarger`
- `TestPersistAndLoadLastHeuristic`
- `TestAfterModel_PersistsTokenCount`, `TestAfterModel_NilUsageMetadata`, `TestAfterModel_SkipsPartials`, `TestAfterModel_UnknownAgent`
- `TestThresholdStrategy_UsesRealTokens`, `TestThresholdStrategy_RealTokens_TriggersWhereHeuristicFails`
- `TestThresholdStrategy_NoRecentTail`
- `TestInjectContinuation_WithUserContent`, `TestInjectContinuation_NilUserContent`, `TestThresholdStrategy_InjectsContinuation`
- `TestLoadTodos_Empty`, `TestLoadTodos_TypedSlice`, `TestLoadTodos_MapSlice`
- `TestBuildSummarizePrompt_WithTodos`, `TestBuildSummarizePrompt_WithoutTodos`
- `TestPluginConfig_HasAfterModelCallback`
- `TestTimingGap_CalibratedHeuristicPreventsOverflow`, `TestTimingGap_MassiveToolResponse`

### References

- Crush CLI compaction: `internal/agent/agent.go` — `Summarize()`, `getSessionMessages()`, `buildSummaryPrompt()`
- ADK `AfterModelCallback` signature: `func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)`
- ADK `UsageMetadata`: `genai.GenerateContentResponseUsageMetadata.PromptTokenCount`, `.CandidatesTokenCount`
- ADK callback ordering: `internal/llminternal/base_flow.go` — `preprocess → BeforeModelCallback → LLM → AfterModelCallback → tools → loop`
- ADK state persistence: `ctx.State().Set()` dual-writes to `stateDelta` and `Session().State()` — immediately visible across callbacks within the same session
