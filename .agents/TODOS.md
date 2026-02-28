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
**Files**: `contextguard.go`, `compaction_strategy_threshold.go`, `compaction_utils.go`, `contextguard_unit_test.go`, `compaction_strategy_singleshot_test.go`

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
- Falls back to `heuristic × 2.0` if no real tokens are available (first turn).

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

93 unit tests + 25 stress tests, including:
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
- 25 `TestStress_*` multi-turn session simulations (see Proposal 6 below)

### References

- Crush CLI compaction: `internal/agent/agent.go` — `Summarize()`, `getSessionMessages()`, `buildSummaryPrompt()`
- ADK `AfterModelCallback` signature: `func(ctx agent.CallbackContext, llmResponse *model.LLMResponse, llmResponseError error) (*model.LLMResponse, error)`
- ADK `UsageMetadata`: `genai.GenerateContentResponseUsageMetadata.PromptTokenCount`, `.CandidatesTokenCount`
- ADK callback ordering: `internal/llminternal/base_flow.go` — `preprocess → BeforeModelCallback → LLM → AfterModelCallback → tools → loop`
- ADK state persistence: `ctx.State().Set()` dual-writes to `stateDelta` and `Session().State()` — immediately visible across callbacks within the same session

---

## Proposal 6: Hardening — correction factor cap, summarizer overflow protection, fallback on failure

**Status**: Implemented  
**Branch**: `feat/crush-style-compaction`  
**Package**: `plugin/contextguard`  
**Files**: `contextguard.go`, `compaction_strategy_threshold.go`, `compaction_utils.go`, `compaction_strategy_multiturn_test.go`

### Problem

After deploying Proposal 5 (Crush-style compaction), three architectural weaknesses were identified:

1. **Uncapped correction factor**: `correction = realTokens / lastHeuristic` could theoretically be very large (e.g., 10x with JSON-heavy tool schemas). A single turn with unusual content could produce a disproportionate correction that persists across turns, causing premature compaction on every subsequent message.

2. **Summarize call can explode**: `summarize()` sends the entire conversation as a prompt to the summarizer LLM. If the conversation is near or exceeds the context window (e.g., a 300k-char tool response in a 200k window), the summarization prompt itself exceeds the LLM's context window and the call fails.

3. **Compaction failure = pass-through explosion**: If `summarize()` failed, `Compact()` returned an error, `beforeModel` logged a warning and **passed through the original bloated request** — which then hit the provider and got rejected for exceeding the context window. The failure mode was worse than doing nothing.

### What was implemented

#### 6.1: Correction factor capped at 5.0x

Added `maxCorrectionFactor = 5.0` constant (`contextguard.go:82`).

In `tokenCount()` (`compaction_utils.go`), the correction factor is now clamped:

```go
correction = float64(realTokens) / float64(lastHeuristic)
if correction < 1.0 {
    correction = 1.0
}
if correction > maxCorrectionFactor {
    correction = maxCorrectionFactor
}
```

**Rationale**: The `len(text)/4` heuristic typically underestimates by 1.5-3x. A 5x cap is generous enough to handle extreme cases (heavily structured JSON, non-ASCII text) while preventing a single anomalous turn from distorting all future estimates. The post-compaction `resetCalibration()` already mitigates stale factors, but the cap adds defense-in-depth during normal operation.

#### 6.2: Conversation truncation before summarization

Added `truncateForSummarizer()` (`compaction_utils.go:195-209`):

```go
func truncateForSummarizer(contents []*genai.Content, contextWindow int) []*genai.Content {
    budget := int(float64(contextWindow) * 0.80)
    total := estimateContentTokens(contents)
    if total <= budget {
        return contents
    }
    for len(contents) > 2 && estimateContentTokens(contents) > budget {
        contents = contents[1:]
    }
    return contents
}
```

Before calling `summarize()`, the conversation is trimmed to 80% of the context window. The 80% budget leaves room for the summarization system prompt, the previous summary (if any), and the output tokens. The oldest messages are dropped first (most recent context is the most valuable).

**Called from**: `Compact()` in `compaction_threshold.go:103` — `contentsForSummary := truncateForSummarizer(req.Contents, contextWindow)`.

#### 6.3: Fallback summary on summarization failure

If the LLM summarization call fails, `Compact()` now falls back to `buildFallbackSummary()` (a mechanical summary that concatenates the first 200 chars of each message) instead of propagating the error:

```go
summary, err := summarize(ctx, s.llm, contentsForSummary, existingSummary, buffer, todos)
if err != nil {
    slog.Warn("ContextGuard [threshold]: summarization failed, using fallback", ...)
    summary = buildFallbackSummary(contentsForSummary, existingSummary)
}
```

**Before**: summarization failure → `Compact()` returns error → `beforeModel` logs warning and passes through → provider rejects the bloated request.

**After**: summarization failure → fallback summary → compaction still fires → request size drops dramatically → provider accepts it. The summary quality is lower but the session survives.

### Comprehensive stress test suite

**File**: `compaction_strategy_multiturn_test.go` (25 tests)

#### Test infrastructure

`simulateSession()` runs a full multi-turn session through the real compaction pipeline, mimicking the ADK flow:

```
For each turn:
  1. User sends message → append to contents
  2. Build LLMRequest (contents + system instruction)
  3. BeforeModelCallback (Compact + persistLastHeuristic)
  4. Detect compaction (tokens decreased + summary exists)
  5. Simulate LLM response → AfterModelCallback (persist real tokens)
  6. Model text response → append to contents
  7. Tool calls (if any) → model FunctionCall + user FunctionResponse → append to contents
  8. Next turn
```

Key types:
- `sessionConfig`: contextWindow, systemPromptSize, modelName, hasUsageMetadata, tokenRatio
- `turnConfig`: userMessage, toolCalls (name + responseSize)
- `sessionResult`: turns, compactions, finalTokens, maxTokensSeen, overflowed, compactionFailed, loopDetected

`longMessage(turn, length)` generates realistic user messages of a given character count (Kubernetes-themed technical content).

#### Test matrix

| Test | Window | Turns | Token ratio | UsageMetadata | Tools | What it validates |
|------|--------|-------|-------------|---------------|-------|-------------------|
| `TestStress_200k_NormalConversation` | 200k | 30 | 2.0 | yes | none | Short messages don't trigger unnecessary compaction |
| `TestStress_200k_ToolHeavy` | 200k | 20 | 2.0 | yes | 3/turn (10k-20k) | Multiple compactions with heavy tool usage |
| `TestStress_200k_SingleGiantToolResponse` | 200k | 3 | 2.0 | yes | 1x 300k | Session survives a tool response larger than the window |
| `TestStress_200k_ToolBurst` | 200k | 3 | 2.0 | yes | 10x 5k burst | 10 tools in one turn doesn't overflow |
| `TestStress_8k_NormalConversation` | 8k | 20 | 1.8 | yes | none | Long text-only messages trigger compaction in small window |
| `TestStress_8k_SmallToolCalls` | 8k | 15 | 1.8 | yes | 1/turn (1k) | Text + small tools compact properly |
| `TestStress_8k_LargeToolResponse` | 8k | 3 | 1.8 | yes | 1x 20k | Tool response 2.5x the window compacts on next turn |
| `TestStress_200k_NoUsageMetadata` | 200k | 25 | 2.5* | no | 1/turn (8k) | Pure heuristic mode (default 1.5x factor) works |
| `TestStress_8k_NoUsageMetadata` | 8k | 25 | 2.0* | no | 1/turn (1.5k) | Pure heuristic compacts in small window |
| `TestStress_200k_LongRunning_50Turns` | 200k | 50 | 2.2 | yes | every 3rd (5k+15k) | Long session with periodic tools doesn't loop |
| `TestStress_8k_LongRunning_40Turns` | 8k | 40 | 1.8 | yes | none | Multiple compactions across long text session |
| `TestStress_200k_MassiveToolBurst` | 200k | 2 | 2.0 | yes | 15x 50k | 750k chars of tool responses in one turn |
| `TestStress_8k_ToolBurst` | 8k | 3 | 1.8 | yes | 3x 3k + 2x 1-2k | Tool burst in small window doesn't overflow |
| `TestStress_200k_HighTokenRatio` | 200k | 15 | 3.0 | yes | 1/turn (10k) | 3x underestimation handled by calibration |
| `TestStress_8k_HighTokenRatio` | 8k | 20 | 3.0 | yes | 1/turn (1k) | High ratio + small window compacts correctly |
| `TestStress_200k_LargeSystemPrompt` | 200k | 20 | 2.0 | yes | 1/turn (5k) | 50k-char system prompt (12.5k heuristic tokens) |
| `TestStress_8k_LargeSystemPrompt` | 8k | 15 | 1.8 | yes | none | 8k-char system prompt eats 25% of small window |
| `TestStress_CompactionNoInfiniteLoop` | 8k | 5 | 2.5 | yes | none | 12k-char system prompt (> window) doesn't cause loop |
| `TestStress_200k_LateUsageMetadata` | 200k | 5+20 | 2.0/2.5 | no→yes | 1/turn (5-10k) | Transition from heuristic to calibrated mode |
| `TestStress_8k_RepeatedCompactions` | 8k | 40 | 1.8 | yes | 1/turn (2k) | ≥6 compactions, no overflow, no loops |
| `TestStress_200k_RepeatedCompactions` | 200k | 60 | 2.0 | yes | every 2nd (30k+10k) | ≥3 compactions in heavy 200k session |
| `TestStress_8k_OnlyToolResponses` | 8k | 10 | 2.0 | yes | 1/turn (5k) | Minimal text, dominated by tool responses |
| `TestStress_200k_VeryHighTokenRatio_4x` | 200k | 10 | 4.0 | yes | 1/turn (20k) | Extreme 4x tokenizer ratio |
| `TestStress_8k_RapidFireShortMessages` | 8k | 80 | 1.8 | yes | none | Many short messages over long session |
| `TestStress_200k_100Turns_MixedWorkload` | 200k | 100 | 2.3 | yes | mixed (1k-50k) | Long session with varied tool sizes |
| `TestStress_8k_AlternatingToolAndText` | 8k | 30 | 2.0 | yes | every 2nd (3k) | Alternating patterns compact correctly |

*Token ratio marked with `*` is irrelevant when `hasUsageMetadata=false` — the system never sees "real" tokens and uses only `heuristic * 1.5`.

#### What each test asserts

All tests assert:
- **No overflow**: `estimatedRealTokens` (= heuristic × tokenRatio) never exceeds the context window
- **No compaction loops**: compaction always reduces token count (tracked via `loopDetected`)

Tests with expected compaction also assert:
- **Compactions > 0**: long sessions or large tool responses actually trigger compaction
- **Compactions >= N**: for long-running sessions, multiple compactions must fire

#### Why `maxTokensSeen` uses `heuristic × tokenRatio` instead of `tokenCount()`

The tests track overflow using `heuristic × tokenRatio` as a proxy for real token counts. This is a worst-case estimate: it represents what the provider would actually see. `tokenCount()` uses the calibrated heuristic (which may be lower than reality on the first turn before calibration data exists). By checking the "real" proxy, we ensure the system prevents actual overflow, not just heuristic overflow.

#### Key observations from passing tests

- No overflow in any scenario (0 failures across all 25 tests)
- No compaction loops detected
- 200k window: compaction fires with heavy tool usage (2-3 compactions in 60 turns)
- 8k window: compaction fires frequently and reliably (5-6 compactions in 40 turns)
- Post-compaction token count is always low (~150-850 heuristic tokens)
- Calibration reset works: no stale correction factors after compaction
- Default 1.5x factor works for providers without UsageMetadata
- 4x token ratio handled without overflow (cap at 5x provides headroom)
- 100-turn sessions with mixed workloads complete cleanly

### Architecture diagram (updated)

```
BeforeModelCallback (step N)
├── tokenCount(ctx, req)
│   ├── currentHeuristic = estimateTokens(req)
│   ├── real = loadRealTokens(ctx)
│   ├── lastHeuristic = loadLastHeuristic(ctx)
│   ├── correction = clamp(real / lastHeuristic, 1.0, 5.0)   ← NEW: capped at 5x
│   ├── calibrated = currentHeuristic × correction
│   └── return max(real, calibrated)
├── if totalTokens < threshold → pass through
├── if totalTokens ≥ threshold:
│   ├── truncateForSummarizer(contents, contextWindow)         ← NEW: trim to 80% for safety
│   ├── summarize(truncated conversation)
│   │   └── on error → buildFallbackSummary()                  ← NEW: fallback instead of propagate
│   ├── replaceSummary(req, summary, nil)
│   ├── injectContinuation(req, userContent)
│   ├── resetCalibration(ctx)
│   └── persistSummary(ctx, summary, totalTokens)
└── persistLastHeuristic(ctx, estimateTokens(req))

LLM call (step N) → response with PromptTokenCount

AfterModelCallback (step N)
└── persistRealTokens(ctx, PromptTokenCount)
```

---

## Proposal 7: ADK flow audit, simulator rewrite, and brutal stress tests (second round)

**Status**: Implemented  
**Branch**: `feat/crush-style-compaction`  
**Package**: `plugin/contextguard`  
**Files**: `compaction_strategy_multiturn_test.go`

### Motivation

The first round of stress tests (Proposal 6) used a simulator that had several gaps relative to the real ADK execution flow. A line-by-line audit of ADK's `base_flow.go` revealed subtle but meaningful differences that needed to be corrected to ensure the tests truly validate production behavior.

### ADK execution flow (confirmed from source)

Read from `google.golang.org/adk@v0.4.0/internal/llminternal/base_flow.go`:

```
Run() (line 97):
  for {
    runOneStep()
    if lastEvent.IsFinalResponse() → break
  }

runOneStep() (line 123):
  1. req = fresh LLMRequest (Contents=nil)               // line 130
  2. preprocess(req):                                      // line 135
     - ContentsRequestProcessor rebuilds Contents from
       ALL session events (entire history)                 // contents_processor.go:50
  3. callLLM(req):                                         // line 153
     3a. Plugin BeforeModelCallback                        // line 288
     3b. User BeforeModelCallbacks                         // line 298
     3c. Model.GenerateContent                             // line 314
     3d. AfterModelCallbacks                               // line 328
  4. postprocess                                           // line 158
  5. handleFunctionCalls:                                   // line 191
     - For each FunctionCall in response:
       - callTool() (BeforeToolCallback → tool.Run → AfterToolCallback)
     - All FunctionResponse parts merged into ONE event    // line 547
  6. Yield function response event                         // line 208
  7. Back in Run(): IsFinalResponse()=false → LOOP
```

**Key invariants confirmed:**
- Each `runOneStep` creates a FRESH `req` — no accumulation between iterations
- `ContentsRequestProcessor` rebuilds `Contents` from session events every time
- Parallel tool calls produce ONE model Content with multiple FunctionCall parts
- Parallel tool responses are merged into ONE user Content with multiple FunctionResponse parts
- `BeforeModelCallback` sees tool results because `preprocess` loads them BEFORE the callback fires
- `AfterModelCallback` fires after EVERY `GenerateContent` call, including tool-result processing calls

### Simulator improvements (second round)

#### 1. Deep clone (`cloneContents`)

Replaced `copyContents` (shallow slice copy) with `cloneContents` that copies each `*genai.Content` struct by value and duplicates the Parts slice. This prevents mutations in `beforeModel` from leaking back to the session history.

```go
func cloneContents(src []*genai.Content) []*genai.Content {
    dst := make([]*genai.Content, len(src))
    for i, c := range src {
        if c == nil { continue }
        clone := *c
        clone.Parts = make([]*genai.Part, len(c.Parts))
        copy(clone.Parts, c.Parts)
        dst[i] = &clone
    }
    return dst
}
```

#### 2. Parallel tool calls modeled correctly

All tool calls within a turn are now placed in a single model Content with multiple `FunctionCall` parts, and all tool responses in a single user Content with multiple `FunctionResponse` parts — matching how ADK's `mergeParallelFunctionResponseEvents` works.

Before (first round):
```
[model: FunctionCall("tool_1")] [user: FunctionResponse("tool_1")]
[model: FunctionCall("tool_2")] [user: FunctionResponse("tool_2")]
```

After (second round):
```
[model: FunctionCall("tool_1"), FunctionCall("tool_2")]
[user: FunctionResponse("tool_1"), FunctionResponse("tool_2")]
```

This is significant because it changes the token estimation — fewer Content entries, same total chars.

#### 3. Sequential tool chains

New `sequential: true` field on `turnConfig`. When enabled, each tool call is a separate `runOneStep` iteration:

```
User message → runLLMStep("user-msg")
  → model calls tool A → runLLMStep("tool-chain-0")
  → model calls tool B → runLLMStep("tool-chain-1")
  → model calls tool C → runLLMStep("tool-chain-2")
  → model returns text → BREAK
```

This models the real ADK flow where a model chains tool calls sequentially (call A → get result → call B → get result → return text). Each step triggers a full `BeforeModelCallback`, so `ContextGuard` can compact mid-chain.

#### 4. Loop detection fixed

The first round had dead code in loop detection (`tokensAfter >= tokensBefore` inside a `tokensAfter < tokensBefore` block — always false). Replaced with proper within-turn loop detection:

- Tracks whether compaction occurred on the first step of a turn (`turnHadCompaction`)
- Only flags `loopDetected` if BOTH steps within the same turn compact AND the second didn't reduce below the first
- Repeated compactions across different turns with the same post-compaction size is **normal behavior** (incompressible system prompt dominates), not a loop

#### 5. Variable model response size

New `responseSize` field on `turnConfig` allows tests to specify how large the model's text response is. Default is 120 chars. Tests like `TestBrutal_8k_VeryLargeModelResponses` use 2000-char responses to stress accumulation from the model's own output.

### New test suite (47 tests total)

The second round expanded from 25 to 47 tests across three categories:

#### Category 1: Stress tests (`TestStress_*`) — 27 tests

Updated from the first round with corrected parallel tool modeling. Same scenarios, more accurate simulation.

| Test | Window | Turns | Ratio | UsageMetadata | Tools |
|------|--------|-------|-------|---------------|-------|
| `200k_NormalConversation` | 200k | 30 | 2.0 | yes | none |
| `200k_ToolHeavy` | 200k | 20 | 2.0 | yes | 3/turn parallel (10k-20k) |
| `200k_SingleGiantToolResponse` | 200k | 3 | 2.0 | yes | 1×300k |
| `200k_ToolBurst_10Parallel` | 200k | 3 | 2.0 | yes | 10 parallel (5k each) |
| `200k_NoUsageMetadata` | 200k | 25 | 2.5 | no | 1/turn (8k) |
| `200k_LongRunning_50Turns` | 200k | 50 | 2.2 | yes | every 3rd (5k+15k) |
| `200k_HighTokenRatio` | 200k | 15 | 3.0 | yes | 1/turn (10k) |
| `200k_VeryHighTokenRatio_4x` | 200k | 10 | 4.0 | yes | 1/turn (20k) |
| `200k_LargeSystemPrompt` | 200k | 20 | 2.0 | yes | 1/turn (5k), 50k system |
| `200k_MassiveToolBurst` | 200k | 2 | 2.0 | yes | 15 parallel (50k each) |
| `200k_RepeatedCompactions` | 200k | 60 | 2.0 | yes | every 2nd (30k+10k) |
| `200k_LateUsageMetadata` | 200k | 5+20 | 2.0→2.5 | no→yes | 1/turn (5-10k) |
| `200k_100Turns_MixedWorkload` | 200k | 100 | 2.3 | yes | mixed (1k-50k) |
| `8k_NormalConversation` | 8k | 20 | 1.8 | yes | none |
| `8k_SmallToolCalls` | 8k | 15 | 1.8 | yes | 1/turn (1k) |
| `8k_LargeToolResponse` | 8k | 3 | 1.8 | yes | 1×20k |
| `8k_NoUsageMetadata` | 8k | 25 | 2.0 | no | 1/turn (1.5k) |
| `8k_LongRunning_40Turns` | 8k | 40 | 1.8 | yes | none |
| `8k_ToolBurst` | 8k | 3 | 1.8 | yes | 3 parallel (3k) + 2 (1-2k) |
| `8k_HighTokenRatio` | 8k | 20 | 3.0 | yes | 1/turn (1k) |
| `8k_LargeSystemPrompt` | 8k | 15 | 1.8 | yes | none, 8k system |
| `8k_OnlyToolResponses` | 8k | 10 | 2.0 | yes | 1/turn (5k) |
| `8k_RapidFireShortMessages` | 8k | 80 | 1.8 | yes | none |
| `8k_RepeatedCompactions` | 8k | 40 | 1.8 | yes | 1/turn (2k) |
| `8k_AlternatingToolAndText` | 8k | 30 | 2.0 | yes | every 2nd (3k) |
| `CompactionNoInfiniteLoop` | 8k | 5 | 2.5 | yes | none, 12k system |

#### Category 2: Brutal tests (`TestBrutal_*`) — 17 tests

Extreme scenarios designed to break the compaction system:

| Test | Window | Turns | Ratio | Key stress |
|------|--------|-------|-------|------------|
| `8k_ToolResponseBiggerThanWindow` | 8k | 3 | 2.0 | Single 40k tool response (5× window) |
| `8k_EveryTurnExceedsWindow` | 8k | 15 | 2.0 | 2k msg + 15k tool every turn |
| `8k_NoUsageMetadata_HighRatio` | 8k | 20 | 2.0 | Pure heuristic at default factor limit |
| `8k_NoUsageMetadata_BeyondDefault` | 8k | 15 | 3.0 | Ratio > default factor (known limitation) |
| `8k_150Turns` | 8k | 150 | 1.8 | Extreme session length |
| `8k_SystemPromptLargerThanWindow` | 8k | 10 | 2.0 | 15k system prompt > 8k window |
| `8k_ToolChain_MultipleRoundtrips` | 8k | 10 | 2.0 | 5 parallel tools/turn (10k total) |
| `8k_AlternatingHugeAndTiny` | 8k | 30 | 2.0 | "ok" alternating with 10k tools |
| `8k_CompactionEveryStep` | 8k | 30 | 2.0 | 1.2k msg + 4k tool + 4k system |
| `8k_CorrectionFactorDrift` | 8k | 20 | 1.5 | Low ratio testing drift |
| `8k_EmptyToolResponses` | 8k | 25 | 2.0 | 3 tools with 3-10 char responses |
| `8k_VeryLargeModelResponses` | 8k | 20 | 2.0 | 2k-char model responses |
| `8k_JSON_Heavy_ToolResponses` | 8k | 15 | 3.5 | High ratio simulating JSON structure overhead |
| `200k_ConsecutiveMassiveBursts` | 200k | 10 | 2.5 | 80k+30k tools every turn |
| `200k_NoUsageMetadata_LongSession` | 200k | 80 | 2.5 | No calibration over 80 turns |
| `200k_TokenRatio_5x` | 200k | 10 | 5.0 | Extreme 5× underestimation |
| `200k_SingleTurnFillsWindow` | 200k | 2 | 2.0 | 20 parallel tools × 30k = 600k |
| `200k_200Turns` | 200k | 200 | 2.2 | Extreme session length with mixed tools |

#### Category 3: Sequential tool chain tests (`TestBrutal_*_Sequential*`) — 3 tests

Models sequential tool call chains where each call is a separate ADK `runOneStep` iteration:

| Test | Window | Turns | Ratio | UsageMetadata | Tools/turn | Tool sizes | BeforeModelCallback calls/turn |
|------|--------|-------|-------|---------------|------------|------------|-------------------------------|
| `8k_SequentialToolChain` | 8k | 8 | 2.0 | yes | 5 sequential | 2k, 1.5k, 2.5k, 1k, 1.5k | 6 (1 user-msg + 5 tool-chain) |
| `200k_SequentialToolChain_LargeResponses` | 200k | 5 | 2.0 | yes | 5 sequential | 40k, 20k, 30k, 15k, 5k | 6 |
| `8k_SequentialChain_NoUsageMetadata` | 8k | 10 | 2.0 | no | 3 sequential | 1k, 800, 1.2k | 4 |

The 8k sequential chain test with 8 turns and 5 tools/turn exercises the pipeline **48 times** (8 × 6 `BeforeModelCallback` calls). This is the most thorough test of mid-chain compaction behavior.

### Known limitation documented

`TestBrutal_8k_NoUsageMetadata_BeyondDefault` documents that when `tokenRatio > defaultHeuristicCorrectionFactor (2.0)` and the provider doesn't report `UsageMetadata`, the system has no way to learn the real ratio. It still compacts (preventing infinite growth) but may briefly overflow between compaction cycles. The test verifies:
- No compaction loops
- Compaction still fires
- System survives without crashing

This is a fundamental limitation: without calibration data, the system can only be as accurate as its default factor. Providers that don't report `UsageMetadata` should use `WithMaxTokens()` with a conservative override.

### Observations from the full 47-test run

- **Zero overflow** in all tests where `tokenRatio ≤ defaultHeuristicCorrectionFactor` or `hasUsageMetadata=true`
- **Zero within-turn compaction loops** in any test
- **200k window**: 1-3 compactions across long sessions (60-200 turns)
- **8k window**: 1-15 compactions depending on message size and frequency
- **Post-compaction tokens**: consistently 146-3096 heuristic tokens (well below any threshold)
- **Sequential chains**: compaction fires mid-chain when needed, preventing accumulation
- **Correction factor cap (5.0)**: prevents 5× ratio from causing issues when calibration data exists
- **Default factor (2.0)**: covers the typical 1.5-2× underestimation range without calibration
- **Calibration reset after compaction**: no stale correction factors observed
- **150/200-turn sessions**: stable behavior with no degradation over time
