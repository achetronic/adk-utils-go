# Compaction: Token Threshold Strategy

How the threshold strategy decides when and how to compact a conversation.

## Compaction flow

| Step | What happens | Why |
|------|-------------|-----|
| 1. **Estimate tokens** | Calculate how many tokens the current conversation uses. If real token counts from the provider are available, a correction factor (`real / previous heuristic`) is applied. Otherwise, the `len/4` heuristic is multiplied by 2.0x | The heuristic alone underestimates by 2-3x on structured content (JSON, tool schemas) |
| 2. **Compare against threshold** | If estimated tokens < `window - buffer`, do nothing. Buffer = fixed 20k for windows >200k, 20% for smaller windows | The buffer leaves room for the LLM response and error margin |
| 3. **Truncate for the summarizer** | If the conversation is huge, drop the oldest messages until it fits within 80% of the summarizer LLM's context window | Prevents the summarization call itself from blowing up |
| 4. **Summarize everything** | Send the entire conversation to the LLM with a structured prompt. If the LLM call fails, fall back to a mechanical summary (first 200 chars of each message) | A summary is always produced — the bloated conversation is never passed through |
| 5. **Replace contents** | `req.Contents` becomes just 2 messages: `[summary]` + `[continuation instruction]` | Goes from ~140k tokens down to ~1-2k. The agent continues without asking the user to repeat anything |
| 6. **Reset calibration** | Clear real token count and previous heuristic from session state | Prevents the pre-compaction correction factor (huge) from causing an infinite compaction loop |
| 7. **Persist heuristic** | Save the heuristic of the final (compacted) request to session state | The next turn can compute a fresh correction factor |

## Safeguards

| Safeguard | What it prevents |
|-----------|-----------------|
| Correction factor capped at 5.0x | A single unusual turn (heavy JSON) doesn't distort all subsequent turns |
| Truncate to 80% before summarizing | The summarization call doesn't exceed the summarizer's own context window |
| Mechanical fallback if LLM fails | The session survives with a degraded summary instead of crashing |
| Calibration reset after compaction | No infinite loop of compacting every turn with a stale factor |

## Token estimation: calibrated heuristic

The `len(text)/4` heuristic drastically underestimates real token counts. To bridge the gap between `AfterModelCallback` (where real tokens arrive) and `BeforeModelCallback` (where the decision must be made), a calibrated heuristic is used:

```
currentHeuristic  = estimateTokens(req)                          # on the current request
correction        = clamp(realTokens / lastHeuristic, 1.0, 5.0) # how much len/4 was off last time
calibrated        = currentHeuristic × correction                # scale to real-token space
result            = max(realTokens, calibrated)                  # never undercount
```

If no real tokens exist yet (first turn or provider doesn't report usage), the default factor of 1.5x is used.

## Buffer calculation

| Context window | Buffer | Threshold | Rationale |
|---------------|--------|-----------|-----------|
| >200k (e.g. 200k) | 20k fixed | 180k | Large windows don't need proportional buffers |
| ≤200k (e.g. 8k) | 20% (1.6k) | 6.4k | Small windows need proportional headroom |

## Post-compaction result

Before compaction:
```
[user message 1] [model response 1] [tool call] [tool result] ... [user message N]
~140k tokens
```

After compaction:
```
[summary: "Previous conversation summary... Current State... Key Information... Next Steps..."]
[continuation: "The conversation was compacted. The user's current request is: `...`. Continue working."]
~1-2k tokens
```

## Files

| File | What it contains |
|------|-----------------|
| `compaction_strategy_threshold.go` | The `Compact()` method — the flow described above |
| `compaction_utils.go` | Token estimation, calibration, summarization, truncation, state helpers |
| `contextguard.go` | `BeforeModelCallback` (calls Compact + persists heuristic), `AfterModelCallback` (persists real tokens), constants |
| `compaction_strategy_multiturn_test.go` | 47 multi-turn session simulations testing the full ADK flow |
| `compaction_strategy_singleshot_test.go` | Single-shot `Compact()` tests with realistic conversations |
| `contextguard_unit_test.go` | Unit tests for individual functions |

---

## ADK flow audit, simulator rewrite, and brutal stress tests

### Motivation

The first round of stress tests used a simulator that had several gaps relative to the real ADK execution flow. A line-by-line audit of ADK's `base_flow.go` revealed subtle but meaningful differences that needed to be corrected to ensure the tests truly validate production behavior.

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

### Simulator improvements

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

Before:
```
[model: FunctionCall("tool_1")] [user: FunctionResponse("tool_1")]
[model: FunctionCall("tool_2")] [user: FunctionResponse("tool_2")]
```

After:
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

This models the real ADK flow where a model chains tool calls sequentially. Each step triggers a full `BeforeModelCallback`, so `ContextGuard` can compact mid-chain.

#### 4. Immutable session events

The simulator no longer replaces `contents` after compaction. In real ADK, session events are append-only — `ContentsRequestProcessor` rebuilds from ALL events every time. The simulator now models this: `contents` is never modified by `beforeModel`, only appended to when new user/model/tool messages are added.

#### 5. Variable model response size

New `responseSize` field on `turnConfig` allows tests to specify how large the model's text response is. Default is 120 chars.

### Test suite (47 tests total)

#### Category 1: Stress tests (`TestStress_*`) — 27 tests

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

#### Category 3: Sequential tool chain tests — 3 tests

| Test | Window | Turns | Ratio | UsageMetadata | Tools/turn | Tool sizes | BeforeModelCallback calls/turn |
|------|--------|-------|-------|---------------|------------|------------|-------------------------------|
| `8k_SequentialToolChain` | 8k | 8 | 2.0 | yes | 5 sequential | 2k, 1.5k, 2.5k, 1k, 1.5k | 6 (1 user-msg + 5 tool-chain) |
| `200k_SequentialToolChain_LargeResponses` | 200k | 5 | 2.0 | yes | 5 sequential | 40k, 20k, 30k, 15k, 5k | 6 |
| `8k_SequentialChain_NoUsageMetadata` | 8k | 10 | 2.0 | no | 3 sequential | 1k, 800, 1.2k | 4 |

### Known limitation

`TestBrutal_8k_NoUsageMetadata_BeyondDefault` documents that when `tokenRatio > defaultHeuristicCorrectionFactor (2.0)` and the provider doesn't report `UsageMetadata`, the system has no way to learn the real ratio. Providers that don't report `UsageMetadata` should use `WithMaxTokens()` with a conservative override.

---

## Fix: compaction persistence across turns (critical production bug)

### The bug

Compaction had **zero effect** in production. ContextGuard compacted successfully on turn N, but on turn N+1 the full uncompacted context was back:

```
tokens=228967 threshold=120000  → compaction completed, newTokenEstimate=8756
tokens=231400 threshold=120000  → compaction completed, newTokenEstimate=8798
tokens=228967 threshold=120000  → compaction completed, newTokenEstimate=8756
... (every single turn, forever)
```

### Root cause: ADK's session events are immutable

`ContentsRequestProcessor` reads from `ctx.Session().Events().All()` — the **full, immutable, append-only** event list. Session events are never deleted or modified. The processor rebuilds `req.Contents` from scratch every time.

ContextGuard's `Compact()` rewrites `req.Contents` in memory, which affects the current LLM call. But those changes are **not persisted back to the session events**. On the next call, all events are loaded again and the compacted result is lost.

### What `injectSummary` did wrong

`injectSummary` only **prepended** the summary to `req.Contents` without removing the already-summarized events:

```
Before (broken):
  req.Contents = [event1, event2, ..., event50]     ← loaded by ADK from all events
  injectSummary → [SUMMARY, event1, event2, ..., event50]   ← WORSE than before
  tokenCount → exceeds threshold → compacts AGAIN
```

### The fix: watermark-based event stripping

The `contentsAtCompaction` watermark (which already existed in the sliding window strategy but was never used by threshold) now tracks how many session events existed when the summary was produced. On the next call, `injectSummary` uses it to **replace** the already-summarized events:

```
After (fixed):
  req.Contents = [event1, event2, ..., event50, event51, event52]   ← all events
  watermark = 50 (saved at last compaction)
  injectSummary → [SUMMARY, event51, event52]   ← old events stripped
  tokenCount → below threshold → no compaction needed ✓
```

### Changes

**`compaction_utils.go`** — `injectSummary` now takes `contentsAtCompaction int`. When > 0 and <= len(req.Contents), it strips the first N entries and prepends the summary before the remaining new entries.

**`compaction_strategy_threshold.go`** — Captures `totalSessionContents = len(req.Contents)` before any modification and calls `persistContentsAtCompaction(ctx, totalSessionContents)` after compaction.

**`compaction_strategy_sliding_window.go`** — Updated `injectSummary` call to pass `contentsAtLastCompaction`.

**`compaction_strategy_multiturn_test.go`** — Simulator no longer replaces `contents` after compaction (models immutable session events). Loop detection simplified to: compaction is a "loop" only if output >= input.

### Session state keys

```
Session
├── Events []Event                                    ← immutable, append-only
└── State map[string]any
    ├── __context_guard_summary_{agent}               ← summary text
    ├── __context_guard_contents_at_compaction_{agent} ← watermark (event count)
    ├── __context_guard_summarized_at_{agent}          ← token count (diagnostic)
    ├── __context_guard_real_tokens_{agent}            ← last PromptTokenCount
    └── __context_guard_last_heuristic_{agent}         ← last heuristic estimate
```

### Why the tests didn't catch it

The simulator treated `contents` as mutable state that got replaced after compaction, rather than as an append-only event log. With that shortcut, `injectSummary` never had to deal with the full event history. The fix to the simulator makes it model reality, where `injectSummary` must actively strip old events using the watermark.
