# Compaction: Token Threshold Strategy

How the threshold strategy decides when and how to compact a conversation.

## Compaction flow

| Step | What happens | Why |
|------|-------------|-----|
| 1. **Estimate tokens** | Calculate how many tokens the current conversation uses, including contents, system instruction, and tool definitions. If real token counts from the provider are available, a correction factor (`real / previous heuristic`) is applied. Otherwise, the `len/4` heuristic is multiplied by 2.5x | The heuristic alone underestimates by 2-3x on structured content (JSON, tool schemas). Tool definitions and InlineData (images) are now counted in the heuristic. |
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
currentHeuristic  = estimateTokens(req)                          # on the current request (contents + system + tools)
correction        = clamp(realTokens / lastHeuristic, 1.0, 5.0) # how much len/4 was off last time
calibrated        = currentHeuristic × correction                # scale to real-token space
result            = max(realTokens, calibrated)                  # never undercount
```

If no real tokens exist yet (first turn or provider doesn't report usage), the default factor of 2.5x is used.

## Buffer calculation

| Context window | Buffer | Threshold | Rationale |
|---------------|--------|-----------|-----------|
| >=200k (e.g. 200k) | 20k fixed | 180k | Large windows don't need proportional buffers |
| <200k (e.g. 8k) | 20% (1.6k) | 6.4k | Small windows need proportional headroom |

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
| `compaction_strategy_multiturn_test.go` | 91 multi-turn session simulations testing the full ADK flow |
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

### Test suite (91 tests total)

#### Category 1: Stress tests (`TestStress_*`) — 42 tests

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
| `200k_KubeAgent` | 200k | 20 | 2.5 | yes | kubectl_get_pods (15k) + describe (10k) + logs (25k) |
| `200k_MixedDebugSession` | 200k | 25 | 2.2 | yes | prometheus (20k) + sql (15k) + http (500) + grep (8k) mixed |
| `200k_PureToolStorm` | 200k | 30 | 2.0 | yes | 2/turn (5k each), minimal text |
| `200k_CodingAgent` | 200k | 15 | 2.5 | yes | read_file (8k) → edit_file (2k) → run_tests (12k) sequential |
| `1M_NormalConversation` | 1M | 50 | 2.0 | yes | 1/turn (10k) |
| `1M_HeavyTools` | 1M | 30 | 2.0 | yes | read_file (50k) + grep (20k) + tests (30k) |
| `4k_NormalConversation` | 4k | 20 | 1.8 | yes | none |
| `4k_WithSmallTools` | 4k | 10 | 1.8 | yes | 1/turn (500) |
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
| `8k_KubeAgent` | 8k | 10 | 2.0 | yes | kubectl_get_pods (5k) + describe (3k) + logs (8k) |
| `8k_MixedDebugSession` | 8k | 20 | 1.8 | yes | prometheus (3k) + sql (2k) + http (200) mixed |
| `8k_PureToolStorm` | 8k | 20 | 1.8 | yes | 1/turn (2k), minimal text |
| `8k_CodingAgent` | 8k | 10 | 2.0 | yes | read_file (3k) → edit_file (1k) → run_tests (4k) sequential |
| `200k_HeavyToolDefinitions` | 200k | 30 | 2.0 | yes | 2/turn (5k+8k), 20 MCP tool defs (2k schema each) |
| `200k_InlineImages` | 200k | 15 | 2.0 | yes | none, 100KB image/turn |
| `8k_HeavyToolDefinitions` | 8k | 20 | 2.0 | yes | 1/turn (1.5k), 10 MCP tool defs (1k schema each) |
| `8k_InlineSmallImages` | 8k | 10 | 2.0 | yes | none, 10KB image/turn |
| `CompactionNoInfiniteLoop` | 8k | 5 | 2.5 | yes | none, 12k system |

#### Category 2: Brutal tests (`TestBrutal_*`) — 49 tests

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
| `200k_ToolDefinitionsDominateWindow` | 200k | 20 | 2.5 | 50 MCP tools × 4k schema + 3 tool calls/turn |
| `200k_ToolDefinitionsHighRatio` | 200k | 15 | 3.5 | 30 MCP tools × 3k schema, high ratio |
| `200k_LargeInlineDocuments` | 200k | 3 | 2.0 | 500KB + 300KB inline PDFs |
| `200k_MultipleInlinePerTurn` | 200k | 8 | 2.0 | 3 images/turn (80k+60k+90k) |
| `200k_ToolDefsAndInlineImages` | 200k | 20 | 2.5 | 15 MCP tools + images every 3rd turn (production scenario) |
| `200k_ToolDefsAndInlineNoUsageMetadata` | 200k | 15 | 2.0 | 10 MCP tools + images + no calibration |
| `8k_ToolDefinitionsNoUsageMetadata` | 8k | 15 | 2.0 | 8 MCP tools (800ch schema) + no calibration |
| `8k_ToolDefsAndInlineCombined` | 8k | 20 | 2.0 | 5 MCP tools + images + tools + no calibration (all blind spots) |
| `200k_KubeAgent_30Rounds` | 200k | 30 | 2.5 | kubectl_get_pods (20k) + describe (15k) + logs (40k), kube agent pattern |
| `200k_MixedDebugSession_LongInvestigation` | 200k | 50 | 2.3 | prometheus (30k) + grafana (10k) + sql (20k) + http + grep mixed |
| `200k_PureToolStorm_HugeResponses` | 200k | 20 | 2.0 | fetch (50k) + process (20k) per turn, minimal text |
| `8k_PureToolStorm_50Turns` | 8k | 50 | 2.0 | 3k tool/turn, minimal text, 50 turns |
| `200k_CodingAgent_DeepRefactor` | 200k | 25 | 2.5 | 6 sequential tools/turn (read+grep+edit+read+edit+tests), large outputs |
| `8k_CodingAgent_NoUsageMetadata` | 8k | 15 | 2.0 | Sequential read→edit→tests, no calibration |
| `4k_ToolResponseExceedsWindow` | 4k | 3 | 2.0 | Single 20k tool response (5× window) |
| `4k_EveryTurnExceedsWindow` | 4k | 20 | 2.0 | 1k msg + 8k tool every turn |
| `4k_KubeAgent` | 4k | 10 | 2.0 | kubectl pattern in tiny window |
| `4k_CodingAgent` | 4k | 10 | 2.0 | Sequential read→edit→tests in tiny window |
| `4k_MixedDebug` | 4k | 15 | 2.0 | prometheus + sql + http mixed in 4k |
| `4k_PureToolStorm` | 4k | 20 | 2.0 | 2k tool/turn, minimal text in 4k |
| `1M_KubeAgent_ExtremeLongevity` | 1M | 100 | 2.0 | kubectl (30k+20k+50k) per turn, 100 turns |
| `1M_PureToolStorm_MonsterResponses` | 1M | 50 | 2.0 | 100k tool/turn, 50 turns |
| `1M_NoUsageMetadata` | 1M | 40 | 2.5 | 45k tools/turn, no calibration, 1M window |
| `200k_MagecProductionScenario` | 200k | 30 | 2.5 | 25 MCP tools + images + 2-4 tool calls/turn (exact production setup) |
| `200k_MagecProductionScenario_NoUsageMetadata` | 200k | 20 | 2.5 | 20 MCP tools + images + no calibration |
| `200k_SequentialEscalatingSizes` | 200k | 10 | 2.0 | 5 sequential tools escalating: 2k→5k→20k→40k→60k |
| `8k_SequentialEscalatingSizes` | 8k | 10 | 2.0 | 4 sequential tools escalating: 500→1.5k→3k→5k |
| `4k_SequentialEscalatingSizes` | 4k | 8 | 2.0 | 3 sequential tools escalating: 300→800→2k |
| `8k_SequentialToolChain` | 8k | 8 | 2.0 | 5 sequential tools (2k, 1.5k, 2.5k, 1k, 1.5k) |
| `200k_SequentialToolChain_LargeResponses` | 200k | 5 | 2.0 | 5 sequential tools (40k, 20k, 30k, 15k, 5k) |
| `8k_SequentialChain_NoUsageMetadata` | 8k | 10 | 2.0 | 3 sequential tools (1k, 800, 1.2k), no calibration |

### Known limitation

`TestBrutal_8k_NoUsageMetadata_BeyondDefault` documents that when `tokenRatio > defaultHeuristicCorrectionFactor (2.5)` and the provider doesn't report `UsageMetadata`, the system has no way to learn the real ratio. Providers that don't report `UsageMetadata` should use `WithMaxTokens()` with a conservative override.

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

---

## Fix: token estimation blind spots (production overflow)

### The bug

ContextGuard never fired compaction on a 200k context window with `claude-sonnet-4-6`. Anthropic returned `400 Bad Request: prompt is too long: 200819 tokens` — the context was exceeded without any compaction attempt.

```
time=2026-02-28T13:52:44Z level=ERROR msg="prompt is too long: 200819 tokens"
time=2026-02-28T13:55:11Z level=ERROR msg="prompt is too long: 200911 tokens"
```

No `ContextGuard [threshold]: threshold exceeded` log appeared — `tokenCount()` never reached the threshold.

### Root cause: three blind spots in token estimation

**1. Tool definitions not counted.** `estimateTokens()` counted `req.Contents` and `req.Config.SystemInstruction`, but completely ignored `req.Config.Tools`. Tool declarations (name, description, JSON parameter schemas) are serialized and sent with every LLM call. For agents with multiple tools or complex schemas (especially MCP-sourced), this can be thousands of tokens invisible to the heuristic.

**2. InlineData not counted.** `estimatePartTokens()` handled `Text`, `FunctionCall`, and `FunctionResponse` parts, but `InlineData` (arbitrary binary blobs: images, PDFs, audio, video) was counted as 0 tokens. A single base64-encoded image can be tens of thousands of tokens.

**3. Buffer boundary off-by-one.** `computeBuffer()` used `contextWindow > 200000` — for a 200k window exactly, this was `false`, so the 20% ratio applied instead of the fixed 20k buffer. Result: buffer = 40k, threshold = 160k instead of the intended buffer = 20k, threshold = 180k. Not the primary cause, but it moved the threshold 20k lower than designed.

**4. Default correction factor too conservative.** `defaultHeuristicCorrectionFactor` was 2.0, but real-world structured content (JSON tool schemas, markdown, non-ASCII) typically underestimates by 2-3x. With tool definitions invisible and a 2.0 factor, the heuristic couldn't reach the threshold even with calibration capped at 5x.

### Why calibration couldn't save it

In the production session, the heuristic saw ~24k tokens but Anthropic counted ~200k — a ratio of **8.3x**. The calibration system bridges this gap using `correction = realTokens / lastHeuristic`, but:

- The correction factor is capped at 5.0x to prevent anomalous turns from distorting future estimates
- 24k × 5.0 = 120k, still below the 160k threshold (with the old buffer)
- `max(realTokens, calibrated)` returns the stale `realTokens` from the previous turn, which hadn't exceeded the threshold yet
- The conversation grew by ~50k tokens in a single turn (new user message + tool results), pushing past 200k

The fundamental problem: when the heuristic is missing entire categories of token-consuming content (tools, inline data), the calibration correction factor cannot compensate fast enough.

### The fixes

**`compaction_utils.go` — `estimateToolTokens()` (new function)**

```go
func estimateToolTokens(tools []*genai.Tool) int {
    // Counts name + description + marshaled JSON schema for each FunctionDeclaration
}
```

`estimateTokens()` now calls `estimateToolTokens(req.Config.Tools)` alongside contents and system instruction.

**`compaction_utils.go` — `estimatePartTokens()` now counts InlineData**

```go
if part.InlineData != nil {
    total += len(part.InlineData.MIMEType) / 4
    total += len(part.InlineData.Data) / 4
}
```

`InlineData` is a generic `*genai.Blob` carrying raw bytes + MIME type — it covers images, PDFs, audio, video, and any other binary content. The fix counts all of them.

**`compaction_utils.go` — `computeBuffer()` boundary fix**

```go
// Before: contextWindow > largeContextWindowThreshold  (200k > 200k = false)
// After:  contextWindow >= largeContextWindowThreshold  (200k >= 200k = true)
```

200k windows now get the fixed 20k buffer (threshold = 180k) instead of the 20% buffer of 40k (threshold = 160k).

**`contextguard.go` — `defaultHeuristicCorrectionFactor` raised to 2.5**

From 2.0 to 2.5. Accounts for the typical 2-3x underestimation of `len/4` on structured content, plus the overhead from tool definitions and system prompts that tokenize denser than plain text. Conservative enough to avoid premature compaction, aggressive enough to catch real overflow before calibration data is available.

### Tests added

| Test | What it validates |
|------|-------------------|
| `TestEstimateToolTokens_Nil` | nil tools → 0 tokens |
| `TestEstimateToolTokens_WithDeclarations` | 2 tools with `ParametersJsonSchema` → >50 tokens |
| `TestEstimateToolTokens_WithParametersSchema` | Tools using `*genai.Schema` instead of raw JSON |
| `TestEstimateTokens_IncludesTools` | `estimateTokens(req with tools) > estimateTokens(req without tools)` |
| `TestEstimatePartTokens_InlineData` | 10k bytes of inline data → counted at ≥ `len/4` tokens |
| `TestComputeBuffer/exactly_at_threshold` | 200k window → 20k buffer (was 40k) |

### Why `WithMaxTokens(180000)` worked as a workaround

Setting `maxTokens = 180000` bypasses the registry lookup. `computeBuffer(180000)` → `180000 < 200000` → buffer = 36k → threshold = 144k. With `realTokens` from `afterModel` persisted at ~180k (previous turn), `tokenCount()` returned `max(180000, calibrated) ≥ 180000 > 144000` → compaction fired. The lower threshold made it possible for the stale `realTokens` alone to exceed it, sidestepping the heuristic blind spots entirely.
