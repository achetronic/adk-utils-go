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
| `compaction_strategy_multiturn_test.go` | 25 multi-turn session simulations testing the full ADK flow |
| `compaction_strategy_singleshot_test.go` | Single-shot `Compact()` tests with realistic conversations |
| `contextguard_unit_test.go` | Unit tests for individual functions |
