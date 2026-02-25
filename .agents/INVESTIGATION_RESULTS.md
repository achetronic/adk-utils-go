# Investigation Results: Compaction Retry Rounds & Giant Tool Responses

**Date**: February 2026  
**Test**: `TestCompactionInvestigation_RetryRoundsAndGiantResponses` in `compaction_simulation_test.go`  
**Context**: After implementing Proposals 1, 2, and 4, we investigated whether increasing compaction retry rounds solves remaining failing cases and how the system handles enormous tool_responses.

---

## Part 1: Retry Rounds vs `kube-3rounds / 8k ctx`

Does increasing the number of compaction attempts help the hardest failing case?

| Scenario | Attempts | Tokens Before | Tokens After | Reduction | Contents | Fits? |
|---|---|---|---|---|---|---|
| kube-3r/8k/attempts=1 | 1 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=2 | 2 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=3 | 3 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=5 | 5 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=10 | 10 | 24,126 | 8,068 | 66.6% | 24→9 | NO |
| kube-3r/8k/attempts=20 | 20 | 24,126 | 8,068 | 66.6% | 24→9 | NO |

### Conclusion

**More retry rounds provide zero benefit.** Compaction converges in a single attempt: 15 old messages are summarized into 1 summary, but the 9 remaining recent messages (3 tool_call/tool_response pairs with large JSON payloads from pods, describe, logs) weigh ~8,068 tokens on their own. Additional retry rounds only re-compact already-compacted content — the recent window is never touched.

---

## Part 2: Tool Response Size Scaling (10 pairs, 8k context)

At what tool_response size does compaction break down for a small context window?

| Scenario | Attempts | Tokens Before | Tokens After | Reduction | Contents | Fits? |
|---|---|---|---|---|---|---|
| 10pairs×0.5k/8k | 0 | 1,760 | 1,760 | 0.0% | 20→20 | YES |
| 10pairs×1k/8k | 0 | 3,010 | 3,010 | 0.0% | 20→20 | YES |
| 10pairs×2k/8k | 0 | 5,510 | 5,510 | 0.0% | 20→20 | YES |
| 10pairs×5k/8k | 1 | 13,010 | 1,280 | 90.2% | 20→2 | YES |
| 10pairs×10k/8k | 1 | 25,510 | 2,530 | 90.1% | 20→2 | YES |
| **10pairs×50k/8k** | **1** | **125,510** | **12,530** | **90.0%** | **20→2** | **NO** |
| **10pairs×200k/8k** | **1** | **500,510** | **50,030** | **90.0%** | **20→2** | **NO** |

### Conclusion

With 8k context, the system handles up to **~10k chars per tool_response** (10 pairs). At 50k chars, even after compacting 18 of 20 messages (90% reduction), the remaining 2 recent messages (1 tool_call + 1 tool_response) individually exceed the 8k window. The breakpoint is when a **single pair weighs more than the context window**.

---

## Part 3: Tool Response Size Scaling (10 pairs, 128k context)

Same scenario with a larger context window (128k).

| Scenario | Attempts | Tokens Before | Tokens After | Reduction | Contents | Fits? |
|---|---|---|---|---|---|---|
| 10pairs×5k/128k | 0 | 13,010 | 13,010 | 0.0% | 20→20 | YES |
| 10pairs×10k/128k | 0 | 25,510 | 25,510 | 0.0% | 20→20 | YES |
| 10pairs×50k/128k | 1 | 125,510 | 12,577 | 90.0% | 20→3 | YES |
| 10pairs×200k/128k | 1 | 500,510 | 50,030 | 90.0% | 20→2 | YES |
| **10pairs×500k/128k** | **1** | **1,250,510** | **125,030** | **90.0%** | **20→2** | **NO** |

### Conclusion

128k context handles tool_responses up to **~200k chars** (10 pairs). The pattern holds: failure occurs when a single remaining pair exceeds the window. At 500k chars per response (~125k tokens), even one pair overflows 128k.

---

## Part 4: Kube-like Conversations with Varying Rounds and Context Sizes

Realistic Kubernetes agent conversations (get pods, describe pod, get logs per round) across different context windows, all with 10 retry attempts.

| Scenario | Attempts Used | Tokens Before | Tokens After | Reduction | Contents | Fits? |
|---|---|---|---|---|---|---|
| kube-3r/8k | 10 | 24,126 | 8,068 | 66.6% | 24→9 | **NO** |
| kube-3r/32k | 0 | 24,126 | 24,126 | 0.0% | 24→24 | YES |
| kube-3r/128k | 0 | 24,126 | 24,126 | 0.0% | 24→24 | YES |
| kube-5r/8k | 10 | 40,210 | 8,068 | 79.9% | 40→9 | **NO** |
| kube-5r/32k | 1 | 40,210 | 8,068 | 79.9% | 40→9 | YES |
| kube-5r/128k | 0 | 40,210 | 40,210 | 0.0% | 40→40 | YES |
| kube-10r/8k | 10 | 80,420 | 8,068 | 90.0% | 80→9 | **NO** |
| kube-10r/32k | 1 | 80,420 | 8,068 | 90.0% | 80→9 | YES |
| kube-10r/128k | 0 | 80,420 | 80,420 | 0.0% | 80→80 | YES |
| kube-20r/8k | 10 | 160,840 | 8,068 | 95.0% | 160→9 | **NO** |
| kube-20r/32k | 1 | 160,840 | 8,068 | 95.0% | 160→9 | YES |
| kube-20r/128k | 1 | 160,840 | 24,223 | 84.9% | 160→26 | YES |

### Conclusions

1. **8k context always fails** for kube conversations, regardless of rounds or retry attempts. The 9 recent messages (~8k tokens) are irreducible.
2. **32k context handles everything** — even 20-round conversations compact in 1 attempt to fit.
3. **128k context rarely needs compaction** — only at 20 rounds does it trigger (1 attempt suffices).
4. The convergence to exactly 8,068 tokens / 9 contents across all 8k scenarios confirms the bottleneck: recent messages are untouched by compaction.

---

## Part 5: Single Giant Tool Response

Worst case: a conversation with just 1 user message + 1 model tool_call + 1 user tool_response + 1 model text reply. How does the system handle a single enormous tool_response?

| Scenario | Attempts | Tokens Before | Tokens After | Reduction | Contents | Fits? |
|---|---|---|---|---|---|---|
| 1resp×10k/8k | 0 | 2,528 | 2,528 | 0.0% | 4→4 | YES |
| 1resp×10k/128k | 0 | 2,528 | 2,528 | 0.0% | 4→4 | YES |
| 1resp×50k/8k | 1 | 12,528 | 36 | 99.7% | 4→2 | YES |
| 1resp×50k/128k | 0 | 12,528 | 12,528 | 0.0% | 4→4 | YES |
| 1resp×200k/8k | 1 | 50,028 | 36 | 99.9% | 4→2 | YES |
| 1resp×200k/128k | 0 | 50,028 | 50,028 | 0.0% | 4→4 | YES |
| 1resp×1000k/8k | 1 | 250,028 | 36 | 100.0% | 4→2 | YES |
| 1resp×1000k/128k | 1 | 250,028 | 36 | 100.0% | 4→2 | YES |

### Conclusions

**Positive surprise.** Even a 1M-char tool_response (250k tokens) is handled successfully — the compaction summarizes the old messages (including the giant response) into a tiny summary. This works because:

1. The giant response falls into the "old" portion (it's not the most recent message).
2. The summary replaces it entirely with a ~36-token summary.
3. Only 2 recent messages remain (the last model reply + summary).

The system only fails when the giant response is in the **recent window** (untouched by compaction).

---

## Overall Findings

### Two regimes

| Regime | Status | Solution |
|---|---|---|
| Old messages are large | **SOLVED** | Proposals 1+2+4: pair-boundary splitting + iterative retry + accurate token estimation |
| Recent messages are large | **UNSOLVED** | Requires Proposal 3: truncation of tool_responses in recent window |

### Key numbers

| Metric | Value |
|---|---|
| Retry convergence | Always in **1 attempt** (additional rounds provide zero benefit) |
| Max tool_response for 8k ctx | ~10k chars per response (if multiple pairs remain recent) |
| Max tool_response for 128k ctx | ~200k chars per response |
| Kube agent minimum context | **32k** (8k is structurally insufficient) |
| Single giant response | Handled up to **1M chars** (when it falls in old portion) |

### Recommendation

The current `maxCompactionAttempts = 3` is sufficient — compaction never needs more than 1 attempt. The remaining failures (`kube / 8k ctx`) require **Proposal 3** (truncation of giant tool_responses before summarization and optionally in the recent window). This is the only path to supporting small context windows with large tool_responses.
