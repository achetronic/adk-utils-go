# TODOs

Open work only. Design rationale for what already exists lives in
`DECISIONS.md`; this file is just what is still pending.

## LLM adapter integration coverage

The integration tier (`-tags=integration`, see DECISIONS D6) is uneven between
providers.

- [ ] **OpenAI: no real-API (step B) run yet.** The schema step (A) passes and
      the step-B test exists, but it has never run against the real API (no key used
      so far). Run `OPENAI_API_KEY=... go test -tags=integration ./genai/openai/...`
      and confirm a non-4xx, then keep the test.
- [ ] **OpenAI: no semantic-rule probing.** Anthropic was probed against the
      live API to find cross-message rules the schema does not express (orphaned
      tool_use, thinking placement, trailing whitespace, must-end-on-user). The same
      probing has not been done for OpenAI, so its only integration test is the
      no-args tool call. Probe the live API for OpenAI's equivalent rules and add a
      canary+adapter test per confirmed rule.

## Anthropic semantic-rule gaps

Rules the live API enforces that the adapter does not yet handle (see DECISIONS
A10 for why some are left to the caller):

- [ ] **Orphaned `tool_result`.** `repairMessageHistory` drops a `tool_use`
      with no following `tool_result`, but not the reverse: a `tool_result` with no
      preceding `tool_use` still reaches the API and is rejected. The fix is
      symmetric to the existing tool_use repair.
- [ ] **Conversation ending on an assistant turn** (DECISIONS A10). Left as a
      caller contract on purpose (prefill is model-dependent), but `repairMessageHistory`
      can produce it by dropping a trailing orphan tool_use. Decide whether to warn
      when it happens.

## Follow-ups from PR #12 (`feat: (openai) forward FunctionCallingConfig.Mode as tool_choice`)

- [ ] **Test for `ModeUnspecified`** in `genai/openai/openai_test.go` and `genai/anthropic/anthropic_test.go`: verify that the zero value of `FunctionCallingConfigMode` leaves `params.ToolChoice` untouched. Today both adapters do the right thing (the value falls through the switch default and no branch assigns), but no test pins it down. If someone later refactors the switch into a map lookup or adds a catch-all `default` clause, the behaviour could silently change to a specific `tool_choice` value. A regression test per adapter, modelled exactly like the existing cases, closes this coverage hole for both providers in one go.

- [ ] **`slog.Warn` when `ModeAny` has `len(AllowedFunctionNames) > 1`** in `genai/openai/openai.go` and `genai/anthropic/anthropic.go`: neither provider's `tool_choice` accepts a list of allowed names, so both adapters currently fall back to a "force tool use, any tool" value (`"required"` for OpenAI, `{type: "any"}` for Anthropic). That is a silent narrowing of the caller's intent: someone writing `AllowedFunctionNames: ["a", "b"]` expects "the model must pick one of these two", not "the model must pick any available tool". A `slog.Warn` at each fallback point keeps behaviour unchanged but surfaces the mismatch in production logs the first time it happens, instead of requiring someone to diff the wire payload to notice. The comments in both adapters already document the limitation; the log turns documentation into a runtime signal.

## Follow-ups from the ADK v1.4.0 upgrade

- [ ] **Streaming memory tools via `functiontool.NewStreaming`** in `tools/memory/toolset.go`: ADK v1.3.0 added `functiontool.NewStreaming[TArgs](cfg Config, handler StreamingFunc[TArgs]) (tool.Tool, error)` with `type StreamingFunc[TArgs any] func(tool.Context, TArgs) iter.Seq2[string, error]`, which lets a tool emit partial results in chunks instead of returning one consolidated JSON blob. Today the memory tools are built with `functiontool.New()` and return their result all at once. `search_memory` in particular would benefit from streaming entries to the model as they are retrieved (better latency/UX on large result sets), and `save_to_memory`/`update_memory`/`delete_memory` could stream progress. **Value: high, effort: low**: it is swapping the constructor on an existing, well-tested toolset. Keep the non-streaming variant for backends/clients that don't need it.

- [ ] **Populate the new `model.LLMResponse` fields in the LLM adapters** (`genai/openai/openai.go`, `genai/anthropic/anthropic.go`): ADK v1.4.0 added `InputTranscription`/`OutputTranscription` (`*genai.Transcription`) and `SessionResumptionHandle` (`string`) to `model.LLMResponse`. Both adapters currently leave them at their zero value, which is correct for plain request/response OpenAI/Anthropic calls (no audio transcription, no live session resumption). This is an **enabler, not a bug**: only actionable if/when the adapters grow audio/voice support or live (`runner.RunLive`) bidirectional streaming. Tracked here so the integration point isn't forgotten when that work starts.
