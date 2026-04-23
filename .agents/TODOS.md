# TODOs

## Follow-ups from PR #12 (`feat: (openai) forward FunctionCallingConfig.Mode as tool_choice`)

Merged in `b4f327d`. These are deferred items discussed during review:

- [ ] **Test for `ModeUnspecified`** in `genai/openai/openai_test.go` and `genai/anthropic/anthropic_test.go`: verify that the zero value of `FunctionCallingConfigMode` leaves `params.ToolChoice` untouched. Today both adapters do the right thing (the value falls through the switch default and no branch assigns), but no test pins it down. If someone later refactors the switch into a map lookup or adds a catch-all `default` clause, the behaviour could silently change to a specific `tool_choice` value. A regression test per adapter, modelled exactly like the existing cases, closes this coverage hole for both providers in one go.

- [ ] **`slog.Warn` when `ModeAny` has `len(AllowedFunctionNames) > 1`** in `genai/openai/openai.go` and `genai/anthropic/anthropic.go`: neither provider's `tool_choice` accepts a list of allowed names, so both adapters currently fall back to a "force tool use, any tool" value (`"required"` for OpenAI, `{type: "any"}` for Anthropic). That is a silent narrowing of the caller's intent — someone writing `AllowedFunctionNames: ["a", "b"]` expects "the model must pick one of these two", not "the model must pick any available tool". A `slog.Warn` at each fallback point keeps behaviour unchanged but surfaces the mismatch in production logs the first time it happens, instead of requiring someone to diff the wire payload to notice. The comments in both adapters already document the limitation; the log turns documentation into a runtime signal.
