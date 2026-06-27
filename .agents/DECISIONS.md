# DECISIONS.md

Design decisions for the `genai/*` LLM adapters (`genai/openai` and
`genai/anthropic`). Each entry records *what* was decided and *why*, so the next
person (or agent) doesn't re-litigate it or "fix" it back into a bug.

`adk-utils-go` is a public, general-purpose library with several consumers
(baifo, Magec, and others). That frames every decision below:

- **Wire-schema rules of a provider live in that provider's adapter.** Anything
  the provider's API strictly requires (ID shapes, schema casing,
  object-vs-null payloads, thinking-block placement) is the adapter's job.
- **Application/history policy does NOT live here.** What to send, token
  hygiene, cross-provider mixing, stripping stale thoughts for app reasons -
  that belongs in the consumer (e.g. baifo), not in this library.
- **No consumer-specific assumptions.** The adapters cannot assume how a
  consumer calls them (concurrency, caching, reuse of `genai.Content`), so
  converters are read-only over their input.

---

## Cross-provider decisions (apply to both adapters)

### D1 - Empty tool payloads serialise to `{}`, never `null`

A tool with no parameters leaves `genai.FunctionCall.Args` nil; a tool that
returns nothing leaves `genai.FunctionResponse.Response` nil. `json.Marshal` of
a nil Go map produces the literal `null`. On the wire, a tool call's
`arguments` and a tool message/result's content are JSON-object strings; `null`
is not an object.

- **Decision:** normalise nil/empty/unmarshalable payloads to the canonical
  empty object `{}` in both adapters, for both `FunctionCall.Args` **and**
  `FunctionResponse.Response`.
- **Why:** strict server-side parsers on OpenAI-compatible backends reject
  `null` where they expect an object - Qwen's chat template on vLLM/llama.cpp
  raises a Jinja error. The official OpenAI endpoint and Anthropic both *accept*
  `null`, so this is **portability for strict OpenAI-compatible runtimes**, not
  an API requirement. Origin is PR #21 (which only fixed `FunctionCall.Args` on
  the OpenAI side).
- **Canonical vs ADK:** `genai.FunctionCall.Args` is `map[string]any` with a
  `json:"args,omitempty"` tag, so Gemini-native ADK never emits `null` (it omits
  the field). Google's OpenAI-compat path (adk-python `lite_llm.py`) actually
  emits `null` via `json.dumps(None)`. We deliberately diverge from `lite_llm`:
  `{}` is strictly more portable and matches the `omitempty` semantics (absent =
  empty object, never null).
- **Implementation:** a single exported helper `common.MarshalToolPayload(any)
  (json.RawMessage, error)` in the **`genai/common`** package, used by both
  adapters for both `FunctionCall.Args` and `FunctionResponse.Response`. It works
  on the marshalled bytes and **never mutates** the caller's `genai.Content` (see
  D2). It fast-paths a payload that is already a `json.RawMessage` (treating an
  empty one as the benign "no payload" case), which only Anthropic tool inputs
  can be; that branch is inert for OpenAI, which only ever passes maps.
- **Single source of truth (RESOLVED: shared package):** the helper lives in
  `genai/common`, not duplicated per adapter. The decision (D1/D5) is identical
  across providers, so it is implemented exactly once; this removes the risk of
  the two adapters drifting. Cost accepted: both adapters import `genai/common`.
- **Error handling (RESOLVED: propagate):** a genuine `json.Marshal` failure is
  **propagated**, not swallowed: both adapters return it from
  `convertContentToMessage(s)` and the run fails. A nil/empty payload (including
  an empty `json.RawMessage`) is the benign "no payload" case and still
  normalises to `{}`; only a value that genuinely cannot be marshalled errors.
  This is in service of D5: a payload that breaks one provider must break the
  other identically, never silently degrade on one and not the other. (A
  marshal failure on a real `map[string]any` is effectively impossible, but the
  contract is now explicit and symmetric.)
- **Tests:** `common/payload_test.go` pins the helper's unit contract (100%
  coverage); each adapter's `tool_payload_test.go` pins the *integration* (that
  `convertContentToMessage(s)` actually routes both tool sides through it),
  including a **canary** that fails if a future change normalises the payload by
  mutating the shared input in place.

### D2 - Converters are read-only over their `genai.Content`

The `convertContentToMessage(s)` functions (and everything they call) must not
write back into the `genai.Content` / `genai.Part` / `FunctionCall` /
`FunctionResponse` they receive.

- **Why:** as a public library we cannot assume the caller isn't sharing or
  reusing that `Content` (persisted session, multi-agent history, concurrent
  conversion). A converter that mutates its input is a data race waiting to
  happen and a surprise for every consumer. **This is why PR #21's in-place
  `part.FunctionCall.Args = make(...)` was rejected** in favour of D1's
  byte-level normalisation.

### D3 - `tool_choice` mapping from `FunctionCallingConfig.Mode`

Both adapters translate `genai.GenerateContentConfig.ToolConfig.FunctionCallingConfig`
into the provider-native `tool_choice` field. Full table and rationale live in
`AGENTS.md` ("LLM Adapters - tool_choice Mapping"). Key points:

- `ModeAny` with multiple `AllowedFunctionNames` falls back to "force any tool"
  (`required` / `{type: any}`) because neither provider accepts a list of
  allowed names. See `TODOS.md` for the pending `slog.Warn` at that fallback.
- The zero value (`ModeUnspecified`) leaves `tool_choice` unset in both.

### D4 - Optional HTTP-level injection via `HTTPOptions`

Both `Config` structs carry an `HTTPOptions{ Client *http.Client; Headers
http.Header }` forwarded to the SDK via `option.WithHTTPClient` / header
options.

- **Why:** lets consumers inject a custom `http.Client` (OAuth transports,
  proxies, test servers) and extra headers **without** baking any consumer-
  specific auth/billing logic into the library. The library stays agnostic;
  the domain hacks (e.g. baifo's Anthropic OAuth transport) live in the
  consumer.

### D5 - Providers must be behaviourally interchangeable under adk-go

The adapters do **not** have to be identical: each has its own `Config` and
constructor (OpenAI has `tool_call_id` hashing, Anthropic has prompt caching,
thinking blocks, OAuth-via-`HTTPOptions`, etc.). What they MUST guarantee is
that, once constructed and handed to adk-go as a `model.LLM`, swapping one for
the other does not break a running agent. Same inputs -> behaviourally
equivalent, working outputs.

- **What "interchangeable" means concretely:**
  - Both implement `GenerateContent` (streaming and non-streaming) and yield a
    `genai.Content` with the same shape conventions: `Role = model`, text as
    text Parts, tool calls as `FunctionCall` Parts with a populated `ID`/`Name`,
    usage in `UsageMetadata` (or nil, never a bogus zero block).
  - A tool round-trip survives a provider swap: the `FunctionCall.ID` an
    adapter emits must be the same value its own `FunctionResponse`/tool_result
    path expects back, so an agent loop (call -> result -> next turn) works on
    either provider. ID *encodings* differ (O1 hash vs A1 sanitise) but each is
    internally consistent and reversible where it needs to be.
  - Wire quirks that would otherwise make one provider reject a history the
    other accepts are normalised in the adapter, not pushed onto the caller:
    empty tool payloads (D1), `tool_choice` semantics (D3), thinking-block
    placement (A3), orphaned tool_use repair (A2).
  - Failure modes are symmetric: a payload that can't be marshalled fails on
    both, not one (D1 error policy).
- **What is allowed to differ:** constructor/config surface, caching, billing/
  auth transport, provider-only features (reasoning blocks). These are opt-in at
  construction time and don't change the `model.LLM` runtime contract.
- **Why:** consumers (baifo, Magec, …) pick a provider by config and expect the
  agent to "just work". A divergence where "this provider does X and the other
  does Y and that's why it breaks" is a bug in *this* library, not the
  consumer's problem. Every cross-provider decision above (D1-D4) exists to hold
  this invariant.
- **Practical rule when editing an adapter:** if a change makes the two adapters
  behave differently in a way a downstream agent could observe (message shape,
  ID round-trip, error vs success on the same input), either apply it to both or
  justify here why the difference is invisible to adk-go.

### D6 - Three test tiers, escalating cost

1. **Unit/conversion** (default `go test`): offline, deterministic. Assert the
   adapter fills the right fields.
2. **Wire body** (default `go test`): a local `httptest` server captures the
   exact bytes the SDK puts on the wire (`captureBody` / `captureBodyFor`).
   Catches "I emit `null` where `{}` is required" without network.
3. **Integration** (`-tags=integration`, excluded from default): step A
   validates the captured body against the pinned OpenAPI spec
   (`genai/testdata/openapi/`), free and offline; step B, only if A passes and
   the API key env var is set, sends to the real API and requires non-4xx.

Why real-API at all: the SDK and a fake server both just serialise, neither
enforces server rules, so neither proves the request is accepted. Only step B
does. Why schema-before-real: fail cheap on structural errors before spending
tokens. Validator is per provider: OpenAI's spec declares 3.1 but uses the 3.0
`nullable` keyword, so `kin-openapi` (parses as 3.0) validates it and
`libopenapi` cannot; Anthropic's spec is clean 3.1 and uses `libopenapi`.
Schema validation only catches structural errors, not Anthropic's cross-message
rules (tool_use needs a following tool_result); step B is the only guard for
those.

---

## OpenAI adapter (`genai/openai`)

Targets: OpenAI proper + OpenAI-compatible servers (Ollama, vLLM, LocalAI,
LiteLLM, ...). Skews towards portability, not just the official endpoint.

### O1 - `tool_call_id` <= 40 chars via hash + reverse map

OpenAI rejects `tool_call_id` longer than 40 chars. `normalizeToolCallID`
hashes over-long IDs (sha256, round-trippable) to `tc_` + hex, and stores the
mapping in `toolCallIDMap` (guarded by `toolCallIDMapMu` `sync.RWMutex`) so
`tool_result` can be correlated back to the original ADK ID via
`denormalizeToolCallID`.

- **Why a map + mutex:** the hash is one-way; without storing the pair we
  couldn't recover the original ID. The mutex is the one piece of per-Model
  mutable state - conversion itself stays pure (see D2).

### O2 - Role mapping: `model` -> `assistant`

`convertRole` maps genai `model` to OpenAI `assistant`; `user` and `system`
pass through unchanged.

### O3 - Object schemas always get a `properties` field + lowercase types

`convertToFunctionParams` runs `lowercaseTypes` (genai emits `type` in
upper-case, e.g. `STRING`) and `ensureObjectProperties` (an `"object"` schema
with no `properties` gets an empty one) before sending.

- **Why:** OpenAI / strict structured-output validators reject an `object`
  schema that lacks `properties`, and upper-case type names aren't valid JSON
  Schema.

### O4 - Structured output uses `strict: true`

When `ResponseSchema` is set, the adapter emits a `json_schema` response format
with `Strict: true` (and a bare `json_object` format when only
`ResponseMIMEType == "application/json"`).

### O5 - User messages take the plain-string path unless media is present

`buildUserMessage` emits a simple string `content` when there are no media
parts, and only switches to the array-of-parts shape when there are images.

- **Why:** the array shape breaks OpenAI-compatible servers that don't support
  multi-modal input (older Ollama etc.); the simple path keeps those working.

### O6 - Usage with zero total tokens is dropped

`convertUsageMetadata` returns `nil` when `TotalTokens == 0`, so the adapter
doesn't report an all-zero usage block (e.g. Ollama not returning usage).

### O7 - Nil args/response -> `{}` (see D1)

Both `FunctionCall.Args` and `FunctionResponse.Response` go through
`common.MarshalToolPayload`.

---

## Anthropic adapter (`genai/anthropic`)

### A1 - tool IDs must match `^[a-zA-Z0-9_-]+$` via `sanitizeToolID`

Anthropic rejects tool_use IDs with characters outside `[a-zA-Z0-9_-]`.
`sanitizeToolID` replaces an invalid ID with `toolu_` + sha256 (16 bytes hex).
Applied to both tool_use and tool_result IDs so they still match afterwards.

### A2 - Repair history: every tool_use needs a matching tool_result

`repairMessageHistory` drops orphaned `tool_use` blocks (those without a
`tool_result` in the immediately following user message) before sending.

- **Why:** Anthropic rejects a request where an assistant `tool_use` isn't
  followed by its `tool_result`; ADK histories can end mid-tool-call (cancel,
  compaction, agent switch). This is a *wire-shape* repair (not content
  policy), so it belongs in the adapter.

### A3 - Thinking blocks: echo back in assistant turns, drop from non-assistant

- On the way *in* (`convertResponse`): a `ThinkingBlock` becomes a thought
  `Part` (`Thought=true`, `Text`=reasoning, `ThoughtSignature`=signature); a
  `RedactedThinkingBlock` becomes a thought Part with empty Text and the opaque
  blob in `ThoughtSignature`.
- On the way *out* (`convertContentToMessage`): thought Parts are rebuilt as
  their dedicated block types and placed before `tool_use`, **but only in
  assistant messages**. Under any other role they are dropped.
- **Why drop under user role:** Anthropic returns 400 if thinking/redacted
  blocks appear outside assistant messages. ADK's contents processor rewrites
  foreign-agent events as user-role "For context:" content and passes non-text
  parts through verbatim; those foreign reasoning signatures are useless (and
  illegal) here, so we drop them rather than let the API bounce the request.
  This is a *wire-schema* rule (where blocks are legal), not history policy;
  app-level stale-thought hygiene stays in the consumer.

### A4 - Prompt caching is ON by default, 3 breakpoints (see caching.go)

`applyCacheControl` (unless `disablePromptCaching`) stamps `cache_control:
ephemeral` on 3 prefixes: last tool def, last system block, and the last
cacheable block of the last message (walking past thinking/redacted blocks,
which can't carry cache_control). Called last in `buildMessageParams` (after
repair/cache ordering is final). Full rationale in `caching.go`'s header.

### A5 - Usage accounting sums the three cache buckets

With caching active, Anthropic splits the prompt into `InputTokens` (un-cached
suffix), `CacheReadInputTokens` and `CacheCreationInputTokens`.
`PromptTokenCount` is the sum of all three (the model processed the whole
prompt); `CachedContentTokenCount` carries the read-hit portion for cost-aware
consumers.

### A6 - Tool input schema `Type` forced to `"object"`; default max tokens 4096

`convertTools` sets `inputSchema.Type = "object"` unconditionally (Anthropic
requires it). `buildMessageParams` defaults `MaxTokens` to 4096 when the caller
doesn't set `MaxOutputTokens` (Anthropic requires a non-zero `max_tokens`).

### A7 - Role mapping: unknown roles fall back to `user`

`convertRoleToAnthropic`: `user`->`user`, `model`->`assistant`, anything else
-> `user` (Anthropic only has user/assistant).

### A8 - Nil args/response -> `{}` (see D1)

Both `FunctionCall.Args` (tool_use.input) and `FunctionResponse.Response`
(tool_result content) go through `common.MarshalToolPayload`, so tool_use and
tool_result stay symmetric.

### A9 - Trailing whitespace on a final assistant turn is trimmed

Anthropic rejects a request whose final assistant content ends in whitespace
("final assistant content cannot end with trailing whitespace"): in prefill the
model continues from those exact tokens and a trailing space is ambiguous.
`trimFinalAssistantWhitespace` right-trims the last text block of a trailing
assistant message after `repairMessageHistory`; a block left empty is dropped
(empty text blocks are also rejected). Verified against the live API.

### A10 - Conversation ending in an assistant turn is NOT forced to user (caller contract)

Some models reject a conversation that ends with an assistant message: "This
model does not support assistant message prefill. The conversation must end with
a user message." This is model-dependent, not a universal wire rule (prefill is
a real Anthropic feature other models accept), so the adapter does NOT rewrite
it: dropping the final assistant loses content, and synthesising a user turn
fabricates input. It is the caller's responsibility not to end on an assistant
turn unless the target model supports prefill. Note `repairMessageHistory` can
leave a history ending in assistant (after dropping a trailing orphan tool_use);
that is the most likely way to hit this. Observed on `claude-sonnet-4-6`.
