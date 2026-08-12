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
- **Why:** consumers (baifo, Magec, ...) pick a provider by config and expect the
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

The adapter is OpenAI-pure by default: with no dialect configured it reads
no provider field and sends none, which matches OpenAI's own API, where
reasoning models never expose the reasoning text in Chat Completions.
Providers that diverge from the documented OpenAI wire shape plug a dialect
in (O10).

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

### O8 - Streaming requests always set `stream_options.include_usage=true`

OpenAI's Chat Completions API only emits a final usage chunk on the SSE stream
when the caller opts in via `stream_options.include_usage`. Without it, the
`ChatCompletionAccumulator`'s `Usage` stays zero, and `buildStreamFinalResponse`
already reads that accumulator (`convertUsageMetadata(acc.Usage)`) into the
terminal `LLMResponse`'s `UsageMetadata`: so the plumbing was there, but the
opt-in was missing.

- **Decision:** `generateStream` sets `params.StreamOptions.IncludeUsage =
  param.NewOpt(true)` before `NewStreaming`. The final `LLMResponse` on the
  streaming path now carries populated `UsageMetadata`, matching the non-
  streaming path (`generate` -> `convertResponse`).
- **Why:** consumers that price token spend (Langfuse, billing dashboards)
  need usage on **every** turn, not just the non-streaming ones. Forcing
  callers to pick between streaming UX and usage accounting turned the
  adapter into an all-or-nothing choice.
- **Symmetry with the non-streaming path:** `generate` returns usage via
  `convertResponse(resp.Usage)`. Under D5 the two paths must be
  behaviourally interchangeable; the include-usage opt-in restores that.
- **Providers without the field:** Ollama and other OpenAI-compat servers
  that don't implement `stream_options.include_usage` ignore it (see the
  documented server behaviour), and O6 still drops the resulting all-zero
  usage block. No behaviour change for those.
- **Interrupted streams:** OpenAI's docs note that a broken stream may drop
  the terminal usage chunk. That surfaces as an accumulator with zero usage
  and O6 drops it: consistent with the pre-change behaviour on those failed
  turns.
- **Tests:** `wire_test.go::TestWireBody_StreamRequestsUsage` fires one
  streaming request through the wire-capture fixture and asserts the JSON
  body has `stream: true` and `stream_options.include_usage: true`.

### O9 - Cached prompt tokens map to `CachedContentTokenCount`

OpenAI Chat Completions reports cache hits in
`PromptTokensDetails.CachedTokens`. The count is a subset of `PromptTokens`, not
an additional token bucket. `convertUsageMetadata` maps it to genai's
`CachedContentTokenCount` while leaving `PromptTokenCount` and `TotalTokenCount`
inclusive and unchanged.

- **Why:** ADK's OpenTelemetry instrumentation emits
  `gen_ai.usage.cache_read.input_tokens` from this field. Cost-aware consumers
  can then apply the provider's discounted cache-read rate without changing the
  total context usage.
- **Missing details:** compatible providers that omit the field leave it at
  zero; genai's `omitempty` keeps the detail absent on serialisation.

---

### O10 - Provider divergences plug in through a Dialect of small capabilities

The adapter knows no provider-specific wire shape by itself. A `Dialect`,
plugged through `Config.Dialect`, opts into the areas it needs by
implementing capability interfaces; a nil dialect keeps the adapter
OpenAI-pure.

- **Why capabilities, not one big interface:** the divergence inventory of
  the compatible providers (DeepSeek, Kimi, Mistral, OpenRouter, vLLM, xAI,
  Ollama) clusters into seven areas, and a provider touches one to three of
  them, never all. One fat interface would force every dialect to
  stub four areas it does not use; five small interfaces let a dialect
  implement only what it needs, and let the adapter grow a sixth area later
  without touching the existing dialects.
- **The five capability interfaces:**
  - `ToolIDNormalizer`: the tool_call_id wire shape. OpenAI allows up to 40
    characters from [a-zA-Z0-9_-] (the built-in O1 rule); Mistral rejects
    anything that is not exactly 9 alphanumeric characters. The dialect owns
    the shape; the adapter keeps the wire-to-original mapping so ADK keeps
    seeing its own IDs on both the tool_calls and the tool messages that
    refer back to them.
  - `ParamsAdjuster`: a last pass over the outgoing request params, fired
    after the adapter finishes building them and merging ExtraBody, with the
    stream flag in hand. For providers that reject combinations the OpenAI
    schema accepts: xAI's reasoning models refuse stop sequences and the
    penalty knobs, and some gateways refuse stream_options.
  - `ReasoningDecoder`: the reasoning fields the schema does not define, on
    ingest (O12 covers the streaming half).
  - `ReasoningEncoder`: the same on egress, as assistant-message extra
    fields in the native egress mode (O11).
  - `UsageDecoder`: usage buckets reported outside the standard object,
    folded into the metadata the adapter already built. DeepSeek puts
    prompt_cache_hit_tokens at the usage root, not in
    prompt_tokens_details.
  - `ThinkingMapper`: the provider-native reasoning-effort knob. Implemented,
    the dialect owns the mapping from genai's thinking level entirely; not
    implemented means the typed OpenAI field reasoning_effort is used.
    OpenRouter's effort lives in a reasoning object at the request root and
    vLLM and Qwen use enable_thinking, so the typed field cannot serve them.
    The dialect's knob wins over a reasoning key a caller set in ExtraBody:
    the effort is the dialect's area.
  - `EgressPolicy`: the replay shapes the provider tolerates. Resolved once in
    `New`: a requested mode the dialect vetoes is replaced by an accepted one
    and logged, so the caller still picks any tolerated shape and nothing the
    provider rejects reaches the wire. The DeepSeek dialect pins the replay
    to native, because thinking mode rejects a tool-call history whose
    assistant turns lack the reasoning key.
- **Pipeline touch points, in request order:** tool IDs are normalised while
  the messages are built; the thinking level is applied in
  `applyGenerationConfig`; `ParamsAdjuster` fires last in
  `buildChatCompletionParams`, after ExtraBody, so it sees the exact body
  the wire gets; ingest decodes in `convertResponse` and `generateStream`;
  usage decodes in `decodeUsageMetadata`, only when the response reported
  tokens; egress encodes in `convertContentToMessages`, in the native mode
  only. The order is documented on the `Dialect` type itself.
- **Capabilities are asserted once in `New`** and held as fields, so the
  conversion path is a nil check, not a type assertion per request.
- **What the adapter still owns:** the pipeline rules a dialect cannot
  change. Thought Parts never reach the reply text; reasoning attaches to
  assistant turns only; a reasoning-only turn sends no message; and
  `Config.ReasoningEgress` is the policy applied on top of whatever a
  dialect encodes. A native-mode dialect without an encoder degrades to
  think tags rather than dropping, so a replayed session keeps its trace.
- **Why a Name() method:** the base `Dialect` interface is otherwise empty;
  `Name()` gives logs and errors something to point at without forcing any
  behaviour.
- **Tests:** `dialect_capability_test.go` pins each capability with a
  dialect-shaped test double (Mistral-style 9-char IDs correlating across
  the tool pair on the wire, an xAI-style adjuster stripping stop, a
  DeepSeek-style usage fold); `wire_test.go` pins the nil-dialect default:
  no provider fields on the wire, stray thoughts fold into think tags;
  `reasoning_test.go` pins the veto: the DeepSeek dialect forces native
  over think tags and omit, and omit stands against OpenRouter.

### O11 - Three dialects ship: `TextDialect` for plain text, `DeepSeek` for the strict replay, `OpenRouter` for `reasoning_details`

**TextDialect** carries reasoning as a single plain-text field on the
assistant message. Read and write are split on purpose: what varies between
providers is the name they *emit*, while they all accept `reasoning_content`
back. `ReadFields` (default `["reasoning_content", "reasoning"]`, covering
DeepSeek, Kimi, Mistral, OpenRouter's text shape and newer vLLM) is an
ordered probe list for ingest; the first present non-empty string wins.
`WriteField` (default `reasoning_content`) is the field the reasoning goes
out in. Both are knobs on the struct, so a provider with its own field
names plugs in with no new code.

**DeepSeek** layers a replay veto on the text dialect: it embeds
`TextDialect`, so the read and write fields are the same, and implements
`EgressPolicy` pinning the replay to native. DeepSeek in thinking mode
rejects a tool-call history whose assistant turns lack the reasoning key
with a 400, so think tags and omit are refused at construction and the
override is logged.

**OpenRouter** carries OpenRouter's structured shape: a `reasoning_details`
array of typed blocks (`reasoning.text` with an optional signature,
`reasoning.summary`, `reasoning.encrypted`) alongside a plain-text copy.
The encrypted variant is what models that do not expose readable reasoning
hand back, and the case the plain-text field cannot express at all.

- **Storage:** one thought Part per block, in wire order, with the block
  kept verbatim in `Part.PartMetadata` under the exported
  `ReasoningDetailMetadataKey` (`openai.reasoning_detail`). `Part.Text`
  mirrors the readable text so consumers filtering on `Thought` still see
  reasoning; an encrypted block yields a Part with empty `Text` and
  metadata only, the same convention as A3's redacted thinking block. One
  Part per block keeps the required order implicit in Part order.
- **Blocks are opaque:** decoded as `map[string]any` and never re-typed,
  reordered, filtered or rewritten, nulls included. OpenRouter requires the
  replayed sequence to match what the model produced, the block schema is
  open, and the documented `format` list keeps growing, so an unknown block
  type or vendor key still round-trips.
- **No gating knob:** the dialect is all-or-nothing at construction. A
  backend that never sends the array never has one replayed, because the
  encoder only writes the array when a Part actually carries a block.
- **Native mode only for the array:** a block is replayed as an array
  element only in native mode, the one shape with a field to hold it. In
  think-tag or omit mode its readable text degrades into the plain-text
  shape (think tags) or drops (omit); an encrypted block has no text and is
  lost there, the unavoidable cost of a backend that cannot take the array.
- **Blocks beat the plain-text field:** when both arrive they describe the
  same reasoning and the blocks carry strictly more, so the string is
  skipped on ingest. On egress a Part feeds either the array or the string,
  never both; a turn mixing blocked and plain thoughts sends both fields,
  each fed by its own Parts.
- **Tests:** `dialect_reasoning_test.go` covers decode (blocks, text
  fallback, malformed array), encode (verbatim blocks, mixed turns, empty
  input) and the accumulators; `reasoning_test.go` covers the byte-identical
  round trip and the degradation outside native mode.

### O12 - Streamed reasoning is accumulated by the dialect, not read off the SDK accumulator

`openai-go`'s `ChatCompletionAccumulator` merges chunks field by field and
keeps **no raw JSON** on the message it aggregates (`accumulateDelta` states
that it ignores the JSON field). Every reasoning field is non-standard and
lives only in raw JSON, so `acc.Choices[0].Message.RawJSON()` is the empty
string at the end of a real stream.

- **Consequence:** probing that aggregate silently yields nothing, so a
  streamed turn's terminal response would carry no thought Part at all. The
  terminal response is what becomes history, so streamed reasoning would be
  lost even though the partial responses carried it.
- **Decision:** `generateStream` asks the dialect's decoder for a fresh
  accumulator via `NewAccumulator`, feeds it every delta's decoded Parts,
  and passes it to `buildStreamFinalResponse`, which renders `Parts()`
  ahead of the answer. Without a reasoning decoder there is no accumulator
  and nothing is decoded. The merge semantics belong to the dialect:
  TextDialect concatenates the texts; OpenRouter merges `reasoning_details`
  blocks by their reported `index` (`text`, `summary` and `data`
  concatenate; everything else takes the newest non-empty value, so a
  signature arriving late is kept and a null never erases one), and blocks
  without an index append in arrival order.
- **Do not "simplify" this back:** reading reasoning off the SDK aggregate
  looks equivalent and is not. A test that builds an accumulator with
  `json.Unmarshal` of a whole response passes while the live path is broken,
  because unmarshalling populates raw JSON a stream never has. That is
  exactly how this gap went unnoticed.
- **Tests:** `wire_test.go` drives real chunks through the public streaming
  path against a fake SSE server and asserts the terminal response carries
  the concatenated reasoning, and that the nil-dialect default surfaces no
  thought Parts even when the stream sends a reasoning field.

### O13 - Provider extensions at the request root via `ExtraBody`

OpenAI-compatible providers add top-level request fields that Chat Completions
does not define: OpenRouter alone has `reasoning`, `provider`, `transforms` and
`plugins`. `Config.ExtraBody map[string]any` is merged into the root of every
request body through the same `SetExtraFields` escape hatch the reasoning fields
use, on the streaming and non-streaming paths alike.

- **Why config, not per request:** these describe the endpoint a Model points
  at, decided once at construction like `BaseURL`. ADK's
  `GenerateContentConfig` has no field to carry them, and adding typed options
  per provider would drag the whole OpenRouter surface into this library.
- **Why a map, not `[]option.RequestOption`:** exposing the SDK's option type
  would leak `openai-go` types into `Config`, which otherwise uses only stdlib
  types. A map keeps the public surface provider-agnostic.
- **Collisions:** a key matching a field the adapter sets replaces it on the
  wire, since extra fields win at marshal time. Documented as an extension
  point rather than defended against: filtering keys would be its own surprise,
  and the escape hatch is deliberate.
- **Copied at construction:** `New` copies the caller's map, so a caller reusing
  or mutating its own map cannot race with a request in flight, and Model state
  stays read-only during conversion (D2).
- **Tests:** `wire_test.go` covers a nested object landing at the body root.

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
