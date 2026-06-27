# Pinned OpenAPI specs

These are the upstream OpenAPI specs the integration tests validate request
bodies against (the schema step, before any real API call). They are committed
so the check is deterministic and offline; refresh them with `./update.sh` and
commit the result when you want to validate against newer specs.

| File            | Source                                                                                              | Notes                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `openai.yaml`   | `github.com/openai/openai-openapi` (`openapi.yaml`, master)                                         | Declares `openapi: 3.1.0` but uses the 3.0 `nullable:` keyword.       |
| `anthropic.json`| Stainless spec URL read from `anthropic-sdk-typescript/.stats.yml` (`openapi_spec_url`)             | JSON despite the `.json` name being the spec; OpenAPI 3.1.0.          |

Validator choice is per provider because of the OpenAI spec defect above:

- OpenAI: `kin-openapi`, which parses the document as 3.0 and tolerates
  `nullable`. `libopenapi` refuses to compile it ("`nullable` not supported in
  3.1").
- Anthropic: `libopenapi-validator`, which does real 3.1 validation.

The schema step catches structural errors (missing required fields, wrong
types, bad enums). It does NOT catch Anthropic's cross-message semantic rules
(every `tool_use` needs a following `tool_result`, thinking blocks only in
assistant turns): only the real-API step does.
