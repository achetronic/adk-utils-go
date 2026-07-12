// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bedrock

import "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

// GuardrailConfig attaches an Amazon Bedrock Guardrail to every Converse
// call made by a Model.
//
// This is deliberately the Converse-API shape, not the raw InvokeModel
// shape. The two are not interchangeable: InvokeModel applies a guardrail
// via the X-Amzn-Bedrock-GuardrailIdentifier/-GuardrailVersion HTTP headers
// *plus* a top-level "amazon-bedrock-guardrailConfig" body field (the
// request fails if the header is set without the body field, or vice
// versa), and for input tagging it additionally requires wrapping the
// natural-language spans of the prompt in
// <amazon-bedrock-guardrails-guardContent_*> markers. Converse avoids all of
// that: GuardrailConfig is a normal, typed request field and applies to the
// whole message set by default - see "Use a guardrail with the Converse
// API" in the Bedrock User Guide.
type GuardrailConfig struct {
	// Identifier is the guardrail ID or ARN.
	Identifier string

	// Version is the guardrail version (a numeric version string, or
	// "DRAFT" for the working draft).
	Version string

	// Trace controls whether Bedrock returns a detailed guardrail trace
	// (which policies were evaluated, what they matched, what action was
	// taken) alongside the response. One of "enabled", "enabled_full", or
	// "" (disabled, the default). When enabled, the adapter surfaces the
	// trace in LLMResponse.CustomMetadata["bedrockGuardrail"]["trace"]; see
	// guardrailMetadata in response.go.
	Trace string
}

// toConverseConfig converts a GuardrailConfig into the typed Converse API
// request field.
func (g *GuardrailConfig) toConverseConfig() *types.GuardrailConfiguration {
	if g == nil || g.Identifier == "" {
		return nil
	}

	cfg := &types.GuardrailConfiguration{
		GuardrailIdentifier: &g.Identifier,
		GuardrailVersion:    &g.Version,
	}

	switch g.Trace {
	case "enabled":
		cfg.Trace = types.GuardrailTraceEnabled
	case "enabled_full":
		cfg.Trace = types.GuardrailTraceEnabledFull
	default:
		cfg.Trace = types.GuardrailTraceDisabled
	}

	return cfg
}

// toConverseStreamConfig converts a GuardrailConfig into the typed
// ConverseStream request field. ConverseStream uses a distinct
// GuardrailStreamConfiguration type (not GuardrailConfiguration) because it
// additionally supports StreamProcessingMode; this adapter always requests
// synchronous processing, which evaluates the guardrail against each chunk
// as it streams rather than buffering the whole response first.
func (g *GuardrailConfig) toConverseStreamConfig() *types.GuardrailStreamConfiguration {
	if g == nil || g.Identifier == "" {
		return nil
	}

	cfg := &types.GuardrailStreamConfiguration{
		GuardrailIdentifier:  &g.Identifier,
		GuardrailVersion:     &g.Version,
		StreamProcessingMode: types.GuardrailStreamProcessingModeSync,
	}

	switch g.Trace {
	case "enabled":
		cfg.Trace = types.GuardrailTraceEnabled
	case "enabled_full":
		cfg.Trace = types.GuardrailTraceEnabledFull
	default:
		cfg.Trace = types.GuardrailTraceDisabled
	}

	return cfg
}

// Intervened reports whether a Bedrock Guardrail blocked or altered this
// turn, reading the metadata stamped by guardrailMetadata. Returns false for
// any response that wasn't produced by this adapter, or that didn't have a
// guardrail attached.
func Intervened(customMetadata map[string]any) bool {
	g, ok := customMetadata["bedrockGuardrail"].(map[string]any)
	if !ok {
		return false
	}
	intervened, _ := g["intervened"].(bool)
	return intervened
}
