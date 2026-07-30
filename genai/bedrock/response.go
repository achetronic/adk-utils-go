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

import (
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// convertOutput converts a non-streaming Converse response into an
// LLMResponse.
func (m *Model) convertOutput(out *bedrockruntime.ConverseOutput) (*model.LLMResponse, error) {
	msg, err := outputMessage(out.Output)
	if err != nil {
		return nil, err
	}

	parts, err := convertMessageToParts(msg)
	if err != nil {
		return nil, err
	}

	var guardrailTrace *types.GuardrailTraceAssessment
	if out.Trace != nil {
		guardrailTrace = out.Trace.Guardrail
	}

	// When a guardrail blocks the turn, tag it with an invisible marker
	// rather than replacing the text: the actual guardrail message is
	// preserved for display, while the marker lets a later request redact
	// the triggering user turn (see redactBlockedInputs). guardrailBlockedStage
	// picks the input- or output-stage message when Bedrock returned more
	// than one (see markGuardrailBlocked). Content must stay non-nil and the
	// turn non-partial: ADK's flow loop requires the last yielded event of a
	// turn to be a final (non-partial) response, and a nil Content here
	// would make ADK skip storing the event entirely, which for the
	// streaming path can leave a dangling partial chunk as the last event
	// and trip ADK's "last event is not final" internal error. See B8.
	if out.StopReason == types.StopReasonGuardrailIntervened {
		parts = markGuardrailBlocked(parts, guardrailBlockedStage(guardrailTrace))
	}

	resp := &model.LLMResponse{
		Content: &genai.Content{
			Role:  genai.RoleModel,
			Parts: parts,
		},
		FinishReason: convertStopReason(out.StopReason),
	}

	if out.Usage != nil {
		resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32derefOr(out.Usage.InputTokens),
			CandidatesTokenCount: int32derefOr(out.Usage.OutputTokens),
		}
	}
	if meta := guardrailMetadata(out.StopReason, guardrailTrace); meta != nil {
		resp.CustomMetadata = meta
	}

	return resp, nil
}

// convertMessageToParts converts a Bedrock Message's content blocks into
// genai Parts.
func convertMessageToParts(msg *types.Message) ([]*genai.Part, error) {
	parts := make([]*genai.Part, 0, len(msg.Content))

	for _, block := range msg.Content {
		switch v := block.(type) {
		case *types.ContentBlockMemberText:
			parts = append(parts, &genai.Part{Text: v.Value})

		case *types.ContentBlockMemberToolUse:
			args, err := documentToMap(v.Value.Input)
			if err != nil {
				return nil, fmt.Errorf("bedrock: decoding tool use input for %q: %w", derefOr(v.Value.Name), err)
			}
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   derefOr(v.Value.ToolUseId),
					Name: derefOr(v.Value.Name),
					Args: args,
				},
			})

		case *types.ContentBlockMemberReasoningContent:
			// Provider-specific reasoning trace (e.g. Claude/Nova
			// extended thinking via Bedrock). Surface it as a thought
			// part so consumers that care about it (or want to echo it
			// back on the next turn) can; convertPart drops thought
			// parts on the way out for any non-assistant role, and
			// silently for assistant role too since Converse's common
			// content-block set has no input variant for it - the
			// model is expected to regenerate reasoning rather than
			// have it replayed.
			if rc, ok := v.Value.(*types.ReasoningContentBlockMemberReasoningText); ok {
				parts = append(parts, &genai.Part{
					Text:    derefOr(rc.Value.Text),
					Thought: true,
				})
			}

		default:
			// Other block kinds (image, document, video, citations,
			// search result, guard content echo) aren't expected in a
			// model-authored turn; skip rather than fail the response.
		}
	}

	return parts, nil
}

// convertStopReason maps Bedrock's StopReason to genai.FinishReason.
func convertStopReason(reason types.StopReason) genai.FinishReason {
	switch reason {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return genai.FinishReasonStop
	case types.StopReasonToolUse:
		return genai.FinishReasonStop
	case types.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	case types.StopReasonGuardrailIntervened, types.StopReasonContentFiltered:
		return genai.FinishReasonSafety
	case types.StopReasonMalformedModelOutput, types.StopReasonMalformedToolUse:
		return genai.FinishReasonOther
	case types.StopReasonModelContextWindowExceeded:
		return genai.FinishReasonMaxTokens
	default:
		return genai.FinishReasonUnspecified
	}
}

// guardrailMetadata surfaces Bedrock's guardrail intervention status and
// (when trace was requested) the full guardrail trace into
// LLMResponse.CustomMetadata, under the "bedrockGuardrail" key. LLMResponse
// has no dedicated field for this, and silently dropping it would hide the
// one thing a guardrail-enabled caller most needs to know: whether the
// guardrail touched this turn. Takes the GuardrailTraceAssessment directly
// (rather than the enclosing ConverseTrace/ConverseStreamTrace, which are
// distinct, non-interchangeable types with an identically-shaped Guardrail
// field) so both generate() and generateStream() can share one
// implementation.
func guardrailMetadata(stopReason types.StopReason, guardrailTrace *types.GuardrailTraceAssessment) map[string]any {
	intervened := stopReason == types.StopReasonGuardrailIntervened
	if !intervened && guardrailTrace == nil {
		return nil
	}

	meta := map[string]any{
		"intervened": intervened,
	}

	if guardrailTrace != nil {
		if raw, err := json.Marshal(guardrailTrace); err == nil {
			var decoded any
			if json.Unmarshal(raw, &decoded) == nil {
				meta["trace"] = decoded
			}
		}
	}

	return map[string]any{"bedrockGuardrail": meta}
}

// documentToMap decodes a Bedrock document.Interface (a tool's input, as
// echoed back by the model) into a genai.FunctionCall-compatible
// map[string]any.
func documentToMap(doc interface{ UnmarshalSmithyDocument(any) error }) (map[string]any, error) {
	if doc == nil {
		return map[string]any{}, nil
	}
	var v map[string]any
	if err := doc.UnmarshalSmithyDocument(&v); err != nil {
		return nil, err
	}
	if v == nil {
		v = map[string]any{}
	}
	return v, nil
}

func derefOr(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func int32derefOr(i *int32) int32 {
	if i == nil {
		return 0
	}
	return *i
}
