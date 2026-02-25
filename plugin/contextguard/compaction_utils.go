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

package contextguard

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
)

const summarizeSystemPrompt = `You are summarizing a conversation to preserve context for continuing later.

Critical: This summary will be the ONLY context available when the conversation resumes. Assume all previous messages will be lost. Be thorough.

Required sections:

## Current State

- What was being discussed or worked on (exact user request if applicable)
- Current progress and what has been completed
- What was being addressed right now (incomplete work or open thread)
- What remains to be done or answered (specific, not vague)

## Key Information

- Facts, data, and specific details mentioned (names, dates, numbers, URLs, identifiers)
- User preferences, instructions, and constraints stated during the conversation
- Definitions, terminology, or domain knowledge established
- Any external resources, references, or sources mentioned

## Context & Decisions

- Decisions made during the conversation and why
- Alternatives that were considered and discarded (and why)
- Assumptions made
- Important clarifications or corrections that occurred
- Any blockers, risks, or open questions identified

## Exact Next Steps

Be specific. Don't write "continue with the task" — write exactly what should happen next, with enough detail that someone reading only this summary can pick up without asking questions.

Tone: Write as if briefing a colleague taking over mid-conversation. Include everything they would need to continue without asking questions. Write in the same language as the conversation.

Length: A dynamic word limit will be appended to this prompt at runtime based on the model's buffer size. Within that limit, err on the side of too much detail rather than too little. Critical context is worth the tokens.`

// --- Session state helpers ---

// loadSummary reads the running conversation summary from session state.
// Returns an empty string if no summary has been stored yet.
func loadSummary(ctx agent.CallbackContext) string {
	key := stateKeyPrefixSummary + ctx.AgentName()
	val, err := ctx.State().Get(key)
	if err != nil {
		return ""
	}
	s, _ := val.(string)
	return s
}

// persistSummary writes the summary and a diagnostic token count to session
// state. Errors are logged but not propagated.
func persistSummary(ctx agent.CallbackContext, summary string, tokenCount int) {
	keySummary := stateKeyPrefixSummary + ctx.AgentName()
	keySummarizedAt := stateKeyPrefixSummarizedAt + ctx.AgentName()
	if err := ctx.State().Set(keySummary, summary); err != nil {
		slog.Warn("ContextGuard: failed to persist summary", "error", err)
	}
	if err := ctx.State().Set(keySummarizedAt, tokenCount); err != nil {
		slog.Warn("ContextGuard: failed to persist token count", "error", err)
	}
}

// loadContentsAtCompaction reads the Content count recorded at the last
// sliding-window compaction. Returns 0 if no compaction has happened yet.
func loadContentsAtCompaction(ctx agent.CallbackContext) int {
	key := stateKeyPrefixContentsAtCompaction + ctx.AgentName()
	val, err := ctx.State().Get(key)
	if err != nil {
		return 0
	}
	if val == nil {
		return 0
	}
	switch v := val.(type) {
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

// persistContentsAtCompaction records the total Content count at which
// compaction was performed, so the next call can compute turns since then.
func persistContentsAtCompaction(ctx agent.CallbackContext, count int) {
	key := stateKeyPrefixContentsAtCompaction + ctx.AgentName()
	if err := ctx.State().Set(key, count); err != nil {
		slog.Warn("ContextGuard: failed to persist contents count", "error", err)
	}
}

// --- Summarization ---

// summarize calls the given LLM to produce a concise summary of the provided
// conversation contents. bufferTokens controls the dynamic word limit: the
// summary may use up to 50% of the buffer, converted to words at a 0.75
// words-per-token ratio. If the LLM returns an empty response, a mechanical
// fallback summary (truncated excerpts) is used instead.
func summarize(ctx context.Context, llm model.LLM, contents []*genai.Content, previousSummary string, bufferTokens int) (string, error) {
	maxOutputTokens := int32(float64(bufferTokens) * 0.50)
	maxWords := int(float64(maxOutputTokens) * 0.75)

	systemPrompt := summarizeSystemPrompt + fmt.Sprintf("\n\nKeep the summary under %d words.", maxWords)
	userPrompt := buildSummarizePrompt(contents, previousSummary)

	req := &model.LLMRequest{
		Model: llm.Name(),
		Contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: userPrompt}},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: systemPrompt}},
			},
			MaxOutputTokens: maxOutputTokens,
		},
	}

	var result string
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return "", fmt.Errorf("summarization LLM call failed: %w", err)
		}
		if resp != nil && resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part != nil && part.Text != "" {
					result += part.Text
				}
			}
		}
	}

	if result == "" {
		return buildFallbackSummary(contents, previousSummary), nil
	}

	return result, nil
}

// buildSummarizePrompt assembles the user-facing prompt sent to the LLM for
// summarization: a request to summarize, any previous summary for continuity,
// and a transcript of the conversation contents.
func buildSummarizePrompt(contents []*genai.Content, previousSummary string) string {
	var sb strings.Builder
	sb.WriteString("Provide a detailed summary of the following conversation.")
	sb.WriteString("\n\n")

	if previousSummary != "" {
		sb.WriteString("[Previous summary for context]\n")
		sb.WriteString(previousSummary)
		sb.WriteString("\n[End previous summary]\n\n")
		sb.WriteString("Incorporate the previous summary into your new summary, updating any information that has changed.\n\n")
	}

	sb.WriteString("[Conversation to summarize]\n")

	for _, content := range contents {
		if content == nil {
			continue
		}
		role := content.Role
		if role == "" {
			role = "unknown"
		}
		for _, part := range content.Parts {
			if part == nil {
				continue
			}
			if part.Text != "" {
				sb.WriteString(role)
				sb.WriteString(": ")
				sb.WriteString(part.Text)
				sb.WriteString("\n")
			}
			if part.FunctionCall != nil {
				sb.WriteString(role)
				sb.WriteString(": [called tool: ")
				sb.WriteString(part.FunctionCall.Name)
				sb.WriteString("]\n")
			}
			if part.FunctionResponse != nil {
				sb.WriteString(role)
				sb.WriteString(": [tool ")
				sb.WriteString(part.FunctionResponse.Name)
				sb.WriteString(" returned a result]\n")
			}
		}
	}
	sb.WriteString("[End of conversation]\n")

	return sb.String()
}

// buildFallbackSummary creates a best-effort summary without an LLM by
// concatenating the first 200 characters of each message. Used when the
// real summarization call fails or returns empty.
func buildFallbackSummary(contents []*genai.Content, previousSummary string) string {
	var sb strings.Builder
	if previousSummary != "" {
		sb.WriteString(previousSummary)
		sb.WriteString("\n\n---\n\n")
	}
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			if part != nil && part.Text != "" {
				role := content.Role
				if role == "" {
					role = "unknown"
				}
				sb.WriteString(role)
				sb.WriteString(": ")
				if len(part.Text) > 200 {
					sb.WriteString(part.Text[:200])
					sb.WriteString("...")
				} else {
					sb.WriteString(part.Text)
				}
				sb.WriteString("\n")
			}
		}
	}
	return sb.String()
}

// --- Token estimation ---

// estimatePartTokens returns a rough token count for a single Part using
// the ~4 chars per token heuristic. It accounts for Text, FunctionCall
// (name + args), and FunctionResponse (name + response).
func estimatePartTokens(part *genai.Part) int {
	if part == nil {
		return 0
	}
	total := 0
	if part.Text != "" {
		total += len(part.Text) / 4
	}
	if part.FunctionCall != nil {
		total += len(part.FunctionCall.Name) / 4
		for k, v := range part.FunctionCall.Args {
			total += len(k) / 4
			total += len(fmt.Sprintf("%v", v)) / 4
		}
	}
	if part.FunctionResponse != nil {
		total += len(part.FunctionResponse.Name) / 4
		total += len(fmt.Sprintf("%v", part.FunctionResponse.Response)) / 4
	}
	return total
}

// estimateTokens returns a rough token count for the entire LLM request
// (contents + system instruction) using the ~4 chars per token heuristic.
func estimateTokens(req *model.LLMRequest) int {
	total := estimateContentTokens(req.Contents)
	if req.Config != nil && req.Config.SystemInstruction != nil {
		for _, part := range req.Config.SystemInstruction.Parts {
			total += estimatePartTokens(part)
		}
	}
	return total
}

// estimateContentTokens returns a rough token count for a slice of Content
// entries using the ~4 chars per token heuristic. It counts all part types
// (Text, FunctionCall, FunctionResponse).
func estimateContentTokens(contents []*genai.Content) int {
	total := 0
	for _, content := range contents {
		if content == nil {
			continue
		}
		for _, part := range content.Parts {
			total += estimatePartTokens(part)
		}
	}
	return total
}

// computeBuffer returns the token buffer for a given context window:
// fixed 20k for windows >200k, 20% for smaller ones.
func computeBuffer(contextWindow int) int {
	if contextWindow > largeContextWindowThreshold {
		return largeContextWindowBuffer
	}
	return int(float64(contextWindow) * smallContextWindowRatio)
}

// --- Content splitting and summary injection ---

// findSplitIndex determines where to split Contents into "old" (to be
// summarized) and "recent" (to keep verbatim). It walks backwards from
// the end, accumulating tokens until recentBudget is reached.
func findSplitIndex(contents []*genai.Content, recentBudget int) int {
	tokens := 0
	for i := len(contents) - 1; i >= 0; i-- {
		if contents[i] == nil {
			continue
		}
		for _, part := range contents[i].Parts {
			tokens += estimatePartTokens(part)
		}
		if tokens >= recentBudget {
			if i < len(contents)-2 {
				return safeSplitIndex(contents, i+1)
			}
			return safeSplitIndex(contents, len(contents)-2)
		}
	}
	if len(contents) > 2 {
		return safeSplitIndex(contents, len(contents)/2)
	}
	return safeSplitIndex(contents, 1)
}

// safeSplitIndex adjusts a candidate split index so it never lands in the
// middle of a tool_call/tool_response pair. It first tries walking backwards
// to find a clean boundary (text message or start of a tool pair). If that
// would regress past the original candidate, it walks forward instead to
// the next pair boundary. This ensures that in pure-tool conversations the
// split lands between complete pairs rather than collapsing to index 0.
func safeSplitIndex(contents []*genai.Content, idx int) int {
	if idx <= 0 || idx >= len(contents) {
		return idx
	}

	origIdx := idx

	idx = walkBackToPairBoundary(contents, idx)

	if idx <= 0 {
		idx = walkForwardToPairBoundary(contents, origIdx)
	}

	if idx <= 0 {
		idx = 1
	}
	if idx >= len(contents) {
		idx = len(contents) - 1
	}

	return idx
}

// walkBackToPairBoundary walks backwards from idx looking for a position
// that is not in the middle of a tool_call/tool_response pair. Returns the
// adjusted index, or 0 if it exhausted all messages.
func walkBackToPairBoundary(contents []*genai.Content, idx int) int {
	for idx > 0 {
		c := contents[idx]
		if c == nil {
			break
		}

		if c.Role == "user" && contentHasFunctionResponse(c) {
			idx--
			continue
		}

		if c.Role == "model" && contentHasFunctionCall(c) {
			idx--
			continue
		}

		break
	}
	return idx
}

// walkForwardToPairBoundary walks forward from idx to the nearest complete
// tool pair boundary. A pair is [model:FunctionCall, user:FunctionResponse].
// The function advances past the current incomplete pair and stops right
// after the tool_response, which is a valid split point (between two pairs).
func walkForwardToPairBoundary(contents []*genai.Content, idx int) int {
	for idx < len(contents) {
		c := contents[idx]
		if c == nil {
			break
		}

		if c.Role == "model" && contentHasFunctionCall(c) {
			idx++
			continue
		}

		if c.Role == "user" && contentHasFunctionResponse(c) {
			idx++
			break
		}

		break
	}
	return idx
}

// contentHasFunctionResponse reports whether a Content entry contains at
// least one FunctionResponse part (a tool_result block).
func contentHasFunctionResponse(c *genai.Content) bool {
	for _, part := range c.Parts {
		if part != nil && part.FunctionResponse != nil {
			return true
		}
	}
	return false
}

// contentHasFunctionCall reports whether a Content entry contains at
// least one FunctionCall part (a tool_use block).
func contentHasFunctionCall(c *genai.Content) bool {
	for _, part := range c.Parts {
		if part != nil && part.FunctionCall != nil {
			return true
		}
	}
	return false
}

// injectSummary prepends a summary content block to the request. If a
// summary block already exists as the first message, it is left untouched
// to avoid duplicates.
func injectSummary(req *model.LLMRequest, summary string) {
	summaryText := fmt.Sprintf("[Previous conversation summary]\n%s\n[End of summary — conversation continues below]", summary)

	if len(req.Contents) > 0 && req.Contents[0] != nil &&
		req.Contents[0].Role == "user" && len(req.Contents[0].Parts) > 0 {
		first := req.Contents[0]
		if first.Parts[0] != nil && first.Parts[0].Text != "" &&
			strings.HasPrefix(first.Parts[0].Text, "[Previous conversation summary]") {
			return
		}
	}

	summaryContent := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: summaryText},
		},
	}
	req.Contents = append([]*genai.Content{summaryContent}, req.Contents...)
}

// replaceSummary rewrites req.Contents to [summary + recentContents],
// discarding everything older than the split point.
func replaceSummary(req *model.LLMRequest, summary string, recentContents []*genai.Content) {
	summaryContent := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: fmt.Sprintf("[Previous conversation summary]\n%s\n[End of summary — conversation continues below]", summary)},
		},
	}
	req.Contents = append([]*genai.Content{summaryContent}, recentContents...)
}
