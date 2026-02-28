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
	"encoding/json"
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

// persistRealTokens writes the real token count from the provider to session
// state. Called by the AfterModelCallback.
func persistRealTokens(ctx agent.CallbackContext, tokens int) {
	key := stateKeyPrefixRealTokens + ctx.AgentName()
	if err := ctx.State().Set(key, tokens); err != nil {
		slog.Warn("ContextGuard: failed to persist real token count", "error", err)
	}
}

// loadRealTokens reads the real token count from session state. Returns 0 if
// no count has been recorded yet (first turn or provider doesn't report usage).
func loadRealTokens(ctx agent.CallbackContext) int {
	key := stateKeyPrefixRealTokens + ctx.AgentName()
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

// persistLastHeuristic writes the heuristic token estimate of the request
// that was sent to the LLM. Called by beforeModel AFTER compaction so it
// reflects the final request. Used to compute a calibration factor.
func persistLastHeuristic(ctx agent.CallbackContext, tokens int) {
	key := stateKeyPrefixLastHeuristic + ctx.AgentName()
	if err := ctx.State().Set(key, tokens); err != nil {
		slog.Warn("ContextGuard: failed to persist last heuristic", "error", err)
	}
}

// loadLastHeuristic reads the heuristic estimate from the previous LLM call.
// Returns 0 if not yet recorded.
func loadLastHeuristic(ctx agent.CallbackContext) int {
	key := stateKeyPrefixLastHeuristic + ctx.AgentName()
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

// resetCalibration clears the real token count and last heuristic from
// session state. Called after compaction so the next turn starts fresh
// instead of applying a stale correction factor derived from a much
// larger pre-compaction request.
func resetCalibration(ctx agent.CallbackContext) {
	keyReal := stateKeyPrefixRealTokens + ctx.AgentName()
	keyHeuristic := stateKeyPrefixLastHeuristic + ctx.AgentName()
	if err := ctx.State().Set(keyReal, 0); err != nil {
		slog.Warn("ContextGuard: failed to reset real tokens", "error", err)
	}
	if err := ctx.State().Set(keyHeuristic, 0); err != nil {
		slog.Warn("ContextGuard: failed to reset last heuristic", "error", err)
	}
}

// truncateForSummarizer trims the conversation contents so that the
// summarization prompt itself doesn't exceed the summarizer LLM's context
// window. It keeps the most recent messages (freshest context) and drops
// the oldest ones when the total exceeds 80% of contextWindow. The 80%
// budget leaves room for the system prompt, previous summary, and output.
func truncateForSummarizer(contents []*genai.Content, contextWindow int) []*genai.Content {
	budget := int(float64(contextWindow) * 0.80)
	total := estimateContentTokens(contents)
	if total <= budget {
		return contents
	}

	for len(contents) > 2 && estimateContentTokens(contents) > budget {
		contents = contents[1:]
	}
	return contents
}

// tokenCount returns the best available token estimate for the current
// request. It uses a calibrated heuristic to close the timing gap between
// AfterModelCallback (where real tokens are recorded) and BeforeModelCallback
// (where the check runs on a potentially larger request).
//
// Algorithm:
//  1. Compute the heuristic on the current request (reflects tool results
//     added since the last LLM call).
//  2. If we have both real tokens and a heuristic from the previous call,
//     derive a correction factor and apply it to the current heuristic.
//  3. Return max(realTokens, calibratedHeuristic) so neither stale real
//     tokens nor an inaccurate heuristic can cause an undercount.
//  4. If no real tokens are available, fall back to the raw heuristic
//     scaled by a conservative default factor.
func tokenCount(ctx agent.CallbackContext, req *model.LLMRequest) int {
	currentHeuristic := estimateTokens(req)
	realTokens := loadRealTokens(ctx)

	if realTokens <= 0 {
		result := int(float64(currentHeuristic) * defaultHeuristicCorrectionFactor)
		slog.Debug("ContextGuard [tokenCount]: no calibration data, using default factor",
			"agent", ctx.AgentName(),
			"heuristic", currentHeuristic,
			"factor", defaultHeuristicCorrectionFactor,
			"result", result,
		)
		return result
	}

	lastHeuristic := loadLastHeuristic(ctx)
	var calibrated int
	var correction float64
	if lastHeuristic > 0 {
		correction = float64(realTokens) / float64(lastHeuristic)
		if correction < 1.0 {
			correction = 1.0
		}
		if correction > maxCorrectionFactor {
			correction = maxCorrectionFactor
		}
		calibrated = int(float64(currentHeuristic) * correction)
	} else {
		correction = defaultHeuristicCorrectionFactor
		calibrated = int(float64(currentHeuristic) * correction)
	}

	result := calibrated
	if realTokens > calibrated {
		result = realTokens
	}

	slog.Debug("ContextGuard [tokenCount]: calibrated estimate",
		"agent", ctx.AgentName(),
		"heuristic", currentHeuristic,
		"realTokens", realTokens,
		"lastHeuristic", lastHeuristic,
		"correction", fmt.Sprintf("%.2f", correction),
		"calibrated", calibrated,
		"result", result,
	)
	return result
}

// --- Todo state helpers ---

// TodoItem represents a single task tracked in session state.
type TodoItem struct {
	Content    string `json:"content"`
	Status     string `json:"status"`
	ActiveForm string `json:"active_form,omitempty"`
}

// loadTodos reads the todo list from session state. Returns nil if no todos
// are stored. Supports both []TodoItem and []any (from JSON deserialization).
func loadTodos(ctx agent.CallbackContext) []TodoItem {
	val, err := ctx.State().Get("todos")
	if err != nil || val == nil {
		return nil
	}

	switch v := val.(type) {
	case []TodoItem:
		return v
	case []any:
		var items []TodoItem
		for _, raw := range v {
			m, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			item := TodoItem{}
			if c, ok := m["content"].(string); ok {
				item.Content = c
			}
			if s, ok := m["status"].(string); ok {
				item.Status = s
			}
			if a, ok := m["active_form"].(string); ok {
				item.ActiveForm = a
			}
			if item.Content != "" {
				items = append(items, item)
			}
		}
		return items
	}
	return nil
}

// --- Summarization ---

// summarize calls the given LLM to produce a concise summary of the provided
// conversation contents. bufferTokens controls the dynamic word limit: the
// summary may use up to 50% of the buffer, converted to words at a 0.75
// words-per-token ratio. If the LLM returns an empty response, a mechanical
// fallback summary (truncated excerpts) is used instead.
//
// When todos is non-empty, the todo list is appended to the summarization
// prompt so it can be preserved across compaction boundaries.
func summarize(ctx context.Context, llm model.LLM, contents []*genai.Content, previousSummary string, bufferTokens int, todos []TodoItem) (string, error) {
	maxOutputTokens := int32(float64(bufferTokens) * 0.50)
	maxWords := int(float64(maxOutputTokens) * 0.75)

	systemPrompt := summarizeSystemPrompt + fmt.Sprintf("\n\nKeep the summary under %d words.", maxWords)
	userPrompt := buildSummarizePrompt(contents, previousSummary, todos)

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
// a transcript of the conversation contents, and optionally the current todo
// list for preservation.
func buildSummarizePrompt(contents []*genai.Content, previousSummary string, todos []TodoItem) string {
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

	if len(todos) > 0 {
		sb.WriteString("\n[Current todo list]\n")
		for _, t := range todos {
			fmt.Fprintf(&sb, "- [%s] %s\n", t.Status, t.Content)
		}
		sb.WriteString("[End todo list]\n\n")
		sb.WriteString("Include these tasks and their statuses in your summary under a dedicated \"## Todo List\" section. ")
		sb.WriteString("Instruct the resuming assistant to restore them using the `todos` tool to continue tracking progress.\n")
	}

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
	if part.InlineData != nil {
		total += len(part.InlineData.MIMEType) / 4
		total += len(part.InlineData.Data) / 4
	}
	return total
}

// estimateTokens returns a rough token count for the entire LLM request
// (contents + system instruction + tool definitions) using the ~4 chars per
// token heuristic. Tool definitions (function declarations with their JSON
// schemas) are sent with every request and can consume significant tokens,
// especially with MCP-sourced tools or complex parameter schemas.
func estimateTokens(req *model.LLMRequest) int {
	total := estimateContentTokens(req.Contents)
	if req.Config != nil {
		if req.Config.SystemInstruction != nil {
			for _, part := range req.Config.SystemInstruction.Parts {
				total += estimatePartTokens(part)
			}
		}
		total += estimateToolTokens(req.Config.Tools)
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

// estimateToolTokens returns a rough token count for the tool definitions
// attached to the request. Tool declarations (name, description, and
// parameter JSON schemas) are serialized and sent with every LLM call.
// For agents with many tools or complex schemas (e.g. MCP-sourced tools),
// this can amount to thousands of tokens that must be counted to avoid
// underestimating the total prompt size.
func estimateToolTokens(tools []*genai.Tool) int {
	total := 0
	for _, tool := range tools {
		if tool == nil {
			continue
		}
		for _, fd := range tool.FunctionDeclarations {
			if fd == nil {
				continue
			}
			total += len(fd.Name) / 4
			total += len(fd.Description) / 4
			if fd.ParametersJsonSchema != nil {
				data, err := json.Marshal(fd.ParametersJsonSchema)
				if err == nil {
					total += len(data) / 4
				}
			} else if fd.Parameters != nil {
				data, err := json.Marshal(fd.Parameters)
				if err == nil {
					total += len(data) / 4
				}
			}
		}
	}
	return total
}

// computeBuffer returns the token buffer for a given context window:
// fixed 20k for windows >=200k, 20% for smaller ones.
func computeBuffer(contextWindow int) int {
	if contextWindow >= largeContextWindowThreshold {
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

// injectSummary replaces events that were already summarized with the
// summary content block. contentsAtCompaction is the number of Content
// entries in req.Contents when the summary was produced. Events after
// that watermark are kept as-is (they are new since the last compaction).
// If contentsAtCompaction is 0 or exceeds the current length, the summary
// is simply prepended (first compaction or safety fallback).
func injectSummary(req *model.LLMRequest, summary string, contentsAtCompaction int) {
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

	if contentsAtCompaction > 0 && contentsAtCompaction <= len(req.Contents) {
		newContents := req.Contents[contentsAtCompaction:]
		req.Contents = append([]*genai.Content{summaryContent}, newContents...)
	} else {
		req.Contents = append([]*genai.Content{summaryContent}, req.Contents...)
	}
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

// injectContinuation appends a continuation instruction to req.Contents so
// the agent knows to resume work without re-asking the user. If userContent
// is available, the original user request is included for reference.
func injectContinuation(req *model.LLMRequest, userContent *genai.Content) {
	var text string
	if userContent != nil {
		for _, part := range userContent.Parts {
			if part != nil && part.Text != "" {
				text = part.Text
				break
			}
		}
	}

	var msg string
	if text != "" {
		msg = fmt.Sprintf(
			"[System: The conversation was compacted because it exceeded the context window. "+
				"The summary above contains all prior context. The user's current request is: `%s`. "+
				"Continue working on this request without asking the user to repeat anything.]", text)
	} else {
		msg = "[System: The conversation was compacted because it exceeded the context window. " +
			"The summary above contains all prior context. " +
			"Continue working without asking the user to repeat anything.]"
	}

	req.Contents = append(req.Contents, &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: msg}},
	})
}
