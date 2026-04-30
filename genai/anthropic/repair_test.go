// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// repairMessageHistory enforces Anthropic's invariant that every tool_use must
// be immediately followed by a user message containing a tool_result with the
// matching ID. Conversations replayed from session storage often violate this
// (e.g. when a tool call was made but the agent crashed before recording the
// result), and Anthropic rejects the entire request with a 400 if we don't
// scrub the orphans before sending. The function should:
//   - never alter messages that don't carry tool_use
//   - drop tool_use blocks whose IDs aren't present in the next user message
//   - drop the whole assistant message if scrubbing leaves it empty
//   - keep the message when at least one tool_use survives or other content
//     (e.g. text) was present alongside the orphans
func TestRepairMessageHistory(t *testing.T) {
	textBlock := func(s string) anthropic.ContentBlockParamUnion {
		return anthropic.NewTextBlock(s)
	}
	toolUseBlock := func(id string) anthropic.ContentBlockParamUnion {
		return anthropic.ContentBlockParamUnion{
			OfToolUse: &anthropic.ToolUseBlockParam{ID: id},
		}
	}
	toolResultBlock := func(id string) anthropic.ContentBlockParamUnion {
		return anthropic.ContentBlockParamUnion{
			OfToolResult: &anthropic.ToolResultBlockParam{ToolUseID: id},
		}
	}

	cases := []struct {
		name           string
		messages       []anthropic.MessageParam
		wantLen        int
		wantSurviving  []string // tool_use IDs that must remain in the first message
		wantNotPresent []string // tool_use IDs that must NOT remain anywhere
	}{
		{
			name:     "empty input is returned untouched",
			messages: []anthropic.MessageParam{},
			wantLen:  0,
		},
		{
			name: "messages without tool_use are passed through verbatim",
			messages: []anthropic.MessageParam{
				{Role: anthropic.MessageParamRoleUser, Content: []anthropic.ContentBlockParamUnion{textBlock("hi")}},
				{Role: anthropic.MessageParamRoleAssistant, Content: []anthropic.ContentBlockParamUnion{textBlock("hello")}},
			},
			wantLen: 2,
		},
		{
			name: "tool_use without any following message is dropped entirely",
			messages: []anthropic.MessageParam{
				{
					Role:    anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{toolUseBlock("orphan_1")},
				},
			},
			wantLen:        0,
			wantNotPresent: []string{"orphan_1"},
		},
		{
			name: "tool_use with matching tool_result is preserved",
			messages: []anthropic.MessageParam{
				{
					Role:    anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{toolUseBlock("matched")},
				},
				{
					Role:    anthropic.MessageParamRoleUser,
					Content: []anthropic.ContentBlockParamUnion{toolResultBlock("matched")},
				},
			},
			wantLen:       2,
			wantSurviving: []string{"matched"},
		},
		{
			name: "mixed assistant message keeps text plus matched tool_use, drops orphan",
			messages: []anthropic.MessageParam{
				{
					Role: anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{
						toolUseBlock("matched"),
						toolUseBlock("orphan"),
						textBlock("ran the tool"),
					},
				},
				{
					Role:    anthropic.MessageParamRoleUser,
					Content: []anthropic.ContentBlockParamUnion{toolResultBlock("matched")},
				},
			},
			wantLen:        2,
			wantSurviving:  []string{"matched"},
			wantNotPresent: []string{"orphan"},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := repairMessageHistory(c.messages)

			if len(got) != c.wantLen {
				t.Fatalf("len = %d, want %d", len(got), c.wantLen)
			}

			if c.wantLen == 0 {
				return
			}

			survivingIDs := extractToolUseIDs(got[0])
			survivingSet := toSet(survivingIDs)

			for _, id := range c.wantSurviving {
				if !survivingSet[id] {
					t.Errorf("expected tool_use %q to survive, got %v", id, survivingIDs)
				}
			}
			for _, id := range c.wantNotPresent {
				if survivingSet[id] {
					t.Errorf("expected tool_use %q to be removed, got %v", id, survivingIDs)
				}
			}
		})
	}
}

func toSet(ss []string) map[string]bool {
	out := make(map[string]bool, len(ss))
	for _, s := range ss {
		out[s] = true
	}
	return out
}
