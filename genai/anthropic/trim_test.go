// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

func assistantText(text string) anthropic.MessageParam {
	return anthropic.MessageParam{
		Role:    anthropic.MessageParamRoleAssistant,
		Content: []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock(text)},
	}
}

func TestTrimFinalAssistantWhitespace(t *testing.T) {
	t.Run("trims trailing whitespace on a final assistant text", func(t *testing.T) {
		msgs := trimFinalAssistantWhitespace([]anthropic.MessageParam{assistantText("ok ")})
		if got := msgs[0].Content[0].OfText.Text; got != "ok" {
			t.Errorf("text = %q, want %q", got, "ok")
		}
	})

	t.Run("leaves a final user message untouched", func(t *testing.T) {
		user := anthropic.MessageParam{
			Role:    anthropic.MessageParamRoleUser,
			Content: []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock("hi ")},
		}
		msgs := trimFinalAssistantWhitespace([]anthropic.MessageParam{user})
		if got := msgs[0].Content[0].OfText.Text; got != "hi " {
			t.Errorf("user text = %q, want it untouched %q", got, "hi ")
		}
	})

	t.Run("only the final message is trimmed", func(t *testing.T) {
		msgs := trimFinalAssistantWhitespace([]anthropic.MessageParam{
			assistantText("first "),
			anthropic.MessageParam{Role: anthropic.MessageParamRoleUser, Content: []anthropic.ContentBlockParamUnion{anthropic.NewTextBlock("q")}},
		})
		if got := msgs[0].Content[0].OfText.Text; got != "first " {
			t.Errorf("non-final assistant text = %q, want it untouched", got)
		}
	})

	t.Run("a block left empty after trimming is dropped", func(t *testing.T) {
		msgs := trimFinalAssistantWhitespace([]anthropic.MessageParam{assistantText("   ")})
		if len(msgs[0].Content) != 0 {
			t.Errorf("expected the whitespace-only block to be dropped, got %#v", msgs[0].Content)
		}
	})
}
