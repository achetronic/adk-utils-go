// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"github.com/anthropics/anthropic-sdk-go"
)

// caching.go stamps Anthropic prompt-cache markers on a request.
//
// The cache is opt-in and prefix-based: a cache_control marker on a block
// makes the API cache everything up to and including that block, and a later
// request with a byte-identical prefix reads it back at ~10% of the input
// price. Setting a marker is always safe: prefixes shorter than the model's
// minimum cacheable length (1024-4096 tokens) are silently not cached.
//
// Anthropic caches in the order tools -> system -> messages and allows at
// most 4 breakpoints per request. We set 3:
//
//  1. The last tool definition. Tool schemas are static per agent, so this
//     prefix survives every request of a session.
//
//  2. The last system block. Static too, and separate from (1) so editing
//     the prompt (e.g. a hot reload) does not invalidate the tools cache.
//
//  3. The last cacheable block of the last message. Turn N's history becomes
//     turn N+1's cached prefix; this is where the savings come from in
//     agentic loops, which re-send the whole history on every tool
//     round-trip. Thinking and redacted-thinking blocks cannot carry a
//     marker, so this one walks backwards past them to the nearest eligible
//     block.
//
// applyCacheControl stamps the three breakpoints onto params, in place. Call
// it as the LAST step of buildMessageParams so every section is final (e.g.
// after repairMessageHistory, which can reorder/merge message blocks).
func applyCacheControl(params *anthropic.MessageNewParams) {
	marker := anthropic.NewCacheControlEphemeralParam()

	// (1) Tools: mark the last definition so the whole tool array is
	// covered by one breakpoint.
	if n := len(params.Tools); n > 0 {
		if cc := params.Tools[n-1].GetCacheControl(); cc != nil {
			*cc = marker
		}
	}

	// (2) System prompt: mark the last block.
	if n := len(params.System); n > 0 {
		params.System[n-1].CacheControl = marker
	}

	// (3) Conversation: walk messages from the tail looking for the
	// last block that supports cache_control. GetCacheControl returns
	// nil for thinking/redacted-thinking blocks, which skips them.
	for i := len(params.Messages) - 1; i >= 0; i-- {
		if markLastCacheableBlock(params.Messages[i].Content, marker) {
			return
		}
	}
}

// markLastCacheableBlock sets the marker on the last block of the
// slice that can carry cache_control. Reports whether a block was
// marked, so the caller knows to stop scanning earlier messages.
func markLastCacheableBlock(blocks []anthropic.ContentBlockParamUnion, marker anthropic.CacheControlEphemeralParam) bool {
	for i := len(blocks) - 1; i >= 0; i-- {
		if cc := blocks[i].GetCacheControl(); cc != nil {
			*cc = marker
			return true
		}
	}
	return false
}
