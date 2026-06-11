// Copyright 2026 Achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

package anthropic

import (
	"github.com/anthropics/anthropic-sdk-go"
)

// caching.go implements Anthropic prompt caching.
//
// Anthropic's cache is OPT-IN and prefix-based: a `cache_control:
// {"type":"ephemeral"}` marker on a content block tells the API
// "everything from the start of the request up to and including this
// block may be cached". A follow-up request whose prefix matches
// byte-for-byte reads those tokens from cache at ~10% of the normal
// input price (writing the cache the first time costs ~25% extra,
// amortised from the second request on). Without markers nothing is
// cached and every request bills its full input. Prefixes shorter
// than the model's minimum cacheable length (1024-4096 tokens) are
// silently not cached — the marker is legal but inert, so it is
// always safe to set.
//
// Anthropic caches in the order tools → system → messages and allows
// at most 4 breakpoints per request. We spend 3, one per section:
//
//  1. The last tool definition. Tool schemas are construction-time
//     constants for an agent, so this prefix is stable across every
//     request of a session (and across sessions of the same agent).
//
//  2. The last system block. The system prompt is equally static.
//     Separate from (1) so editing the prompt — e.g. a hot reload —
//     invalidates only the system+messages cache, not the tools'.
//
//  3. The last cacheable block of the last message. This is the
//     incremental marker: turn N's full conversation becomes turn
//     N+1's cached prefix, which is where the bulk of the savings
//     come from in agentic loops (each tool round-trip re-sends the
//     whole history).
//
// Thinking and redacted-thinking blocks cannot carry cache_control
// (the SDK union exposes no slot for it), so marker (3) walks
// backwards past them to the nearest eligible block.

// applyCacheControl stamps the three breakpoints described above onto
// params, in place. Call it as the LAST step of buildMessageParams so
// every section is final (e.g. after repairMessageHistory, which can
// reorder/merge message blocks).
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
