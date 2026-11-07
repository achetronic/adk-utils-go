// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package common

import "time"

// Options groups the non-HTTP request options forwarded by the LLM adapters.
//
// Both SDKs retry twice by default, so one failed request becomes three
// attempts. Fields are pointers so a caller overrides one knob and leaves the
// rest at their SDK default; the zero value means "change nothing".
type Options struct {
	// MaxRetries is the maximum number of attempts after the first.
	// nil leaves the SDK default (2). A pointer to 0 disables retries; a
	// negative value clamps to 0.
	MaxRetries *int

	// MaxRetryDelay caps the pause between attempts, over backoff and Retry-After.
	// nil leaves the SDK default; non-positive values are ignored. The Anthropic
	// SDK has no equivalent knob, so it ignores this field.
	MaxRetryDelay *time.Duration
}

// ResolveMaxRetries returns the configured budget and whether the caller set
// one. Negative values clamp to 0 because the SDK panics on them.
func (o Options) ResolveMaxRetries() (retries int, set bool) {
	if o.MaxRetries == nil {
		return 0, false
	}
	if *o.MaxRetries < 0 {
		return 0, true
	}
	return *o.MaxRetries, true
}

// ResolveMaxRetryDelay returns the configured delay cap and whether the caller
// set one. Nil and non-positive values report unset because the SDK panics on
// non-positive input.
func (o Options) ResolveMaxRetryDelay() (delay time.Duration, set bool) {
	if o.MaxRetryDelay == nil || *o.MaxRetryDelay <= 0 {
		return 0, false
	}
	return *o.MaxRetryDelay, true
}
