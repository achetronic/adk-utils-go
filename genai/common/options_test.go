// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package common

import (
	"testing"
	"time"
)

func TestOptions_ResolveMaxRetries(t *testing.T) {
	zero := 0
	negative := -3
	positive := 5

	cases := []struct {
		name    string
		field   *int
		wantVal int
		wantSet bool
	}{
		{name: "nil leaves the SDK default", field: nil, wantVal: 0, wantSet: false},
		{name: "zero disables retries", field: &zero, wantVal: 0, wantSet: true},
		{name: "negative clamps to zero", field: &negative, wantVal: 0, wantSet: true},
		{name: "positive is passed through", field: &positive, wantVal: 5, wantSet: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, set := (Options{MaxRetries: tc.field}).ResolveMaxRetries()
			if got != tc.wantVal || set != tc.wantSet {
				t.Errorf("ResolveMaxRetries() = (%d, %t), want (%d, %t)", got, set, tc.wantVal, tc.wantSet)
			}
		})
	}
}

func TestOptions_ResolveMaxRetryDelay(t *testing.T) {
	zero := 0 * time.Millisecond
	negative := -1 * time.Millisecond
	positive := 250 * time.Millisecond

	cases := []struct {
		name    string
		field   *time.Duration
		wantVal time.Duration
		wantSet bool
	}{
		{name: "nil leaves the SDK default", field: nil, wantVal: 0, wantSet: false},
		{name: "zero is treated as unset", field: &zero, wantVal: 0, wantSet: false},
		{name: "negative is treated as unset", field: &negative, wantVal: 0, wantSet: false},
		{name: "positive is passed through", field: &positive, wantVal: 250 * time.Millisecond, wantSet: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, set := (Options{MaxRetryDelay: tc.field}).ResolveMaxRetryDelay()
			if got != tc.wantVal || set != tc.wantSet {
				t.Errorf("ResolveMaxRetryDelay() = (%s, %t), want (%s, %t)", got, set, tc.wantVal, tc.wantSet)
			}
		})
	}
}
