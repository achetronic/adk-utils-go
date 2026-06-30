// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package common

import (
	"encoding/json"
	"testing"
)

// nil/empty payloads must encode to "{}", real ones pass through, marshal
// errors propagate.
func TestMarshalToolPayload(t *testing.T) {
	cases := []struct {
		name    string
		payload any
		want    string
		wantErr bool
	}{
		{"untyped nil serialises to empty object", nil, "{}", false},
		{"nil typed map serialises to empty object, not null", map[string]any(nil), "{}", false},
		{"empty non-nil map serialises to empty object", map[string]any{}, "{}", false},
		{"populated map passes through", map[string]any{"q": "weather"}, `{"q":"weather"}`, false},
		{"empty raw message falls back to default", json.RawMessage(""), "{}", false},
		{"raw message passes through verbatim", json.RawMessage(`{"a":1}`), `{"a":1}`, false},
		{"struct gets marshalled", struct {
			Foo string `json:"foo"`
		}{Foo: "bar"}, `{"foo":"bar"}`, false},
		{"unmarshalable value propagates the error", make(chan int), "", true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := MarshalToolPayload(c.payload)
			if (err != nil) != c.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, c.wantErr)
			}
			if c.wantErr {
				return
			}
			if string(got) != c.want {
				t.Errorf("MarshalToolPayload(%#v) = %q, want %q", c.payload, got, c.want)
			}
		})
	}
}
