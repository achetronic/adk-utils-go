// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Package common holds helpers shared by the genai LLM adapters so wire-format
// rules are implemented once.
package common

import "encoding/json"

var emptyJSONObject = json.RawMessage(`{}`)

// MarshalToolPayload encodes a tool-call args or tool-response map for the wire.
// nil/empty become "{}", never "null": strict OpenAI-compatible parsers (Qwen on
// vLLM/llama.cpp) reject "null" where they expect an object. An already-encoded
// json.RawMessage passes through. The input is never mutated.
func MarshalToolPayload(payload any) (json.RawMessage, error) {
	if payload == nil {
		return emptyJSONObject, nil
	}

	if raw, ok := payload.(json.RawMessage); ok {
		if len(raw) == 0 {
			return emptyJSONObject, nil
		}
		return raw, nil
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 || string(data) == "null" {
		return emptyJSONObject, nil
	}
	return data, nil
}
