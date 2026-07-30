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

// Package bedrock implements a model.LLM adapter backed by Amazon Bedrock's
// Converse API. Unlike a provider-specific SDK transport, Converse is AWS's
// unified inference API: the same adapter works against any Bedrock model
// that supports it (Anthropic Claude, Meta Llama, Amazon Nova, Mistral,
// Cohere, AI21, and others), and it has first-class support for Bedrock
// Guardrails without the request-body gymnastics required by the raw
// InvokeModel API (see guardrail.go).
package bedrock

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/v2/model"
)

var _ model.LLM = &Model{}

var (
	// ErrNoOutputInResponse is returned when Bedrock's Converse response has
	// no message in its Output field, which should not happen for a
	// successful, non-streaming call.
	ErrNoOutputInResponse = errors.New("no output message in Bedrock Converse response")
)

// Model implements model.LLM using Amazon Bedrock's Converse API.
type Model struct {
	client          *bedrockruntime.Client
	modelID         string
	maxOutputTokens int
	guardrail       *GuardrailConfig

	// additionalModelRequestFields passes provider-specific fields straight
	// through to Converse's AdditionalModelRequestFields (e.g. Anthropic's
	// "thinking" block, or a Nova-specific sampling knob). The adapter never
	// inspects or validates these; an unknown field is rejected by Bedrock,
	// not by this code.
	additionalModelRequestFields map[string]any
}

// Config holds configuration for creating a new Model.
type Config struct {
	// ModelID is the Bedrock model, inference-profile, or provisioned
	// throughput identifier to invoke, e.g.
	// "anthropic.claude-sonnet-4-5-20250929-v1:0",
	// "us.meta.llama3-3-70b-instruct-v1:0", or "amazon.nova-pro-v1:0".
	ModelID string

	// AWSConfig lets the caller supply a fully-built aws.Config (custom
	// credential providers, assumed roles, custom retryers, a non-default
	// HTTP client, etc.). When set, Region and Profile below are ignored and
	// the adapter uses this config as-is.
	AWSConfig *aws.Config

	// Region overrides the AWS region used to resolve the default config
	// chain. Ignored when AWSConfig is set. Falls back to the standard
	// AWS_REGION / shared config resolution when empty.
	Region string

	// Profile selects a named profile from the shared AWS config/credentials
	// files. Ignored when AWSConfig is set.
	Profile string

	// MaxOutputTokens caps how many tokens the model may generate. This is
	// an output-only limit; it does not affect the input or context window.
	// When zero, Converse falls back to the model's own default.
	MaxOutputTokens int

	// Guardrail, when set, attaches a Bedrock Guardrail to every request
	// made by this Model. See GuardrailConfig.
	Guardrail *GuardrailConfig

	// AdditionalModelRequestFields passes provider-specific extra fields
	// straight through to every Converse call's
	// AdditionalModelRequestFields. Use this for knobs Converse's common
	// InferenceConfiguration doesn't expose (e.g. Anthropic extended
	// thinking via {"thinking": {"type": "enabled", "budget_tokens": N}}).
	AdditionalModelRequestFields map[string]any
}

// New creates a Bedrock Converse API client from config. It resolves AWS
// credentials and region using the standard AWS SDK default chain (env vars,
// shared config/credentials files, IAM role, SSO, etc.) unless cfg.AWSConfig
// is provided.
func New(ctx context.Context, cfg Config) (*Model, error) {
	if cfg.ModelID == "" {
		return nil, errors.New("bedrock: ModelID is required")
	}

	awsCfg := cfg.AWSConfig
	if awsCfg == nil {
		loadOpts := []func(*config.LoadOptions) error{}
		if cfg.Region != "" {
			loadOpts = append(loadOpts, config.WithRegion(cfg.Region))
		}
		if cfg.Profile != "" {
			loadOpts = append(loadOpts, config.WithSharedConfigProfile(cfg.Profile))
		}

		loaded, err := config.LoadDefaultConfig(ctx, loadOpts...)
		if err != nil {
			return nil, fmt.Errorf("bedrock: loading AWS config: %w", err)
		}
		awsCfg = &loaded
	}

	return &Model{
		client:                       bedrockruntime.NewFromConfig(*awsCfg),
		modelID:                      cfg.ModelID,
		maxOutputTokens:              cfg.MaxOutputTokens,
		guardrail:                    cfg.Guardrail,
		additionalModelRequestFields: cfg.AdditionalModelRequestFields,
	}, nil
}

// Name returns the Bedrock model identifier this adapter was configured with.
func (m *Model) Name() string {
	return m.modelID
}

// GenerateContent sends the request to Bedrock's Converse API and returns
// responses (streaming or single).
func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}
	return m.generate(ctx, req)
}

// generate sends a single Converse request and yields one complete response.
func (m *Model) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		input, err := m.buildConverseInput(req)
		if err != nil {
			yield(nil, err)
			return
		}

		out, err := m.client.Converse(ctx, input)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp, err := m.convertOutput(out)
		if err != nil {
			yield(nil, err)
			return
		}

		yield(llmResp, nil)
	}
}

// outputMessage unwraps the typed ConverseOutput union, returning the
// model's response message.
func outputMessage(out types.ConverseOutput) (*types.Message, error) {
	switch v := out.(type) {
	case *types.ConverseOutputMemberMessage:
		return &v.Value, nil
	default:
		return nil, ErrNoOutputInResponse
	}
}
