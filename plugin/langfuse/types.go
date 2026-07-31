// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package langfuse

import (
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// Config holds the credentials and optional metadata needed to export traces
// to a Langfuse instance via OTLP/HTTP.
//
// Host applications should embed or reference this struct directly (e.g. as a
// YAML field) rather than defining their own mirror type. Struct tags for yaml
// and json are provided so it can be unmarshalled from configuration files
// without any adapter code.
type Config struct {
	// PublicKey is the Langfuse project public key used as the Basic Auth
	// username for the OTLP ingestion endpoint.
	PublicKey string `yaml:"publicKey" json:"publicKey"`

	// SecretKey is the Langfuse project secret key used as the Basic Auth
	// password for the OTLP ingestion endpoint.
	SecretKey string `yaml:"secretKey" json:"secretKey"`

	// Host is the base URL of the Langfuse server (e.g.
	// "https://eu.cloud.langfuse.com"). When empty it defaults to the Langfuse
	// Cloud US endpoint.
	Host string `yaml:"host" json:"host"`

	// Environment is an optional deployment environment tag forwarded to every
	// trace (e.g. "production", "staging").
	Environment string `yaml:"environment" json:"environment"`

	// Release is an optional application version tag forwarded to every trace.
	Release string `yaml:"release" json:"release"`

	// ServiceName is the OpenTelemetry service.name resource attribute. Host
	// applications should set this to their own name. Defaults to
	// "langfuse-adk" when empty.
	ServiceName string `yaml:"serviceName,omitempty" json:"serviceName,omitempty"`

	// Insecure disables TLS for the OTLP/HTTP exporter. Set to true only when
	// connecting to a self-hosted Langfuse instance that does not serve HTTPS
	// (e.g. plain-HTTP local development). The default (false) keeps TLS
	// enabled, which is required for the public Langfuse Cloud endpoints.
	Insecure bool `yaml:"insecure,omitempty" json:"insecure,omitempty"`

	// TracerProviderOptions are extra options for the trace provider Setup
	// builds, on top of the exporter and resource it wires itself. Use it
	// for settings the default wiring does not expose: a custom ID generator
	// (for example one that derives the trace ID from your own run ID, so a
	// paused agent resuming in another HTTP request stays in one Langfuse
	// trace), a sampler to cap ingestion volume, or span limits.
	//
	// The options run after Setup's own resource and span processor, so an
	// option with replace semantics (sdktrace.WithResource, for example)
	// overrides what Setup wired. The resource Setup wires is merged over
	// resource.Default(), matching what the ADK assembles on its default
	// path.
	//
	// Caveat: with this field set the ADK receives a finished trace provider
	// and skips the extra OTLP trace exporters it would otherwise wire from
	// the OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
	// environment variables. Log exporters are unaffected.
	//
	// Programmatic only: the field cannot be set from YAML/JSON config
	// files. When empty, Setup behaves exactly as before.
	TracerProviderOptions []sdktrace.TracerProviderOption `yaml:"-" json:"-"`
}

// IsEnabled reports whether the minimum required credentials (PublicKey and
// SecretKey) are configured. Callers should gate Langfuse setup behind this
// check so that the plugin is silently disabled when no credentials are
// provided.
func (c *Config) IsEnabled() bool {
	return c != nil && c.PublicKey != "" && c.SecretKey != ""
}
