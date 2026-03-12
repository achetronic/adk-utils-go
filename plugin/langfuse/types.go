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

package langfuse

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
}

// IsEnabled reports whether the minimum required credentials (PublicKey and
// SecretKey) are configured. Callers should gate Langfuse setup behind this
// check so that the plugin is silently disabled when no credentials are
// provided.
func (c *Config) IsEnabled() bool {
	return c != nil && c.PublicKey != "" && c.SecretKey != ""
}
