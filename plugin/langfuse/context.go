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

import "context"

// contextKey is an unexported type used as key for context.WithValue to avoid
// collisions with keys defined in other packages.
type contextKey int

const (
	userIDKey        contextKey = iota // Langfuse user identity
	tagsKey                            // free-form trace tags
	traceMetadataKey                   // arbitrary key-value pairs attached to the trace
	environmentKey                     // deployment environment (e.g. "production")
	releaseKey                         // application version tag
	traceNameKey                       // explicit trace name override
)

// WithUserID stores a Langfuse user ID in the context. The spanEnricher reads
// it in beforeAgent and sets the langfuse.user.id span attribute. When absent
// the plugin falls back to the ADK-native ctx.UserID().
//
// Typical caller: an HTTP middleware that extracts the authenticated user from
// the request (e.g. from a JWT).
func WithUserID(ctx context.Context, userID string) context.Context {
	return context.WithValue(ctx, userIDKey, userID)
}

// UserIDFromContext returns the Langfuse user ID previously stored with
// WithUserID, or "" if none was set.
func UserIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(userIDKey).(string); ok {
		return v
	}
	return ""
}

// WithTags stores a set of free-form tags in the context. The spanEnricher
// reads them in beforeAgent and sets the langfuse.trace.tags span attribute.
//
// Tags appear in the Langfuse UI as filterable labels on the trace.
func WithTags(ctx context.Context, tags []string) context.Context {
	return context.WithValue(ctx, tagsKey, tags)
}

// TagsFromContext returns the tags previously stored with WithTags, or nil if
// none were set.
func TagsFromContext(ctx context.Context) []string {
	if v, ok := ctx.Value(tagsKey).([]string); ok {
		return v
	}
	return nil
}

// WithTraceMetadata stores arbitrary key-value metadata in the context. The
// spanEnricher reads the map in beforeAgent and sets one
// langfuse.trace.metadata.<key> span attribute per entry.
//
// Multiple callers can cooperate by reading, copying, and extending the
// existing map — the context itself is immutable so no mutex is needed.
func WithTraceMetadata(ctx context.Context, metadata map[string]string) context.Context {
	return context.WithValue(ctx, traceMetadataKey, metadata)
}

// TraceMetadataFromContext returns the metadata map previously stored with
// WithTraceMetadata, or nil if none was set.
func TraceMetadataFromContext(ctx context.Context) map[string]string {
	if v, ok := ctx.Value(traceMetadataKey).(map[string]string); ok {
		return v
	}
	return nil
}

// WithEnvironment stores the deployment environment name in the context. The
// spanEnricher reads it in beforeAgent and sets the langfuse.environment span
// attribute. This supplements the static Config.Environment value when the
// environment needs to vary per request.
func WithEnvironment(ctx context.Context, environment string) context.Context {
	return context.WithValue(ctx, environmentKey, environment)
}

// EnvironmentFromContext returns the environment previously stored with
// WithEnvironment, or "" if none was set.
func EnvironmentFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(environmentKey).(string); ok {
		return v
	}
	return ""
}

// WithRelease stores a release/version tag in the context. The spanEnricher
// reads it in beforeAgent and sets the langfuse.release span attribute. This
// supplements the static Config.Release value when the version needs to vary
// per request.
func WithRelease(ctx context.Context, release string) context.Context {
	return context.WithValue(ctx, releaseKey, release)
}

// ReleaseFromContext returns the release previously stored with WithRelease,
// or "" if none was set.
func ReleaseFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(releaseKey).(string); ok {
		return v
	}
	return ""
}

// WithTraceName stores an explicit trace name in the context. The spanEnricher
// reads it in beforeAgent and sets the langfuse.trace.name span attribute.
// When absent Langfuse auto-generates a name from the root span.
func WithTraceName(ctx context.Context, name string) context.Context {
	return context.WithValue(ctx, traceNameKey, name)
}

// TraceNameFromContext returns the trace name previously stored with
// WithTraceName, or "" if none was set.
func TraceNameFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(traceNameKey).(string); ok {
		return v
	}
	return ""
}
