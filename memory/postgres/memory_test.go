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

package postgres

import (
	"context"
	"iter"
	"testing"
	"time"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

const testConnString = "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable"

func setupTestDB(t *testing.T) *PostgresMemoryService {
	ctx := context.Background()
	svc, err := NewPostgresMemoryService(ctx, PostgresMemoryServiceConfig{
		ConnString: testConnString,
	})
	if err != nil {
		t.Fatalf("Failed to create memory service: %v", err)
	}

	// Clean up test data
	_, err = svc.DB().ExecContext(ctx, "DELETE FROM memory_entries WHERE app_name LIKE 'test_%'")
	if err != nil {
		t.Fatalf("Failed to clean up test data: %v", err)
	}

	return svc
}

// mockSession implements session.Session for testing
type mockSession struct {
	id      string
	appName string
	userID  string
	events  *mockEvents
}

func (s *mockSession) ID() string                { return s.id }
func (s *mockSession) AppName() string           { return s.appName }
func (s *mockSession) UserID() string            { return s.userID }
func (s *mockSession) State() session.State      { return nil }
func (s *mockSession) Events() session.Events    { return s.events }
func (s *mockSession) LastUpdateTime() time.Time { return time.Now() }

type mockEvents struct {
	events []*session.Event
}

func (e *mockEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, evt := range e.events {
			if !yield(evt) {
				return
			}
		}
	}
}

func (e *mockEvents) Len() int {
	return len(e.events)
}

func (e *mockEvents) At(i int) *session.Event {
	if i < 0 || i >= len(e.events) {
		return nil
	}
	return e.events[i]
}

func createTestSession(id, appName, userID string, messages []struct{ author, text string }) *mockSession {
	var events []*session.Event
	for i, msg := range messages {
		events = append(events, &session.Event{
			ID:        id + "-" + string(rune('a'+i)),
			Author:    msg.author,
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{genai.NewPartFromText(msg.text)},
					Role:  msg.author,
				},
			},
		})
	}
	return &mockSession{
		id:      id,
		appName: appName,
		userID:  userID,
		events:  &mockEvents{events: events},
	}
}

func TestAddSession(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-1", "test_app", "user-1", []struct{ author, text string }{
		{"user", "What is the capital of France?"},
		{"assistant", "The capital of France is Paris."},
	})

	err := svc.AddSession(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Verify data was inserted
	var count int
	err = svc.DB().QueryRowContext(ctx, "SELECT COUNT(*) FROM memory_entries WHERE app_name = 'test_app'").Scan(&count)
	if err != nil {
		t.Fatalf("Failed to count entries: %v", err)
	}
	if count != 2 {
		t.Errorf("Expected 2 entries, got %d", count)
	}

	t.Logf("✓ AddSession: inserted %d entries", count)
}

func TestAddSessionDuplicates(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-dup", "test_app", "user-1", []struct{ author, text string }{
		{"user", "Hello world"},
	})

	// Add same session twice
	err := svc.AddSession(ctx, sess)
	if err != nil {
		t.Fatalf("First AddSession failed: %v", err)
	}

	err = svc.AddSession(ctx, sess)
	if err != nil {
		t.Fatalf("Second AddSession failed: %v", err)
	}

	// Should still have only 1 entry (upsert)
	var count int
	err = svc.DB().QueryRowContext(ctx, "SELECT COUNT(*) FROM memory_entries WHERE session_id = 'sess-dup'").Scan(&count)
	if err != nil {
		t.Fatalf("Failed to count entries: %v", err)
	}
	if count != 1 {
		t.Errorf("Expected 1 entry (no duplicates), got %d", count)
	}

	t.Logf("✓ AddSession duplicates: correctly handled, %d entry", count)
}

func TestAddSessionEmptyEvents(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-empty", "test_app", "user-1", nil)

	err := svc.AddSession(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession with empty events failed: %v", err)
	}

	var count int
	err = svc.DB().QueryRowContext(ctx, "SELECT COUNT(*) FROM memory_entries WHERE session_id = 'sess-empty'").Scan(&count)
	if err != nil {
		t.Fatalf("Failed to count entries: %v", err)
	}
	if count != 0 {
		t.Errorf("Expected 0 entries for empty session, got %d", count)
	}

	t.Logf("✓ AddSession empty: correctly handled, %d entries", count)
}

func TestSearchByText(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	// Add test data
	sess := createTestSession("sess-search", "test_app", "user-1", []struct{ author, text string }{
		{"user", "Tell me about Kubernetes and container orchestration"},
		{"assistant", "Kubernetes is an open-source container orchestration platform"},
		{"user", "What about Docker?"},
		{"assistant", "Docker is a containerization platform for packaging applications"},
	})

	err := svc.AddSession(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Search for "Kubernetes"
	resp, err := svc.Search(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-1",
		Query:   "Kubernetes",
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(resp.Memories) == 0 {
		t.Error("Expected to find memories matching 'Kubernetes'")
	}

	foundKubernetes := false
	for _, mem := range resp.Memories {
		if mem.Content != nil && len(mem.Content.Parts) > 0 {
			text := mem.Content.Parts[0].Text
			if contains(text, "Kubernetes") {
				foundKubernetes = true
			}
		}
	}
	if !foundKubernetes {
		t.Error("Search results should contain 'Kubernetes'")
	}

	t.Logf("✓ SearchByText: found %d memories for 'Kubernetes'", len(resp.Memories))
}

func TestSearchByTextNoResults(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	resp, err := svc.Search(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-nonexistent",
		Query:   "something that does not exist xyz123",
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(resp.Memories) != 0 {
		t.Errorf("Expected 0 memories for non-matching query, got %d", len(resp.Memories))
	}

	t.Logf("✓ SearchByText no results: correctly returned %d memories", len(resp.Memories))
}

func TestSearchRecent(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	// Add test data
	sess := createTestSession("sess-recent", "test_app", "user-recent", []struct{ author, text string }{
		{"user", "First message"},
		{"assistant", "First response"},
		{"user", "Second message"},
		{"assistant", "Second response"},
	})

	err := svc.AddSession(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Search with empty query should return recent entries
	resp, err := svc.Search(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-recent",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	if len(resp.Memories) == 0 {
		t.Error("Expected to find recent memories with empty query")
	}

	t.Logf("✓ SearchRecent: found %d recent memories", len(resp.Memories))
}

func TestSearchIsolationByUser(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	// Add data for user-a
	sessA := createTestSession("sess-a", "test_app", "user-a", []struct{ author, text string }{
		{"user", "User A secret information"},
	})
	err := svc.AddSession(ctx, sessA)
	if err != nil {
		t.Fatalf("AddSession for user-a failed: %v", err)
	}

	// Add data for user-b
	sessB := createTestSession("sess-b", "test_app", "user-b", []struct{ author, text string }{
		{"user", "User B different information"},
	})
	err = svc.AddSession(ctx, sessB)
	if err != nil {
		t.Fatalf("AddSession for user-b failed: %v", err)
	}

	// Search as user-a should not find user-b's data
	resp, err := svc.Search(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-a",
		Query:   "information",
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	for _, mem := range resp.Memories {
		if mem.Content != nil && len(mem.Content.Parts) > 0 {
			text := mem.Content.Parts[0].Text
			if contains(text, "User B") {
				t.Error("User A should not see User B's memories")
			}
		}
	}

	t.Logf("✓ SearchIsolationByUser: user isolation works correctly")
}

func TestSearchIsolationByApp(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	// Add data for app-1
	sess1 := createTestSession("sess-app1", "test_app_1", "user-1", []struct{ author, text string }{
		{"user", "App 1 secret data"},
	})
	err := svc.AddSession(ctx, sess1)
	if err != nil {
		t.Fatalf("AddSession for app-1 failed: %v", err)
	}

	// Add data for app-2
	sess2 := createTestSession("sess-app2", "test_app_2", "user-1", []struct{ author, text string }{
		{"user", "App 2 different data"},
	})
	err = svc.AddSession(ctx, sess2)
	if err != nil {
		t.Fatalf("AddSession for app-2 failed: %v", err)
	}

	// Search in app-1 should not find app-2's data
	resp, err := svc.Search(ctx, &memory.SearchRequest{
		AppName: "test_app_1",
		UserID:  "user-1",
		Query:   "data",
	})
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	for _, mem := range resp.Memories {
		if mem.Content != nil && len(mem.Content.Parts) > 0 {
			text := mem.Content.Parts[0].Text
			if contains(text, "App 2") {
				t.Error("App 1 should not see App 2's memories")
			}
		}
	}

	t.Logf("✓ SearchIsolationByApp: app isolation works correctly")
}

func TestClose(t *testing.T) {
	svc := setupTestDB(t)

	err := svc.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Verify connection is closed
	err = svc.DB().Ping()
	if err == nil {
		t.Error("Expected error after Close, connection should be closed")
	}

	t.Logf("✓ Close: connection closed correctly")
}

func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
