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

	"github.com/achetronic/adk-utils-go/memory/memorytypes"
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

	err := svc.AddSessionToMemory(ctx, sess)
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
	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("First AddSession failed: %v", err)
	}

	err = svc.AddSessionToMemory(ctx, sess)
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

	err := svc.AddSessionToMemory(ctx, sess)
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

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Search for "Kubernetes"
	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
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

	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
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

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	// Search with empty query should return recent entries
	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
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
	err := svc.AddSessionToMemory(ctx, sessA)
	if err != nil {
		t.Fatalf("AddSession for user-a failed: %v", err)
	}

	// Add data for user-b
	sessB := createTestSession("sess-b", "test_app", "user-b", []struct{ author, text string }{
		{"user", "User B different information"},
	})
	err = svc.AddSessionToMemory(ctx, sessB)
	if err != nil {
		t.Fatalf("AddSession for user-b failed: %v", err)
	}

	// Search as user-a should not find user-b's data
	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
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
	err := svc.AddSessionToMemory(ctx, sess1)
	if err != nil {
		t.Fatalf("AddSession for app-1 failed: %v", err)
	}

	// Add data for app-2
	sess2 := createTestSession("sess-app2", "test_app_2", "user-1", []struct{ author, text string }{
		{"user", "App 2 different data"},
	})
	err = svc.AddSessionToMemory(ctx, sess2)
	if err != nil {
		t.Fatalf("AddSession for app-2 failed: %v", err)
	}

	// Search in app-1 should not find app-2's data
	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
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

func TestSearchWithID(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-withid", "test_app", "user-withid", []struct{ author, text string }{
		{"user", "I love programming in Go"},
		{"assistant", "Go is a great language for building scalable systems"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-withid",
		Query:   "Go programming",
	})
	if err != nil {
		t.Fatalf("SearchWithID failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected to find memories with SearchWithID")
	}

	for _, entry := range results {
		if entry.ID == 0 {
			t.Error("Expected non-zero ID in SearchWithID results")
		}
		if entry.Content == nil || len(entry.Content.Parts) == 0 {
			t.Error("Expected content in SearchWithID results")
		}
	}

	t.Logf("✓ SearchWithID: found %d entries with IDs", len(results))
}

func TestSearchWithIDRecent(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-withid-recent", "test_app", "user-withid-recent", []struct{ author, text string }{
		{"user", "Remember this fact"},
		{"assistant", "I will remember it"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-withid-recent",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID with empty query failed: %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected recent entries with empty query")
	}

	for _, entry := range results {
		if entry.ID == 0 {
			t.Error("Expected non-zero ID in recent results")
		}
	}

	t.Logf("✓ SearchWithIDRecent: found %d recent entries with IDs", len(results))
}

func TestUpdateMemory(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-update", "test_app", "user-update", []struct{ author, text string }{
		{"assistant", "The user likes cats"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-update",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Expected at least one entry to update")
	}

	entryID := results[0].ID

	err = svc.UpdateMemory(ctx, "test_app", "user-update", entryID, "The user likes dogs now")
	if err != nil {
		t.Fatalf("UpdateMemory failed: %v", err)
	}

	var contentText string
	err = svc.DB().QueryRowContext(ctx, "SELECT content_text FROM memory_entries WHERE id = $1", entryID).Scan(&contentText)
	if err != nil {
		t.Fatalf("Failed to query updated entry: %v", err)
	}
	if contentText != "The user likes dogs now" {
		t.Errorf("Expected updated content, got: %s", contentText)
	}

	t.Logf("✓ UpdateMemory: entry %d updated successfully", entryID)
}

func TestUpdateMemoryNotFound(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	err := svc.UpdateMemory(ctx, "test_app", "user-nonexistent", 999999, "new content")
	if err == nil {
		t.Fatal("Expected error when updating non-existent entry")
	}
	if !contains(err.Error(), "not found") {
		t.Errorf("Expected 'not found' error, got: %v", err)
	}

	t.Logf("✓ UpdateMemoryNotFound: correctly returned error")
}

func TestUpdateMemoryEmptyContent(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	err := svc.UpdateMemory(ctx, "test_app", "user-1", 1, "")
	if err == nil {
		t.Fatal("Expected error when updating with empty content")
	}

	t.Logf("✓ UpdateMemoryEmptyContent: correctly returned error")
}

func TestUpdateMemoryIsolation(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-update-iso", "test_app", "user-update-iso", []struct{ author, text string }{
		{"assistant", "Private data for user-update-iso"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-update-iso",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Expected at least one entry")
	}

	entryID := results[0].ID

	err = svc.UpdateMemory(ctx, "test_app", "attacker-user", entryID, "hacked")
	if err == nil {
		t.Fatal("Expected error when updating another user's entry")
	}

	err = svc.UpdateMemory(ctx, "other_app", "user-update-iso", entryID, "hacked")
	if err == nil {
		t.Fatal("Expected error when updating entry from different app")
	}

	t.Logf("✓ UpdateMemoryIsolation: user/app scoping works correctly")
}

func TestDeleteMemory(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-delete", "test_app", "user-delete", []struct{ author, text string }{
		{"assistant", "Temporary information to delete"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-delete",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Expected at least one entry to delete")
	}

	entryID := results[0].ID

	err = svc.DeleteMemory(ctx, "test_app", "user-delete", entryID)
	if err != nil {
		t.Fatalf("DeleteMemory failed: %v", err)
	}

	var count int
	err = svc.DB().QueryRowContext(ctx, "SELECT COUNT(*) FROM memory_entries WHERE id = $1", entryID).Scan(&count)
	if err != nil {
		t.Fatalf("Failed to count entries: %v", err)
	}
	if count != 0 {
		t.Errorf("Expected entry to be deleted, but found %d", count)
	}

	t.Logf("✓ DeleteMemory: entry %d deleted successfully", entryID)
}

func TestDeleteMemoryNotFound(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	err := svc.DeleteMemory(ctx, "test_app", "user-nonexistent", 999999)
	if err == nil {
		t.Fatal("Expected error when deleting non-existent entry")
	}
	if !contains(err.Error(), "not found") {
		t.Errorf("Expected 'not found' error, got: %v", err)
	}

	t.Logf("✓ DeleteMemoryNotFound: correctly returned error")
}

func TestDeleteMemoryIsolation(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-delete-iso", "test_app", "user-delete-iso", []struct{ author, text string }{
		{"assistant", "Private data for user-delete-iso"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-delete-iso",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Expected at least one entry")
	}

	entryID := results[0].ID

	err = svc.DeleteMemory(ctx, "test_app", "attacker-user", entryID)
	if err == nil {
		t.Fatal("Expected error when deleting another user's entry")
	}

	err = svc.DeleteMemory(ctx, "other_app", "user-delete-iso", entryID)
	if err == nil {
		t.Fatal("Expected error when deleting entry from different app")
	}

	var count int
	err = svc.DB().QueryRowContext(ctx, "SELECT COUNT(*) FROM memory_entries WHERE id = $1", entryID).Scan(&count)
	if err != nil {
		t.Fatalf("Failed to count entries: %v", err)
	}
	if count != 1 {
		t.Error("Entry should still exist after failed cross-user/cross-app delete attempts")
	}

	t.Logf("✓ DeleteMemoryIsolation: user/app scoping works correctly")
}

func TestDeleteThenSearch(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-del-search", "test_app", "user-del-search", []struct{ author, text string }{
		{"assistant", "The user favorite color is blue"},
		{"assistant", "The user works at Acme Corp"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-del-search",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID failed: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("Expected 2 entries, got %d", len(results))
	}

	err = svc.DeleteMemory(ctx, "test_app", "user-del-search", results[0].ID)
	if err != nil {
		t.Fatalf("DeleteMemory failed: %v", err)
	}

	remaining, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-del-search",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID after delete failed: %v", err)
	}
	if len(remaining) != 1 {
		t.Errorf("Expected 1 remaining entry, got %d", len(remaining))
	}

	t.Logf("✓ DeleteThenSearch: correctly shows %d remaining entry after deletion", len(remaining))
}

func TestUpdateThenSearch(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()
	ctx := context.Background()

	sess := createTestSession("sess-upd-search", "test_app", "user-upd-search", []struct{ author, text string }{
		{"assistant", "The user prefers dark mode"},
	})

	err := svc.AddSessionToMemory(ctx, sess)
	if err != nil {
		t.Fatalf("AddSession failed: %v", err)
	}

	results, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-upd-search",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Expected at least one entry")
	}

	entryID := results[0].ID

	err = svc.UpdateMemory(ctx, "test_app", "user-upd-search", entryID, "The user prefers light mode")
	if err != nil {
		t.Fatalf("UpdateMemory failed: %v", err)
	}

	updated, err := svc.SearchWithID(ctx, &memory.SearchRequest{
		AppName: "test_app",
		UserID:  "user-upd-search",
		Query:   "",
	})
	if err != nil {
		t.Fatalf("SearchWithID after update failed: %v", err)
	}
	if len(updated) == 0 {
		t.Fatal("Expected to find updated entry")
	}

	foundUpdated := false
	for _, entry := range updated {
		if entry.Content != nil && len(entry.Content.Parts) > 0 {
			if entry.Content.Parts[0].Text == "The user prefers light mode" {
				foundUpdated = true
			}
		}
	}
	if !foundUpdated {
		t.Error("Expected to find updated content in search results")
	}

	t.Logf("✓ UpdateThenSearch: updated content found in search results")
}

func TestExtendedMemoryServiceInterface(t *testing.T) {
	svc := setupTestDB(t)
	defer svc.Close()

	var _ memorytypes.ExtendedMemoryService = svc

	t.Logf("✓ ExtendedMemoryServiceInterface: PostgresMemoryService satisfies the interface")
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
