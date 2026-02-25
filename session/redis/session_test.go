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

package redis

import (
	"context"
	"fmt"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

const testRedisAddr = "localhost:6379"

func setupTestService(t *testing.T) *RedisSessionService {
	t.Helper()
	svc, err := NewRedisSessionService(RedisSessionServiceConfig{
		Addr: testRedisAddr,
		TTL:  5 * time.Minute,
	})
	if err != nil {
		t.Fatalf("Failed to create Redis session service: %v", err)
	}
	t.Cleanup(func() { svc.Close() })
	return svc
}

func uniquePrefix(t *testing.T) string {
	t.Helper()
	return fmt.Sprintf("test_%d", time.Now().UnixNano())
}

func TestCreateAndGet(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app,
		UserID:  "user-1",
		State:   map[string]any{"counter": float64(1)},
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: resp.Session.ID(),
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	val, err := got.Session.State().Get("counter")
	if err != nil {
		t.Fatalf("State().Get failed: %v", err)
	}
	if val != float64(1) {
		t.Errorf("expected counter=1, got %v", val)
	}
	t.Logf("✓ CreateAndGet: session %s with state counter=%v", got.Session.ID(), val)
}

func TestCreateDuplicate(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	_, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "fixed-id",
	})
	if err != nil {
		t.Fatalf("First create failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "fixed-id",
	})
	if err == nil {
		t.Fatal("Expected error on duplicate create, got nil")
	}
	t.Logf("✓ CreateDuplicate: correctly rejected")
}

func TestCreateAutoID(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app,
		UserID:  "user-1",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
	if resp.Session.ID() == "" {
		t.Fatal("Expected auto-generated session ID, got empty")
	}
	t.Logf("✓ CreateAutoID: generated %s", resp.Session.ID())
}

func TestList(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	for i := 0; i < 3; i++ {
		_, err := svc.Create(ctx, &session.CreateRequest{
			AppName:   app,
			UserID:    "user-1",
			SessionID: fmt.Sprintf("s-%d", i),
		})
		if err != nil {
			t.Fatalf("Create %d failed: %v", i, err)
		}
	}

	resp, err := svc.List(ctx, &session.ListRequest{
		AppName: app,
		UserID:  "user-1",
	})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	if len(resp.Sessions) != 3 {
		t.Errorf("expected 3 sessions, got %d", len(resp.Sessions))
	}
	t.Logf("✓ List: found %d sessions", len(resp.Sessions))
}

func TestDelete(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app,
		UserID:  "user-1",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	err = svc.Delete(ctx, &session.DeleteRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: resp.Session.ID(),
	})
	if err != nil {
		t.Fatalf("Delete failed: %v", err)
	}

	_, err = svc.Get(ctx, &session.GetRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: resp.Session.ID(),
	})
	if err == nil {
		t.Fatal("Expected error after delete, got nil")
	}
	t.Logf("✓ Delete: session removed")
}

func TestAppendEvent(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app,
		UserID:  "user-1",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	evt := &session.Event{
		Author: "user",
		Actions: session.EventActions{
			StateDelta: map[string]any{"step": float64(1)},
		},
	}
	err = svc.AppendEvent(ctx, resp.Session, evt)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: resp.Session.ID(),
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if got.Session.Events().Len() != 1 {
		t.Errorf("expected 1 event, got %d", got.Session.Events().Len())
	}
	val, err := got.Session.State().Get("step")
	if err != nil {
		t.Fatalf("State().Get failed: %v", err)
	}
	if val != float64(1) {
		t.Errorf("expected step=1, got %v", val)
	}
	t.Logf("✓ AppendEvent: event persisted with state delta")
}

func TestAppendEventPartialIgnored(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app,
		UserID:  "user-1",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	err = svc.AppendEvent(ctx, resp.Session, &session.Event{
		LLMResponse: model.LLMResponse{Partial: true},
		Author:      "model",
	})
	if err != nil {
		t.Fatalf("AppendEvent (partial) should return nil, got: %v", err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: resp.Session.ID(),
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if got.Session.Events().Len() != 0 {
		t.Errorf("expected 0 events after partial, got %d", got.Session.Events().Len())
	}
	t.Logf("✓ AppendEventPartialIgnored: partial event correctly skipped")
}

func TestTempStateNotPersisted(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app,
		UserID:  "user-1",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	evt := &session.Event{
		Author: "model",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"keep":          "yes",
				"temp:scratch":  "discard_me",
				"temp:internal": "also_discard",
			},
		},
	}
	err = svc.AppendEvent(ctx, resp.Session, evt)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: resp.Session.ID(),
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	if _, err := got.Session.State().Get("keep"); err != nil {
		t.Error("expected 'keep' to be persisted")
	}
	if _, err := got.Session.State().Get("temp:scratch"); err == nil {
		t.Error("expected 'temp:scratch' to NOT be persisted")
	}
	if _, err := got.Session.State().Get("temp:internal"); err == nil {
		t.Error("expected 'temp:internal' to NOT be persisted")
	}
	t.Logf("✓ TempStateNotPersisted: temp: keys correctly discarded")
}

// --- State Tiers ---

func TestAppStateTierSharedAcrossSessions(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	sess1, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Create sess-1 failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-2",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Create sess-2 failed: %v", err)
	}

	evt := &session.Event{
		Author: "model",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"app:theme": "dark",
			},
		},
	}
	err = svc.AppendEvent(ctx, sess1.Session, evt)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	got1, err := svc.Get(ctx, &session.GetRequest{
		AppName: app, UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Get sess-1 failed: %v", err)
	}
	val, err := got1.Session.State().Get("app:theme")
	if err != nil {
		t.Fatal("sess-1 should see app:theme")
	}
	if val != "dark" {
		t.Errorf("sess-1 app:theme = %v, want dark", val)
	}

	got2, err := svc.Get(ctx, &session.GetRequest{
		AppName: app, UserID: "user-2", SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Get sess-2 failed: %v", err)
	}
	val, err = got2.Session.State().Get("app:theme")
	if err != nil {
		t.Fatal("sess-2 (different user) should also see app:theme")
	}
	if val != "dark" {
		t.Errorf("sess-2 app:theme = %v, want dark", val)
	}
	t.Logf("✓ AppStateTierSharedAcrossSessions: app:theme visible to both users")
}

func TestUserStateTierSharedAcrossSessionsSameUser(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	sess1, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Create sess-1 failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Create sess-2 failed: %v", err)
	}

	evt := &session.Event{
		Author: "model",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"user:lang": "es",
			},
		},
	}
	err = svc.AppendEvent(ctx, sess1.Session, evt)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	got2, err := svc.Get(ctx, &session.GetRequest{
		AppName: app, UserID: "user-1", SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Get sess-2 failed: %v", err)
	}
	val, err := got2.Session.State().Get("user:lang")
	if err != nil {
		t.Fatal("sess-2 (same user) should see user:lang")
	}
	if val != "es" {
		t.Errorf("sess-2 user:lang = %v, want es", val)
	}
	t.Logf("✓ UserStateTierSharedAcrossSessionsSameUser: user:lang visible across sessions")
}

func TestUserStateTierIsolatedBetweenUsers(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	sess1, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Create sess-1 failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-2",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Create sess-2 failed: %v", err)
	}

	evt := &session.Event{
		Author: "model",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"user:lang": "es",
			},
		},
	}
	err = svc.AppendEvent(ctx, sess1.Session, evt)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	got2, err := svc.Get(ctx, &session.GetRequest{
		AppName: app, UserID: "user-2", SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Get sess-2 failed: %v", err)
	}
	_, err = got2.Session.State().Get("user:lang")
	if err == nil {
		t.Fatal("user-2 should NOT see user:lang set by user-1")
	}
	t.Logf("✓ UserStateTierIsolatedBetweenUsers: user:lang correctly scoped")
}

func TestSessionStateTierIsolated(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	sess1, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Create sess-1 failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Create sess-2 failed: %v", err)
	}

	evt := &session.Event{
		Author: "model",
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"counter": float64(42),
			},
		},
	}
	err = svc.AppendEvent(ctx, sess1.Session, evt)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	got2, err := svc.Get(ctx, &session.GetRequest{
		AppName: app, UserID: "user-1", SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Get sess-2 failed: %v", err)
	}
	_, err = got2.Session.State().Get("counter")
	if err == nil {
		t.Fatal("sess-2 should NOT see session-scoped 'counter' from sess-1")
	}
	t.Logf("✓ SessionStateTierIsolated: unprefixed keys stay per-session")
}

func TestCreateWithMixedStateTiers(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	_, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
		State: map[string]any{
			"app:model":      "gemini",
			"user:timezone":  "UTC",
			"local_var":      "private",
			"temp:throwaway": "gone",
		},
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Create sess-2 failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-2",
		SessionID: "sess-3",
	})
	if err != nil {
		t.Fatalf("Create sess-3 failed: %v", err)
	}

	got1, _ := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: "user-1", SessionID: "sess-1"})
	got2, _ := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: "user-1", SessionID: "sess-2"})
	got3, _ := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: "user-2", SessionID: "sess-3"})

	// app:model visible to all
	for _, g := range []struct {
		name string
		sess session.Session
	}{
		{"sess-1", got1.Session},
		{"sess-2", got2.Session},
		{"sess-3", got3.Session},
	} {
		val, err := g.sess.State().Get("app:model")
		if err != nil {
			t.Errorf("%s should see app:model", g.name)
		} else if val != "gemini" {
			t.Errorf("%s app:model = %v, want gemini", g.name, val)
		}
	}

	// user:timezone visible to user-1 sessions only
	val, err := got2.Session.State().Get("user:timezone")
	if err != nil {
		t.Error("sess-2 (user-1) should see user:timezone")
	} else if val != "UTC" {
		t.Errorf("sess-2 user:timezone = %v, want UTC", val)
	}

	_, err = got3.Session.State().Get("user:timezone")
	if err == nil {
		t.Error("sess-3 (user-2) should NOT see user:timezone")
	}

	// local_var only in sess-1
	_, err = got2.Session.State().Get("local_var")
	if err == nil {
		t.Error("sess-2 should NOT see local_var from sess-1")
	}

	// temp: never persisted
	_, err = got1.Session.State().Get("temp:throwaway")
	if err == nil {
		t.Error("temp:throwaway should NOT be persisted")
	}

	t.Logf("✓ CreateWithMixedStateTiers: all tiers correctly routed on Create")
}

func TestStateSetRouting(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	sess1, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Create sess-1 failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-2",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Create sess-2 failed: %v", err)
	}

	// Set via State().Set() instead of StateDelta
	sess1.Session.State().Set("app:feature", "enabled")
	sess1.Session.State().Set("user:pref", "compact")
	sess1.Session.State().Set("local", "only_here")

	got2, err := svc.Get(ctx, &session.GetRequest{
		AppName: app, UserID: "user-2", SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Get sess-2 failed: %v", err)
	}

	val, err := got2.Session.State().Get("app:feature")
	if err != nil {
		t.Fatal("sess-2 should see app:feature set via State().Set()")
	}
	if val != "enabled" {
		t.Errorf("app:feature = %v, want enabled", val)
	}

	_, err = got2.Session.State().Get("user:pref")
	if err == nil {
		t.Fatal("user-2 should NOT see user:pref set by user-1")
	}

	_, err = got2.Session.State().Get("local")
	if err == nil {
		t.Fatal("sess-2 should NOT see session-scoped 'local' from sess-1")
	}

	t.Logf("✓ StateSetRouting: State().Set() correctly routes to tier-specific Redis keys")
}

func TestAppStateOverwrite(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	sess1, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
		State:     map[string]any{"app:version": "v1"},
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	evt := &session.Event{
		Author: "model",
		Actions: session.EventActions{
			StateDelta: map[string]any{"app:version": "v2"},
		},
	}
	err = svc.AppendEvent(ctx, sess1.Session, evt)
	if err != nil {
		t.Fatalf("AppendEvent failed: %v", err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName: app, UserID: "user-1", SessionID: "sess-1",
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	val, _ := got.Session.State().Get("app:version")
	if val != "v2" {
		t.Errorf("app:version = %v, want v2", val)
	}
	t.Logf("✓ AppStateOverwrite: app state correctly updated from v1 to v2")
}

func TestGetNumRecentEvents(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "user-1",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	for i := 0; i < 5; i++ {
		err = svc.AppendEvent(ctx, resp.Session, &session.Event{
			Author: "user",
		})
		if err != nil {
			t.Fatalf("AppendEvent %d failed: %v", i, err)
		}
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:         app,
		UserID:          "user-1",
		SessionID:       resp.Session.ID(),
		NumRecentEvents: 2,
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if got.Session.Events().Len() != 2 {
		t.Errorf("expected 2 recent events, got %d", got.Session.Events().Len())
	}
	t.Logf("✓ GetNumRecentEvents: filtered to 2 events")
}

func TestGetAfterTimestamp(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	resp, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "user-1",
	})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	err = svc.AppendEvent(ctx, resp.Session, &session.Event{Author: "user"})
	if err != nil {
		t.Fatalf("AppendEvent 1 failed: %v", err)
	}

	cutoff := time.Now()
	time.Sleep(10 * time.Millisecond)

	err = svc.AppendEvent(ctx, resp.Session, &session.Event{Author: "user"})
	if err != nil {
		t.Fatalf("AppendEvent 2 failed: %v", err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: resp.Session.ID(),
		After:     cutoff,
	})
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if got.Session.Events().Len() != 1 {
		t.Errorf("expected 1 event after cutoff, got %d", got.Session.Events().Len())
	}
	t.Logf("✓ GetAfterTimestamp: filtered to events after cutoff")
}

func TestListMergesStateTiers(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	_, err := svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-1",
		State:     map[string]any{"app:global": "shared"},
	})
	if err != nil {
		t.Fatalf("Create sess-1 failed: %v", err)
	}

	_, err = svc.Create(ctx, &session.CreateRequest{
		AppName:   app,
		UserID:    "user-1",
		SessionID: "sess-2",
	})
	if err != nil {
		t.Fatalf("Create sess-2 failed: %v", err)
	}

	resp, err := svc.List(ctx, &session.ListRequest{
		AppName: app,
		UserID:  "user-1",
	})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}

	for _, s := range resp.Sessions {
		val, err := s.State().Get("app:global")
		if err != nil {
			t.Errorf("session %s should see app:global", s.ID())
		} else if val != "shared" {
			t.Errorf("session %s app:global = %v, want shared", s.ID(), val)
		}
	}
	t.Logf("✓ ListMergesStateTiers: all listed sessions see app:global")
}
