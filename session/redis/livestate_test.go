// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package redis

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/session"
)

// TestAppendEventUpdatesLiveSessionState guards the intra-invocation state
// contract, matching the official adk in-memory and database services: a
// StateDelta appended through an event must be observable through the SAME
// live session object, without re-fetching from Redis. Anything reading
// ctx.State() later in the invocation depends on this.
func TestAppendEventUpdatesLiveSessionState(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "u1", SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	sess := created.Session

	evt := session.NewEvent(ctx, "inv-1")
	evt.Author = "writer_node"
	evt.Actions.StateDelta["score"] = 0.9
	evt.Actions.StateDelta["app:shared"] = "for-everyone"

	if err := svc.AppendEvent(ctx, sess, evt); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	got, err := sess.State().Get("score")
	if err != nil {
		t.Fatalf("live session state must expose the delta of the same run, got error: %v", err)
	}
	if got != 0.9 {
		t.Fatalf("expected 0.9, got %v", got)
	}
	if got, err := sess.State().Get("app:shared"); err != nil || got != "for-everyone" {
		t.Fatalf("app-tier delta must also reach the live merged state, got %v err %v", got, err)
	}

	// The persisted copy must agree once re-fetched.
	fetched, err := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got, err := fetched.Session.State().Get("score"); err != nil || got != 0.9 {
		t.Fatalf("persisted state disagrees with live state: %v err %v", got, err)
	}
}

// TestAppendEventPreservesEventTimestamp guards against the service stamping
// over a timestamp the emitter already set: consumers of the event history
// rely on emission times, not persistence times.
func TestAppendEventPreservesEventTimestamp(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "u1", SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	original := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	evt := session.NewEvent(ctx, "inv-1")
	evt.Author = "node"
	evt.Timestamp = original

	if err := svc.AppendEvent(ctx, created.Session, evt); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}
	if !evt.Timestamp.Equal(original) {
		t.Fatalf("AppendEvent overwrote the emitter timestamp: %v", evt.Timestamp)
	}
}

// TestLiveStateConcurrentAccess guards the state map against races: adk may
// run agents concurrently within one invocation, so one goroutine may iterate
// or read state while another writes it. Run with -race to make violations
// fatal.
func TestLiveStateConcurrentAccess(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "u1", SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	st := created.Session.State()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func(n int) {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				key := fmt.Sprintf("key_%d_%d", n, j)
				if err := st.Set(key, j); err != nil {
					t.Errorf("Set: %v", err)
					return
				}
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				for range st.All() {
				}
				st.Get("key_0_0")
			}
		}()
	}
	wg.Wait()
}

// TestAppendEventTempKeysVisibleLiveButNotPersisted guards the temp: key
// contract: temporary state keys are invocation-scoped, so they must be
// readable through the live session after AppendEvent, yet excluded from the
// persisted session state and from the stored event.
func TestAppendEventTempKeysVisibleLiveButNotPersisted(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	created, err := svc.Create(ctx, &session.CreateRequest{
		AppName: app, UserID: "u1", SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	sess := created.Session

	evt := session.NewEvent(ctx, "inv-1")
	evt.Author = "node"
	evt.Actions.StateDelta[session.KeyPrefixTemp+"scratch"] = "ephemeral"
	evt.Actions.StateDelta["durable"] = "kept"

	if err := svc.AppendEvent(ctx, sess, evt); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	if got, err := sess.State().Get(session.KeyPrefixTemp + "scratch"); err != nil || got != "ephemeral" {
		t.Fatalf("temp key must be readable in the live session, got %v err %v", got, err)
	}

	fetched, err := svc.Get(ctx, &session.GetRequest{AppName: app, UserID: "u1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if _, err := fetched.Session.State().Get(session.KeyPrefixTemp + "scratch"); err == nil {
		t.Fatal("temp key must not survive persistence")
	}
	if got, err := fetched.Session.State().Get("durable"); err != nil || got != "kept" {
		t.Fatalf("durable key must survive persistence, got %v err %v", got, err)
	}
}

// TestListAcrossUsers guards List parity with the in-memory service: an empty
// UserID lists every user's sessions of the app, and listed sessions carry no
// events.
func TestListAcrossUsers(t *testing.T) {
	svc := setupTestService(t)
	ctx := context.Background()
	app := uniquePrefix(t)

	for _, u := range []string{"u1", "u1", "u2"} {
		if _, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: u}); err != nil {
			t.Fatalf("Create for %s: %v", u, err)
		}
	}

	all, err := svc.List(ctx, &session.ListRequest{AppName: app})
	if err != nil {
		t.Fatalf("List all users: %v", err)
	}
	if len(all.Sessions) != 3 {
		t.Fatalf("expected 3 sessions across users, got %d", len(all.Sessions))
	}

	one, err := svc.List(ctx, &session.ListRequest{AppName: app, UserID: "u2"})
	if err != nil {
		t.Fatalf("List u2: %v", err)
	}
	if len(one.Sessions) != 1 {
		t.Fatalf("expected 1 session for u2, got %d", len(one.Sessions))
	}
	if one.Sessions[0].Events().Len() != 0 {
		t.Fatal("listed sessions must not carry events")
	}

	if _, err := svc.List(ctx, &session.ListRequest{}); err == nil {
		t.Fatal("List without app_name must fail")
	}
}
