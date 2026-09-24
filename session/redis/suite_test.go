// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package redis

import (
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/sessiontestsuite"
)

// The suite expects setup() to hand each subtest an empty store, like the
// in-memory reference does. Redis is a shared server, so each service
// instance gets its own logical database instead.
var suiteDB atomic.Int32

// TestServiceSuite runs the official ADK conformance suite for session
// services against the Redis backend, so behavioural drift from the
// reference implementations surfaces here instead of in consumers.
func TestServiceSuite(t *testing.T) {
	sessiontestsuite.RunServiceTests(t, sessiontestsuite.SuiteOptions{
		SupportsUserProvidedSessionID: true,
	}, func(t *testing.T) session.Service {
		// Stock Redis ships 16 logical databases and the suite asks for
		// more fresh stores than that, so recycle them: the flush below
		// is what actually isolates each subtest.
		db := 1 + int(suiteDB.Add(1))%14
		svc, err := NewRedisSessionService(RedisSessionServiceConfig{
			Addr: testRedisAddr,
			DB:   db,
			TTL:  5 * time.Minute,
		})
		if err != nil {
			t.Fatalf("Failed to create Redis session service: %v", err)
		}
		// Logical databases are reused between runs; wipe ours so every
		// subtest starts from an empty store like the in-memory reference.
		if err := svc.client.FlushDB(t.Context()).Err(); err != nil {
			t.Fatalf("Failed to flush Redis database: %v", err)
		}
		t.Cleanup(func() { svc.Close() })
		return svc
	})
}
