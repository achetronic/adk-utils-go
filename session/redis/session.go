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
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"time"

	"github.com/redis/go-redis/v9"
	"google.golang.org/adk/session"
)

// RedisSessionService implements session.Service using Redis as the backend.
type RedisSessionService struct {
	client *redis.Client
	ttl    time.Duration
}

// RedisSessionServiceConfig holds configuration for RedisSessionService.
type RedisSessionServiceConfig struct {
	// Addr is the Redis server address (e.g., "localhost:6379")
	Addr string
	// Password for Redis authentication (optional)
	Password string
	// DB is the Redis database number
	DB int
	// TTL is the session expiration time (default: 24 hours)
	TTL time.Duration
}

// NewRedisSessionService creates a new Redis-backed session service.
func NewRedisSessionService(cfg RedisSessionServiceConfig) (*RedisSessionService, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     cfg.Addr,
		Password: cfg.Password,
		DB:       cfg.DB,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	ttl := cfg.TTL
	if ttl == 0 {
		ttl = 24 * time.Hour
	}

	return &RedisSessionService{
		client: client,
		ttl:    ttl,
	}, nil
}

// Key helpers
func (s *RedisSessionService) sessionKey(appName, userID, sessionID string) string {
	return fmt.Sprintf("session:%s:%s:%s", appName, userID, sessionID)
}

func (s *RedisSessionService) sessionsIndexKey(appName, userID string) string {
	return fmt.Sprintf("sessions:%s:%s", appName, userID)
}

func (s *RedisSessionService) eventsKey(appName, userID, sessionID string) string {
	return fmt.Sprintf("events:%s:%s:%s", appName, userID, sessionID)
}

// Create creates a new session.
func (s *RedisSessionService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = fmt.Sprintf("%d", time.Now().UnixNano())
	}

	key := s.sessionKey(req.AppName, req.UserID, sessionID)
	eventsKey := s.eventsKey(req.AppName, req.UserID, sessionID)

	sess := &redisSession{
		id:             sessionID,
		appName:        req.AppName,
		userID:         req.UserID,
		state:          newRedisState(req.State, s.client, key, s.ttl),
		events:         newRedisEvents(nil, s.client, eventsKey),
		lastUpdateTime: time.Now(),
	}

	data, err := json.Marshal(sess.toStorable())
	if err != nil {
		return nil, fmt.Errorf("failed to marshal session: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return nil, fmt.Errorf("failed to store session: %w", err)
	}

	// Add to sessions index
	indexKey := s.sessionsIndexKey(req.AppName, req.UserID)
	if err := s.client.SAdd(ctx, indexKey, sessionID).Err(); err != nil {
		return nil, fmt.Errorf("failed to update sessions index: %w", err)
	}
	s.client.Expire(ctx, indexKey, s.ttl)

	return &session.CreateResponse{Session: sess}, nil
}

// Get retrieves a session by ID.
func (s *RedisSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	key := s.sessionKey(req.AppName, req.UserID, req.SessionID)

	data, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, fmt.Errorf("session not found: %s", req.SessionID)
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	var storable storableSession
	if err := json.Unmarshal(data, &storable); err != nil {
		return nil, fmt.Errorf("failed to unmarshal session: %w", err)
	}

	// Load events
	eventsKey := s.eventsKey(req.AppName, req.UserID, req.SessionID)
	eventData, err := s.client.LRange(ctx, eventsKey, 0, -1).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		return nil, fmt.Errorf("failed to get events: %w", err)
	}

	var events []*session.Event
	for _, ed := range eventData {
		var evt session.Event
		if err := json.Unmarshal([]byte(ed), &evt); err != nil {
			continue
		}
		events = append(events, &evt)
	}

	// Apply filters
	if req.NumRecentEvents > 0 && len(events) > req.NumRecentEvents {
		events = events[len(events)-req.NumRecentEvents:]
	}
	if !req.After.IsZero() {
		var filtered []*session.Event
		for _, evt := range events {
			if !evt.Timestamp.Before(req.After) {
				filtered = append(filtered, evt)
			}
		}
		events = filtered
	}

	sess := &redisSession{
		id:             storable.ID,
		appName:        storable.AppName,
		userID:         storable.UserID,
		state:          newRedisState(storable.State, s.client, key, s.ttl),
		events:         newRedisEvents(events, s.client, eventsKey),
		lastUpdateTime: storable.LastUpdateTime,
	}

	return &session.GetResponse{Session: sess}, nil
}

// List returns all sessions for a user.
func (s *RedisSessionService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	indexKey := s.sessionsIndexKey(req.AppName, req.UserID)

	sessionIDs, err := s.client.SMembers(ctx, indexKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to list sessions: %w", err)
	}

	var sessions []session.Session
	for _, sessionID := range sessionIDs {
		resp, err := s.Get(ctx, &session.GetRequest{
			AppName:   req.AppName,
			UserID:    req.UserID,
			SessionID: sessionID,
		})
		if err != nil {
			continue // Skip sessions that can't be retrieved
		}
		sessions = append(sessions, resp.Session)
	}

	return &session.ListResponse{Sessions: sessions}, nil
}

// Delete removes a session.
func (s *RedisSessionService) Delete(ctx context.Context, req *session.DeleteRequest) error {
	key := s.sessionKey(req.AppName, req.UserID, req.SessionID)
	eventsKey := s.eventsKey(req.AppName, req.UserID, req.SessionID)
	indexKey := s.sessionsIndexKey(req.AppName, req.UserID)

	pipe := s.client.Pipeline()
	pipe.Del(ctx, key)
	pipe.Del(ctx, eventsKey)
	pipe.SRem(ctx, indexKey, req.SessionID)

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}

	return nil
}

// AppendEvent appends an event to a session.
func (s *RedisSessionService) AppendEvent(ctx context.Context, sess session.Session, evt *session.Event) error {
	evt.Timestamp = time.Now()
	if evt.ID == "" {
		evt.ID = fmt.Sprintf("%d", time.Now().UnixNano())
	}

	data, err := json.Marshal(evt)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	eventsKey := s.eventsKey(sess.AppName(), sess.UserID(), sess.ID())
	if err := s.client.RPush(ctx, eventsKey, data).Err(); err != nil {
		return fmt.Errorf("failed to append event: %w", err)
	}
	s.client.Expire(ctx, eventsKey, s.ttl)

	// Update session's last update time and persist current state
	key := s.sessionKey(sess.AppName(), sess.UserID(), sess.ID())
	sessData, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		return fmt.Errorf("failed to get session for update: %w", err)
	}

	var storable storableSession
	if err := json.Unmarshal(sessData, &storable); err != nil {
		return fmt.Errorf("failed to unmarshal session: %w", err)
	}

	// Sync state from session to storable
	state := sess.State()
	if state != nil {
		storable.State = make(map[string]any)
		for k, v := range state.All() {
			storable.State[k] = v
		}
	}

	storable.LastUpdateTime = time.Now()
	updatedData, err := json.Marshal(storable)
	if err != nil {
		return fmt.Errorf("failed to marshal updated session: %w", err)
	}

	if err := s.client.Set(ctx, key, updatedData, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to update session: %w", err)
	}

	return nil
}

// Close closes the Redis connection.
func (s *RedisSessionService) Close() error {
	return s.client.Close()
}

// storableSession is the JSON-serializable representation of a session.
type storableSession struct {
	ID             string         `json:"id"`
	AppName        string         `json:"app_name"`
	UserID         string         `json:"user_id"`
	State          map[string]any `json:"state"`
	LastUpdateTime time.Time      `json:"last_update_time"`
}

// redisSession implements session.Session.
type redisSession struct {
	id             string
	appName        string
	userID         string
	state          *redisState
	events         *redisEvents
	lastUpdateTime time.Time
}

func (s *redisSession) ID() string                { return s.id }
func (s *redisSession) AppName() string           { return s.appName }
func (s *redisSession) UserID() string            { return s.userID }
func (s *redisSession) State() session.State      { return s.state }
func (s *redisSession) Events() session.Events    { return s.events }
func (s *redisSession) LastUpdateTime() time.Time { return s.lastUpdateTime }

func (s *redisSession) toStorable() storableSession {
	state := make(map[string]any)
	for k, v := range s.state.All() {
		state[k] = v
	}
	return storableSession{
		ID:             s.id,
		AppName:        s.appName,
		UserID:         s.userID,
		State:          state,
		LastUpdateTime: s.lastUpdateTime,
	}
}

// redisState implements session.State with Redis persistence.
type redisState struct {
	data   map[string]any
	client *redis.Client
	key    string
	ttl    time.Duration
}

func newRedisState(initial map[string]any, client *redis.Client, key string, ttl time.Duration) *redisState {
	data := make(map[string]any)
	for k, v := range initial {
		data[k] = v
	}
	return &redisState{
		data:   data,
		client: client,
		key:    key,
		ttl:    ttl,
	}
}

func (s *redisState) Get(key string) (any, error) {
	v, ok := s.data[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (s *redisState) Set(key string, value any) error {
	s.data[key] = value

	// Persist to Redis immediately
	return s.persist()
}

func (s *redisState) persist() error {
	ctx := context.Background()

	// Get current session data
	data, err := s.client.Get(ctx, s.key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil // Session doesn't exist yet, will be created
		}
		return fmt.Errorf("failed to get session for state update: %w", err)
	}

	var storable storableSession
	if err := json.Unmarshal(data, &storable); err != nil {
		return fmt.Errorf("failed to unmarshal session: %w", err)
	}

	// Update state
	storable.State = make(map[string]any)
	for k, v := range s.data {
		storable.State[k] = v
	}
	storable.LastUpdateTime = time.Now()

	// Save back
	updatedData, err := json.Marshal(storable)
	if err != nil {
		return fmt.Errorf("failed to marshal updated session: %w", err)
	}

	if err := s.client.Set(ctx, s.key, updatedData, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to persist state: %w", err)
	}

	return nil
}

func (s *redisState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		for k, v := range s.data {
			if !yield(k, v) {
				return
			}
		}
	}
}

// redisEvents implements session.Events with live Redis reads.
type redisEvents struct {
	client *redis.Client
	key    string
	// cached events for when we don't have Redis connection info
	cached []*session.Event
}

func newRedisEvents(events []*session.Event, client *redis.Client, key string) *redisEvents {
	if events == nil {
		events = make([]*session.Event, 0)
	}
	return &redisEvents{
		client: client,
		key:    key,
		cached: events,
	}
}

func (e *redisEvents) loadFromRedis() []*session.Event {
	if e.client == nil || e.key == "" {
		return e.cached
	}

	ctx := context.Background()
	eventData, err := e.client.LRange(ctx, e.key, 0, -1).Result()
	if err != nil {
		return e.cached
	}

	var events []*session.Event
	for _, ed := range eventData {
		var evt session.Event
		if err := json.Unmarshal([]byte(ed), &evt); err != nil {
			continue
		}
		events = append(events, &evt)
	}
	return events
}

func (e *redisEvents) All() iter.Seq[*session.Event] {
	events := e.loadFromRedis()
	return func(yield func(*session.Event) bool) {
		for _, evt := range events {
			if !yield(evt) {
				return
			}
		}
	}
}

func (e *redisEvents) Len() int {
	events := e.loadFromRedis()
	return len(events)
}

func (e *redisEvents) At(i int) *session.Event {
	events := e.loadFromRedis()
	if i < 0 || i >= len(events) {
		return nil
	}
	return events[i]
}

// Ensure interfaces are implemented
var _ session.Service = (*RedisSessionService)(nil)
var _ session.Session = (*redisSession)(nil)
var _ session.State = (*redisState)(nil)
var _ session.Events = (*redisEvents)(nil)
