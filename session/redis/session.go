// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package redis

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"google.golang.org/adk/v2/session"
)

// RedisSessionService implements session.Service using Redis as the backend.
type RedisSessionService struct {
	client       *redis.Client
	ttl          time.Duration
	appStateTTL  time.Duration
	userStateTTL time.Duration
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
	// AppStateTTL is the expiration time for app-scoped state.
	// Defaults to 0 (no expiration), matching the canonical ADK behaviour
	// where app state outlives individual sessions.
	AppStateTTL time.Duration
	// UserStateTTL is the expiration time for user-scoped state.
	// Defaults to 0 (no expiration), matching the canonical ADK behaviour
	// where user state outlives individual sessions.
	UserStateTTL time.Duration
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
		client:       client,
		ttl:          ttl,
		appStateTTL:  cfg.AppStateTTL,
		userStateTTL: cfg.UserStateTTL,
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

func (s *RedisSessionService) appStateKey(appName string) string {
	return fmt.Sprintf("appstate:%s", appName)
}

func (s *RedisSessionService) userStateKey(appName, userID string) string {
	return fmt.Sprintf("userstate:%s:%s", appName, userID)
}

// Create creates a new session. It returns an error if a session with the
// same ID already exists, matching the canonical ADK behaviour.
func (s *RedisSessionService) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	sessionID := req.SessionID
	if sessionID == "" {
		sessionID = fmt.Sprintf("%d", time.Now().UnixNano())
	}

	key := s.sessionKey(req.AppName, req.UserID, sessionID)
	eventsKey := s.eventsKey(req.AppName, req.UserID, sessionID)

	if exists, _ := s.client.Exists(ctx, key).Result(); exists > 0 {
		return nil, fmt.Errorf("session %s already exists", sessionID)
	}

	appDelta, userDelta, sessionDelta := extractStateDeltas(req.State)

	appState := s.updateAppState(ctx, req.AppName, appDelta)
	userState := s.updateUserState(ctx, req.AppName, req.UserID, userDelta)
	mergedState := mergeStates(appState, userState, sessionDelta)

	sess := &redisSession{
		id:             sessionID,
		appName:        req.AppName,
		userID:         req.UserID,
		state:          newRedisState(mergedState, s.client, key, s.ttl, s, req.AppName, req.UserID),
		events:         newRedisEvents(nil, s.client, eventsKey),
		lastUpdateTime: time.Now(),
	}

	storable := storableSession{
		ID:             sessionID,
		AppName:        req.AppName,
		UserID:         req.UserID,
		State:          sessionDelta,
		LastUpdateTime: sess.lastUpdateTime,
	}

	data, err := json.Marshal(storable)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal session: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return nil, fmt.Errorf("failed to store session: %w", err)
	}

	indexKey := s.sessionsIndexKey(req.AppName, req.UserID)
	if err := s.client.SAdd(ctx, indexKey, sessionID).Err(); err != nil {
		return nil, fmt.Errorf("failed to update sessions index: %w", err)
	}
	s.client.Expire(ctx, indexKey, s.ttl)

	return &session.CreateResponse{Session: sess}, nil
}

// Get retrieves a session by ID.
func (s *RedisSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, fmt.Errorf("app_name, user_id, session_id are required, got app_name: %q, user_id: %q, session_id: %q", req.AppName, req.UserID, req.SessionID)
	}
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

	appState := s.loadAppState(ctx, req.AppName)
	userState := s.loadUserState(ctx, req.AppName, req.UserID)
	mergedState := mergeStates(appState, userState, storable.State)

	filtered := req.NumRecentEvents > 0 || !req.After.IsZero()

	sess := &redisSession{
		id:             storable.ID,
		appName:        storable.AppName,
		userID:         storable.UserID,
		state:          newRedisState(mergedState, s.client, key, s.ttl, s, req.AppName, req.UserID),
		lastUpdateTime: storable.LastUpdateTime,
	}
	if filtered {
		sess.events = newFilteredRedisEvents(events)
	} else {
		sess.events = newRedisEvents(events, s.client, eventsKey)
	}

	return &session.GetResponse{Session: sess}, nil
}

// List returns the sessions of a user, or of every user of the app when
// UserID is empty, mirroring the in-memory service. Sessions are returned
// without their events, also matching the reference behaviour.
func (s *RedisSessionService) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}

	keys, err := s.listSessionKeys(ctx, req.AppName, req.UserID)
	if err != nil {
		return nil, err
	}

	sessions := make([]session.Session, 0, len(keys))
	for _, key := range keys {
		data, err := s.client.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}
		var storable storableSession
		if err := json.Unmarshal(data, &storable); err != nil {
			continue
		}

		appState := s.loadAppState(ctx, storable.AppName)
		userState := s.loadUserState(ctx, storable.AppName, storable.UserID)
		mergedState := mergeStates(appState, userState, storable.State)

		sessions = append(sessions, &redisSession{
			id:             storable.ID,
			appName:        storable.AppName,
			userID:         storable.UserID,
			state:          newRedisState(mergedState, s.client, key, s.ttl, s, storable.AppName, storable.UserID),
			events:         newFilteredRedisEvents(nil),
			lastUpdateTime: storable.LastUpdateTime,
		})
	}

	return &session.ListResponse{Sessions: sessions}, nil
}

// listSessionKeys resolves the session keys to list: directly from the
// per-user index when UserID is set, or by scanning the app's session key
// space when it is not (the index sets are keyed per user).
func (s *RedisSessionService) listSessionKeys(ctx context.Context, appName, userID string) ([]string, error) {
	if userID != "" {
		sessionIDs, err := s.client.SMembers(ctx, s.sessionsIndexKey(appName, userID)).Result()
		if err != nil {
			return nil, fmt.Errorf("failed to list sessions: %w", err)
		}
		keys := make([]string, 0, len(sessionIDs))
		for _, id := range sessionIDs {
			keys = append(keys, s.sessionKey(appName, userID, id))
		}
		return keys, nil
	}

	var keys []string
	pattern := fmt.Sprintf("session:%s:*", appName)
	iter := s.client.Scan(ctx, 0, pattern, 100).Iterator()
	for iter.Next(ctx) {
		keys = append(keys, iter.Val())
	}
	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan sessions: %w", err)
	}
	return keys, nil
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

// AppendEvent appends an event to a session and applies its StateDelta both to
// the persisted session state and to the live session object, matching the
// behaviour of the official ADK in-memory and database session service
// implementations. Without the live-state update, later reads of the session
// state within the same invocation would never observe deltas appended
// earlier in that invocation.
func (s *RedisSessionService) AppendEvent(ctx context.Context, sess session.Session, evt *session.Event) error {
	if evt.Partial {
		return nil
	}

	if evt.Timestamp.IsZero() {
		evt.Timestamp = time.Now()
	}
	if evt.ID == "" {
		evt.ID = fmt.Sprintf("%d", time.Now().UnixNano())
	}

	// The live state receives the delta BEFORE temp: keys are trimmed:
	// temporary keys are invocation-scoped and must be readable for the rest
	// of the invocation, they are only excluded from persistence. Mirrors the
	// in-memory service, which updates session state and then trims the
	// stored copy.
	liveDelta := evt.Actions.StateDelta

	trimTempStateDelta(evt)

	data, err := json.Marshal(evt)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	eventsKey := s.eventsKey(sess.AppName(), sess.UserID(), sess.ID())
	if err := s.client.RPush(ctx, eventsKey, data).Err(); err != nil {
		return fmt.Errorf("failed to append event: %w", err)
	}
	s.client.Expire(ctx, eventsKey, s.ttl)

	key := s.sessionKey(sess.AppName(), sess.UserID(), sess.ID())
	sessData, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		return fmt.Errorf("failed to get session for update: %w", err)
	}

	var storable storableSession
	if err := json.Unmarshal(sessData, &storable); err != nil {
		return fmt.Errorf("failed to unmarshal session: %w", err)
	}

	if storable.State == nil {
		storable.State = make(map[string]any)
	}

	state := sess.State()
	if state != nil {
		for k, v := range state.All() {
			_, _, sessionOnly := extractSingleKey(k, v)
			if sessionOnly != nil {
				for sk, sv := range sessionOnly {
					storable.State[sk] = sv
				}
			}
		}
	}

	if len(evt.Actions.StateDelta) > 0 {
		appDelta, userDelta, sessionDelta := extractStateDeltas(evt.Actions.StateDelta)
		s.updateAppState(ctx, sess.AppName(), appDelta)
		s.updateUserState(ctx, sess.AppName(), sess.UserID(), userDelta)
		for k, v := range sessionDelta {
			storable.State[k] = v
		}
	}

	// LastUpdateTime tracks the event timestamp, not the wall clock at
	// persistence time, mirroring the in-memory service.
	storable.LastUpdateTime = evt.Timestamp
	updatedData, err := json.Marshal(storable)
	if err != nil {
		return fmt.Errorf("failed to marshal updated session: %w", err)
	}

	if err := s.client.Set(ctx, key, updatedData, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to update session: %w", err)
	}

	// Apply the delta to the live session state so the current invocation
	// observes it immediately. The merged map keeps app:/user: keys prefixed
	// and session keys plain, exactly the shape the delta arrives in.
	if len(liveDelta) > 0 {
		if rs, ok := sess.(*redisSession); ok {
			rs.state.applyLocal(liveDelta)
			rs.touch(evt.Timestamp)
		} else if st := sess.State(); st != nil {
			for k, v := range liveDelta {
				if err := st.Set(k, v); err != nil {
					return fmt.Errorf("failed to apply state delta to live session: %w", err)
				}
			}
		}
	}

	return nil
}

// updateAppState writes app-scoped state deltas to Redis and returns the full
// app state. Uses a Redis HASH for atomic field-level updates.
func (s *RedisSessionService) updateAppState(ctx context.Context, appName string, delta map[string]any) map[string]any {
	if len(delta) == 0 {
		return s.loadAppState(ctx, appName)
	}

	key := s.appStateKey(appName)
	fields := marshalHashFields(delta)
	s.client.HSet(ctx, key, fields)
	if s.appStateTTL > 0 {
		s.client.Expire(ctx, key, s.appStateTTL)
	} else {
		s.client.Persist(ctx, key)
	}
	return s.loadAppState(ctx, appName)
}

// updateUserState writes user-scoped state deltas to Redis and returns the full
// user state. Uses a Redis HASH for atomic field-level updates.
func (s *RedisSessionService) updateUserState(ctx context.Context, appName, userID string, delta map[string]any) map[string]any {
	if len(delta) == 0 {
		return s.loadUserState(ctx, appName, userID)
	}

	key := s.userStateKey(appName, userID)
	fields := marshalHashFields(delta)
	s.client.HSet(ctx, key, fields)
	if s.userStateTTL > 0 {
		s.client.Expire(ctx, key, s.userStateTTL)
	} else {
		s.client.Persist(ctx, key)
	}
	return s.loadUserState(ctx, appName, userID)
}

// loadAppState loads the full app state from Redis.
func (s *RedisSessionService) loadAppState(ctx context.Context, appName string) map[string]any {
	return s.loadHashState(ctx, s.appStateKey(appName))
}

// loadUserState loads the full user state from Redis.
func (s *RedisSessionService) loadUserState(ctx context.Context, appName, userID string) map[string]any {
	return s.loadHashState(ctx, s.userStateKey(appName, userID))
}

// loadHashState loads all fields from a Redis HASH and unmarshals values from JSON.
func (s *RedisSessionService) loadHashState(ctx context.Context, key string) map[string]any {
	result, err := s.client.HGetAll(ctx, key).Result()
	if err != nil || len(result) == 0 {
		return make(map[string]any)
	}
	return unmarshalHashFields(result)
}

// marshalHashFields converts a map[string]any to a map[string]string suitable
// for Redis HASH storage by JSON-encoding each value.
func marshalHashFields(m map[string]any) map[string]string {
	fields := make(map[string]string, len(m))
	for k, v := range m {
		data, err := json.Marshal(v)
		if err != nil {
			continue
		}
		fields[k] = string(data)
	}
	return fields
}

// unmarshalHashFields converts a Redis HASH result back to map[string]any by
// JSON-decoding each value.
func unmarshalHashFields(fields map[string]string) map[string]any {
	m := make(map[string]any, len(fields))
	for k, v := range fields {
		var val any
		if err := json.Unmarshal([]byte(v), &val); err != nil {
			m[k] = v
			continue
		}
		m[k] = val
	}
	return m
}

// extractStateDeltas splits a flat state map into three separate maps based on
// key prefixes, mirroring google.golang.org/adk/v2/internal/sessionutils.ExtractStateDeltas.
// Keys with the "temp:" prefix are discarded.
func extractStateDeltas(delta map[string]any) (appDelta, userDelta, sessionDelta map[string]any) {
	appDelta = make(map[string]any)
	userDelta = make(map[string]any)
	sessionDelta = make(map[string]any)

	if delta == nil {
		return appDelta, userDelta, sessionDelta
	}

	for key, value := range delta {
		if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixApp); found {
			appDelta[cleanKey] = value
		} else if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixUser); found {
			userDelta[cleanKey] = value
		} else if !strings.HasPrefix(key, session.KeyPrefixTemp) {
			sessionDelta[key] = value
		}
	}
	return appDelta, userDelta, sessionDelta
}

// extractSingleKey classifies a single key-value pair into its state tier.
// Returns non-nil maps only for the tier the key belongs to. Used when syncing
// in-memory session state back to the storable (session-scoped only).
func extractSingleKey(key string, value any) (app, user, sessionOnly map[string]any) {
	if strings.HasPrefix(key, session.KeyPrefixApp) || strings.HasPrefix(key, session.KeyPrefixUser) || strings.HasPrefix(key, session.KeyPrefixTemp) {
		return nil, nil, nil
	}
	return nil, nil, map[string]any{key: value}
}

// mergeStates combines app, user, and session state maps into a single flat map,
// re-adding the appropriate prefixes, mirroring
// google.golang.org/adk/v2/internal/sessionutils.MergeStates.
func mergeStates(appState, userState, sessionState map[string]any) map[string]any {
	totalSize := len(appState) + len(userState) + len(sessionState)
	merged := make(map[string]any, totalSize)

	for k, v := range sessionState {
		merged[k] = v
	}
	for k, v := range appState {
		merged[session.KeyPrefixApp+k] = v
	}
	for k, v := range userState {
		merged[session.KeyPrefixUser+k] = v
	}
	return merged
}

// trimTempStateDelta removes keys with the "temp:" prefix from the event's
// StateDelta. These keys are meant to be ephemeral (live only for the current
// invocation) and must not be persisted, matching the ADK's trimTempDeltaState.
func trimTempStateDelta(evt *session.Event) {
	if len(evt.Actions.StateDelta) == 0 {
		return
	}
	filtered := make(map[string]any, len(evt.Actions.StateDelta))
	for k, v := range evt.Actions.StateDelta {
		if !strings.HasPrefix(k, session.KeyPrefixTemp) {
			filtered[k] = v
		}
	}
	evt.Actions.StateDelta = filtered
}

// Close closes the Redis connection.
func (s *RedisSessionService) Close() error {
	return s.client.Close()
}

// storableSession is the JSON-serializable representation of a session.
// State only contains session-scoped keys (no app: or user: prefixed keys).
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

func (s *redisSession) ID() string             { return s.id }
func (s *redisSession) AppName() string        { return s.appName }
func (s *redisSession) UserID() string         { return s.userID }
func (s *redisSession) State() session.State   { return s.state }
func (s *redisSession) Events() session.Events { return s.events }

func (s *redisSession) LastUpdateTime() time.Time {
	s.state.mu.RLock()
	defer s.state.mu.RUnlock()
	return s.lastUpdateTime
}

// touch advances the session's last update time. Guarded by the state mutex
// because AppendEvent may run while another goroutine reads the session.
func (s *redisSession) touch(t time.Time) {
	s.state.mu.Lock()
	s.lastUpdateTime = t
	s.state.mu.Unlock()
}

func (s *redisSession) toStorable() storableSession {
	sessionOnly := make(map[string]any)
	for k, v := range s.state.All() {
		if !strings.HasPrefix(k, session.KeyPrefixApp) && !strings.HasPrefix(k, session.KeyPrefixUser) && !strings.HasPrefix(k, session.KeyPrefixTemp) {
			sessionOnly[k] = v
		}
	}
	return storableSession{
		ID:             s.id,
		AppName:        s.appName,
		UserID:         s.userID,
		State:          sessionOnly,
		LastUpdateTime: s.lastUpdateTime,
	}
}

// redisState implements session.State with Redis persistence.
// It holds the merged (all tiers) state in memory and routes writes to the
// correct Redis key based on the key prefix. The mutex guards the in-memory
// map: adk agents may run concurrently within one invocation (parallel agents,
// workflow branches), so reads and writes can otherwise race.
type redisState struct {
	mu      sync.RWMutex
	data    map[string]any
	client  *redis.Client
	key     string
	ttl     time.Duration
	service *RedisSessionService
	appName string
	userID  string
}

func newRedisState(initial map[string]any, client *redis.Client, key string, ttl time.Duration, service *RedisSessionService, appName, userID string) *redisState {
	data := make(map[string]any)
	for k, v := range initial {
		data[k] = v
	}
	return &redisState{
		data:    data,
		client:  client,
		key:     key,
		ttl:     ttl,
		service: service,
		appName: appName,
		userID:  userID,
	}
}

func (s *redisState) Get(key string) (any, error) {
	s.mu.RLock()
	v, ok := s.data[key]
	s.mu.RUnlock()
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

// applyLocal merges a state delta into the in-memory map without touching
// Redis. Used by AppendEvent, which has already persisted the delta itself.
func (s *redisState) applyLocal(delta map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, v := range delta {
		s.data[k] = v
	}
}

func (s *redisState) Set(key string, value any) error {
	s.mu.Lock()
	s.data[key] = value
	s.mu.Unlock()

	ctx := context.Background()

	if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixApp); found {
		appKey := s.service.appStateKey(s.appName)
		data, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("failed to marshal app state value: %w", err)
		}
		s.client.HSet(ctx, appKey, cleanKey, string(data))
		if s.service.appStateTTL > 0 {
			s.client.Expire(ctx, appKey, s.service.appStateTTL)
		} else {
			s.client.Persist(ctx, appKey)
		}
		return nil
	}

	if cleanKey, found := strings.CutPrefix(key, session.KeyPrefixUser); found {
		userKey := s.service.userStateKey(s.appName, s.userID)
		data, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("failed to marshal user state value: %w", err)
		}
		s.client.HSet(ctx, userKey, cleanKey, string(data))
		if s.service.userStateTTL > 0 {
			s.client.Expire(ctx, userKey, s.service.userStateTTL)
		} else {
			s.client.Persist(ctx, userKey)
		}
		return nil
	}

	if strings.HasPrefix(key, session.KeyPrefixTemp) {
		return nil
	}

	return s.persistSessionState()
}

func (s *redisState) persistSessionState() error {
	ctx := context.Background()

	data, err := s.client.Get(ctx, s.key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil
		}
		return fmt.Errorf("failed to get session for state update: %w", err)
	}

	var storable storableSession
	if err := json.Unmarshal(data, &storable); err != nil {
		return fmt.Errorf("failed to unmarshal session: %w", err)
	}

	storable.State = make(map[string]any)
	s.mu.RLock()
	for k, v := range s.data {
		if !strings.HasPrefix(k, session.KeyPrefixApp) && !strings.HasPrefix(k, session.KeyPrefixUser) && !strings.HasPrefix(k, session.KeyPrefixTemp) {
			storable.State[k] = v
		}
	}
	s.mu.RUnlock()
	storable.LastUpdateTime = time.Now()

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
	// Copy under the read lock so iteration never races with a concurrent
	// write from a parallel branch, mirroring the in-memory service.
	s.mu.RLock()
	snapshot := make(map[string]any, len(s.data))
	for k, v := range s.data {
		snapshot[k] = v
	}
	s.mu.RUnlock()
	return func(yield func(string, any) bool) {
		for k, v := range snapshot {
			if !yield(k, v) {
				return
			}
		}
	}
}

// redisEvents implements session.Events with live Redis reads.
// When filtered is true, the cached slice is the authoritative source (e.g.
// after Get applied NumRecentEvents / After filters) and loadFromRedis returns
// it directly without re-fetching.
type redisEvents struct {
	client   *redis.Client
	key      string
	cached   []*session.Event
	filtered bool
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

func newFilteredRedisEvents(events []*session.Event) *redisEvents {
	if events == nil {
		events = make([]*session.Event, 0)
	}
	return &redisEvents{
		cached:   events,
		filtered: true,
	}
}

func (e *redisEvents) loadFromRedis() []*session.Event {
	if e.filtered || e.client == nil || e.key == "" {
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
