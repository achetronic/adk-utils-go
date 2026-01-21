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
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// EmbeddingModel is an interface for generating embeddings from text.
type EmbeddingModel interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Dimension() int
}

// PostgresMemoryService implements memory.Service using PostgreSQL with pgvector.
type PostgresMemoryService struct {
	db             *sql.DB
	embeddingModel EmbeddingModel
	embeddingDim   int
}

// PostgresMemoryServiceConfig holds configuration for PostgresMemoryService.
type PostgresMemoryServiceConfig struct {
	// ConnString is the PostgreSQL connection string
	// e.g., "postgres://user:pass@localhost:5432/dbname?sslmode=disable"
	ConnString string
	// EmbeddingModel is used to generate embeddings for semantic search (optional)
	EmbeddingModel EmbeddingModel
}

// NewPostgresMemoryService creates a new PostgreSQL-backed memory service.
func NewPostgresMemoryService(ctx context.Context, cfg PostgresMemoryServiceConfig) (*PostgresMemoryService, error) {
	db, err := sql.Open("postgres", cfg.ConnString)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	embeddingDim := 0
	if cfg.EmbeddingModel != nil {
		embeddingDim = cfg.EmbeddingModel.Dimension()
		// If dimension is not preset, probe the model to auto-detect it
		if embeddingDim == 0 {
			embedding, err := cfg.EmbeddingModel.Embed(ctx, "dimension probe")
			if err != nil {
				return nil, fmt.Errorf("failed to probe embedding dimension: %w", err)
			}
			embeddingDim = len(embedding)
		}
	}

	svc := &PostgresMemoryService{
		db:             db,
		embeddingModel: cfg.EmbeddingModel,
		embeddingDim:   embeddingDim,
	}

	if err := svc.initSchema(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return svc, nil
}

// initSchema creates the necessary tables and extensions.
func (s *PostgresMemoryService) initSchema(ctx context.Context) error {
	// Base schema without vector column
	baseSchema := `
		-- Memory entries table
		CREATE TABLE IF NOT EXISTS memory_entries (
			id SERIAL PRIMARY KEY,
			app_name VARCHAR(255) NOT NULL,
			user_id VARCHAR(255) NOT NULL,
			session_id VARCHAR(255) NOT NULL,
			event_id VARCHAR(255) NOT NULL,
			author VARCHAR(255),
			content JSONB NOT NULL,
			content_text TEXT NOT NULL,
			timestamp TIMESTAMPTZ NOT NULL,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			UNIQUE(app_name, user_id, session_id, event_id)
		);

		-- Indexes for efficient querying
		CREATE INDEX IF NOT EXISTS idx_memory_app_user ON memory_entries(app_name, user_id);
		CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries(session_id);
		CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_entries(timestamp);
		CREATE INDEX IF NOT EXISTS idx_memory_content_text ON memory_entries USING gin(to_tsvector('english', content_text));
	`

	if _, err := s.db.ExecContext(ctx, baseSchema); err != nil {
		return fmt.Errorf("failed to create base schema: %w", err)
	}

	// Add vector column if embedding model is configured
	if s.embeddingDim > 0 {
		vectorSchema := fmt.Sprintf(`
			-- Enable pgvector extension
			CREATE EXTENSION IF NOT EXISTS vector;

			-- Add embedding column if not exists
			DO $$
			BEGIN
				IF NOT EXISTS (
					SELECT 1 FROM information_schema.columns 
					WHERE table_name = 'memory_entries' AND column_name = 'embedding'
				) THEN
					ALTER TABLE memory_entries ADD COLUMN embedding vector(%d);
				END IF;
			END $$;

			-- Vector similarity index (IVFFlat for approximate nearest neighbor)
			DO $$
			BEGIN
				IF NOT EXISTS (
					SELECT 1 FROM pg_indexes WHERE indexname = 'idx_memory_embedding'
				) THEN
					CREATE INDEX idx_memory_embedding ON memory_entries 
					USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
				END IF;
			END $$;
		`, s.embeddingDim)

		if _, err := s.db.ExecContext(ctx, vectorSchema); err != nil {
			return fmt.Errorf("failed to create vector schema: %w", err)
		}
	}

	return nil
}

// AddSession extracts memory entries from a session and stores them.
func (s *PostgresMemoryService) AddSession(ctx context.Context, sess session.Session) error {
	events := sess.Events()
	if events == nil || events.Len() == 0 {
		return nil
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Prepare statement based on whether we have embeddings
	var stmt *sql.Stmt
	if s.embeddingModel != nil {
		stmt, err = tx.PrepareContext(ctx, `
			INSERT INTO memory_entries (app_name, user_id, session_id, event_id, author, content, content_text, embedding, timestamp)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
			ON CONFLICT (app_name, user_id, session_id, event_id) DO UPDATE 
			SET content = EXCLUDED.content, content_text = EXCLUDED.content_text, embedding = EXCLUDED.embedding
		`)
	} else {
		stmt, err = tx.PrepareContext(ctx, `
			INSERT INTO memory_entries (app_name, user_id, session_id, event_id, author, content, content_text, timestamp)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (app_name, user_id, session_id, event_id) DO UPDATE 
			SET content = EXCLUDED.content, content_text = EXCLUDED.content_text
		`)
	}
	if err != nil {
		return fmt.Errorf("failed to prepare statement: %w", err)
	}
	defer stmt.Close()

	for event := range events.All() {
		if event.Content == nil || len(event.Content.Parts) == 0 {
			continue
		}

		// Extract text content
		text := extractTextFromContent(event.Content)
		if text == "" {
			continue
		}

		// Serialize content to JSON
		contentJSON, err := json.Marshal(event.Content)
		if err != nil {
			continue
		}

		timestamp := event.Timestamp
		if timestamp.IsZero() {
			timestamp = time.Now()
		}

		eventID := event.ID
		if eventID == "" {
			eventID = fmt.Sprintf("%s-%d", event.InvocationID, timestamp.UnixNano())
		}

		if s.embeddingModel != nil {
			// Generate embedding
			var embeddingStr *string
			embedding, err := s.embeddingModel.Embed(ctx, text)
			if err == nil && len(embedding) > 0 {
				embStr := vectorToString(embedding)
				embeddingStr = &embStr
			}

			_, err = stmt.ExecContext(ctx,
				sess.AppName(),
				sess.UserID(),
				sess.ID(),
				eventID,
				event.Author,
				contentJSON,
				text,
				embeddingStr,
				timestamp,
			)
		} else {
			_, err = stmt.ExecContext(ctx,
				sess.AppName(),
				sess.UserID(),
				sess.ID(),
				eventID,
				event.Author,
				contentJSON,
				text,
				timestamp,
			)
		}
		if err != nil {
			// Log but continue with other events
			continue
		}
	}

	return tx.Commit()
}

// Search finds relevant memory entries for a query.
func (s *PostgresMemoryService) Search(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	var memories []memory.Entry
	var err error

	// If we have an embedding model and a query, try vector search first
	if s.embeddingModel != nil && req.Query != "" {
		embedding, embErr := s.embeddingModel.Embed(ctx, req.Query)
		if embErr == nil && len(embedding) > 0 {
			memories, err = s.searchByVector(ctx, req, embedding)
			if err != nil {
				return nil, err
			}
		}
	}

	// Fallback to text search if no results or no embedding model
	if len(memories) == 0 && req.Query != "" {
		memories, err = s.searchByText(ctx, req)
		if err != nil {
			return nil, err
		}
	}

	// If still no results and query is empty, return recent entries
	if len(memories) == 0 {
		memories, err = s.searchRecent(ctx, req)
		if err != nil {
			return nil, err
		}
	}

	return &memory.SearchResponse{Memories: memories}, nil
}

// searchByVector performs semantic similarity search.
func (s *PostgresMemoryService) searchByVector(ctx context.Context, req *memory.SearchRequest, embedding []float32) ([]memory.Entry, error) {
	query := `
		SELECT content, author, timestamp
		FROM memory_entries
		WHERE app_name = $1 AND user_id = $2 AND embedding IS NOT NULL
		ORDER BY embedding <=> $3
		LIMIT 10
	`

	embeddingStr := vectorToString(embedding)
	rows, err := s.db.QueryContext(ctx, query, req.AppName, req.UserID, embeddingStr)
	if err != nil {
		return nil, fmt.Errorf("failed to search by vector: %w", err)
	}
	defer rows.Close()

	return s.scanMemories(rows)
}

// searchByText performs full-text search using PostgreSQL's tsvector.
func (s *PostgresMemoryService) searchByText(ctx context.Context, req *memory.SearchRequest) ([]memory.Entry, error) {
	query := `
		SELECT content, author, timestamp
		FROM memory_entries
		WHERE app_name = $1 AND user_id = $2
		AND to_tsvector('english', content_text) @@ plainto_tsquery('english', $3)
		ORDER BY ts_rank(to_tsvector('english', content_text), plainto_tsquery('english', $3)) DESC,
		         timestamp DESC
		LIMIT 10
	`

	rows, err := s.db.QueryContext(ctx, query, req.AppName, req.UserID, req.Query)
	if err != nil {
		return nil, fmt.Errorf("failed to search by text: %w", err)
	}
	defer rows.Close()

	return s.scanMemories(rows)
}

// searchRecent returns the most recent memory entries.
func (s *PostgresMemoryService) searchRecent(ctx context.Context, req *memory.SearchRequest) ([]memory.Entry, error) {
	query := `
		SELECT content, author, timestamp
		FROM memory_entries
		WHERE app_name = $1 AND user_id = $2
		ORDER BY timestamp DESC
		LIMIT 10
	`

	rows, err := s.db.QueryContext(ctx, query, req.AppName, req.UserID)
	if err != nil {
		return nil, fmt.Errorf("failed to search recent: %w", err)
	}
	defer rows.Close()

	return s.scanMemories(rows)
}

// scanMemories converts database rows to memory entries.
func (s *PostgresMemoryService) scanMemories(rows *sql.Rows) ([]memory.Entry, error) {
	var memories []memory.Entry

	for rows.Next() {
		var contentJSON []byte
		var author sql.NullString
		var timestamp time.Time

		if err := rows.Scan(&contentJSON, &author, &timestamp); err != nil {
			continue
		}

		var content genai.Content
		if err := json.Unmarshal(contentJSON, &content); err != nil {
			continue
		}

		entry := memory.Entry{
			Content:   &content,
			Timestamp: timestamp,
		}
		if author.Valid {
			entry.Author = author.String
		}

		memories = append(memories, entry)
	}

	return memories, rows.Err()
}

// Close closes the database connection.
func (s *PostgresMemoryService) Close() error {
	return s.db.Close()
}

// DB returns the underlying database connection for testing purposes.
func (s *PostgresMemoryService) DB() *sql.DB {
	return s.db
}

// extractTextFromContent extracts text from a genai.Content.
func extractTextFromContent(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var parts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

// vectorToString converts a float32 slice to PostgreSQL vector format.
func vectorToString(v []float32) string {
	if len(v) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, f := range v {
		if i > 0 {
			sb.WriteString(",")
		}
		fmt.Fprintf(&sb, "%f", f)
	}
	sb.WriteString("]")
	return sb.String()
}

// Ensure interface is implemented
var _ memory.Service = (*PostgresMemoryService)(nil)
