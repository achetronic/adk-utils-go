// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package memorytypes

import (
	"context"
	"time"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// EntryWithID represents a memory entry with its database row ID.
type EntryWithID struct {
	ID        int
	Content   *genai.Content
	Author    string
	Timestamp time.Time
}

// MemoryService defines the base interface for a memory backend.
type MemoryService interface {
	AddSessionToMemory(ctx context.Context, s session.Session) error
	SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error)
}

// ExtendedMemoryService extends MemoryService with update, delete, and ID-aware search.
type ExtendedMemoryService interface {
	MemoryService
	SearchWithID(ctx context.Context, req *memory.SearchRequest) ([]EntryWithID, error)
	UpdateMemory(ctx context.Context, appName, userID string, entryID int, newContent string) error
	DeleteMemory(ctx context.Context, appName, userID string, entryID int) error
}
