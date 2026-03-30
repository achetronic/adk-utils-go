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

package memorytypes

import (
	"context"
	"time"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
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
