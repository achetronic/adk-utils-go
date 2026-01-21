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

package memory

import (
	"context"
	"fmt"
	"iter"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

// MemoryService defines the interface for a memory backend.
// This matches google.golang.org/adk/memory.Service
type MemoryService interface {
	AddSession(ctx context.Context, s session.Session) error
	Search(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error)
}

// Toolset provides tools for the agent to interact with long-term memory.
type Toolset struct {
	memoryService MemoryService
	appName       string
	tools         []tool.Tool
}

// ToolsetConfig holds configuration for the memory toolset.
type ToolsetConfig struct {
	// MemoryService is the memory service to use (can be any implementation)
	MemoryService MemoryService
	// AppName is used to scope memory operations
	AppName string
}

// NewToolset creates a new toolset for memory operations.
func NewToolset(cfg ToolsetConfig) (*Toolset, error) {
	if cfg.MemoryService == nil {
		return nil, fmt.Errorf("MemoryService is required")
	}
	if cfg.AppName == "" {
		return nil, fmt.Errorf("AppName is required")
	}

	ts := &Toolset{
		memoryService: cfg.MemoryService,
		appName:       cfg.AppName,
	}

	// Create search tool
	searchTool, err := functiontool.New(
		functiontool.Config{
			Name:        "search_memory",
			Description: "Search long-term memory for relevant information from past conversations. Use this to recall facts, preferences, or context from previous interactions with the user.",
		},
		ts.searchMemory,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create search_memory tool: %w", err)
	}

	// Create save tool
	saveTool, err := functiontool.New(
		functiontool.Config{
			Name:        "save_to_memory",
			Description: "Save important information to long-term memory for future recall. Use this to remember user preferences, important facts, or anything the user explicitly asks you to remember.",
		},
		ts.saveToMemory,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create save_to_memory tool: %w", err)
	}

	ts.tools = []tool.Tool{searchTool, saveTool}

	return ts, nil
}

// Name returns the name of the toolset.
func (ts *Toolset) Name() string {
	return "memory_toolset"
}

// Tools returns the list of memory tools.
func (ts *Toolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	return ts.tools, nil
}

// SearchArgs are the arguments for the search_memory tool.
type SearchArgs struct {
	// Query is the search query to find relevant memories
	Query string `json:"query"`
}

// SearchResult is the result of the search_memory tool.
type SearchResult struct {
	// Memories contains the found memories
	Memories []Entry `json:"memories"`
	// Count is the number of memories found
	Count int `json:"count"`
}

// Entry represents a single memory entry returned by search.
type Entry struct {
	// Text is the content of the memory
	Text string `json:"text"`
	// Author is who created this memory (user or agent)
	Author string `json:"author"`
	// Timestamp is when this memory was created
	Timestamp string `json:"timestamp"`
}

// searchMemory searches the long-term memory.
func (ts *Toolset) searchMemory(ctx tool.Context, args SearchArgs) (SearchResult, error) {
	if args.Query == "" {
		return SearchResult{}, fmt.Errorf("query cannot be empty")
	}

	userID := ctx.UserID()

	resp, err := ts.memoryService.Search(ctx, &memory.SearchRequest{
		AppName: ts.appName,
		UserID:  userID,
		Query:   args.Query,
	})
	if err != nil {
		return SearchResult{}, fmt.Errorf("failed to search memory: %w", err)
	}

	var entries []Entry
	for _, mem := range resp.Memories {
		text := ""
		if mem.Content != nil && len(mem.Content.Parts) > 0 {
			text = mem.Content.Parts[0].Text
		}
		entries = append(entries, Entry{
			Text:      text,
			Author:    mem.Author,
			Timestamp: mem.Timestamp.Format("2006-01-02 15:04:05"),
		})
	}

	return SearchResult{
		Memories: entries,
		Count:    len(entries),
	}, nil
}

// SaveArgs are the arguments for the save_to_memory tool.
type SaveArgs struct {
	// Content is the information to save to memory
	Content string `json:"content"`
	// Category is an optional category for the memory (e.g., 'preference', 'fact', 'reminder')
	Category string `json:"category,omitempty"`
}

// SaveResult is the result of the save_to_memory tool.
type SaveResult struct {
	// Success indicates if the save was successful
	Success bool `json:"success"`
	// Message provides additional information
	Message string `json:"message"`
}

// saveToMemory saves information to long-term memory.
func (ts *Toolset) saveToMemory(ctx tool.Context, args SaveArgs) (SaveResult, error) {
	if args.Content == "" {
		return SaveResult{
			Success: false,
			Message: "content cannot be empty",
		}, nil
	}

	userID := ctx.UserID()

	// Create a minimal session with just this memory entry
	memorySession := &singleEntrySession{
		id:       fmt.Sprintf("memory-%d", time.Now().UnixNano()),
		appName:  ts.appName,
		userID:   userID,
		content:  args.Content,
		category: args.Category,
	}

	err := ts.memoryService.AddSession(ctx, memorySession)
	if err != nil {
		return SaveResult{
			Success: false,
			Message: fmt.Sprintf("failed to save: %v", err),
		}, nil
	}

	return SaveResult{
		Success: true,
		Message: "Memory saved successfully",
	}, nil
}

// Ensure interface is implemented
var _ tool.Toolset = (*Toolset)(nil)

// singleEntrySession is a minimal session implementation for saving individual memories.
type singleEntrySession struct {
	id       string
	appName  string
	userID   string
	content  string
	category string
}

func (s *singleEntrySession) ID() string                { return s.id }
func (s *singleEntrySession) AppName() string           { return s.appName }
func (s *singleEntrySession) UserID() string            { return s.userID }
func (s *singleEntrySession) State() session.State      { return nil }
func (s *singleEntrySession) LastUpdateTime() time.Time { return time.Now() }

func (s *singleEntrySession) Events() session.Events {
	return &singleEntryEvents{
		content:  s.content,
		category: s.category,
	}
}

// singleEntryEvents provides a single event containing the memory content.
type singleEntryEvents struct {
	content  string
	category string
}

func (e *singleEntryEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		yield(e.createEvent())
	}
}

func (e *singleEntryEvents) Len() int {
	return 1
}

func (e *singleEntryEvents) At(i int) *session.Event {
	if i != 0 {
		return nil
	}
	return e.createEvent()
}

func (e *singleEntryEvents) createEvent() *session.Event {
	text := e.content
	if e.category != "" {
		text = "[" + e.category + "] " + text
	}
	return &session.Event{
		ID:        fmt.Sprintf("memory-entry-%d", time.Now().UnixNano()),
		Author:    "agent",
		Timestamp: time.Now(),
		LLMResponse: model.LLMResponse{
			Content: &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(text)},
				Role:  "assistant",
			},
		},
	}
}
