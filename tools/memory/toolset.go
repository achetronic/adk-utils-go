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
	"fmt"
	"iter"
	"time"

	"github.com/achetronic/adk-utils-go/memory/memorytypes"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"
)

// Toolset provides tools for the agent to interact with long-term memory.
type Toolset struct {
	memoryService    memorytypes.MemoryService
	extMemoryService memorytypes.ExtendedMemoryService
	appName          string
	tools            []tool.Tool
}

// ToolsetConfig holds configuration for the memory toolset.
type ToolsetConfig struct {
	// MemoryService is the memory service to use (can be any implementation).
	// If it also implements memorytypes.ExtendedMemoryService, update and delete tools will be available.
	MemoryService memorytypes.MemoryService
	// AppName is used to scope memory operations
	AppName string
	// DisableExtendedTools prevents registration of update_memory and delete_memory
	// even when the MemoryService supports them.
	DisableExtendedTools bool
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

	if !cfg.DisableExtendedTools {
		if ext, ok := cfg.MemoryService.(memorytypes.ExtendedMemoryService); ok {
			ts.extMemoryService = ext
		}
	}

	searchTool, err := functiontool.New(
		functiontool.Config{
			Name:        "search_memory",
			Description: "Search long-term memory for relevant information from past conversations. Use this to recall facts, preferences, or context from previous interactions with the user. Results include an 'id' field that can be used with update_memory and delete_memory.",
		},
		ts.searchMemory,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create search_memory tool: %w", err)
	}

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

	if ts.extMemoryService != nil {
		updateTool, err := functiontool.New(
			functiontool.Config{
				Name:        "update_memory",
				Description: "Update the content of an existing memory entry. Use this to correct outdated information or refine a previously saved memory. Requires the memory entry ID from search_memory results.",
			},
			ts.updateMemory,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create update_memory tool: %w", err)
		}

		deleteTool, err := functiontool.New(
			functiontool.Config{
				Name:        "delete_memory",
				Description: "Delete a memory entry permanently. Use this to remove incorrect, irrelevant, or outdated information from long-term memory. Requires the memory entry ID from search_memory results.",
			},
			ts.deleteMemory,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create delete_memory tool: %w", err)
		}

		ts.tools = append(ts.tools, updateTool, deleteTool)
	}

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
	Query string `json:"query" jsonschema:"Natural language query describing what information to look for in memory. Be specific and descriptive for best results."`
}

// SearchResult is the result of the search_memory tool.
type SearchResult struct {
	Memories []Entry `json:"memories"`
	Count    int     `json:"count"`
}

// Entry represents a single memory entry returned by search.
type Entry struct {
	ID        int    `json:"id"`
	Text      string `json:"text"`
	Author    string `json:"author"`
	Timestamp string `json:"timestamp"`
}

// searchMemory searches the long-term memory.
func (ts *Toolset) searchMemory(ctx tool.Context, args SearchArgs) (SearchResult, error) {
	if args.Query == "" {
		return SearchResult{}, fmt.Errorf("query cannot be empty")
	}

	userID := ctx.UserID()
	req := &memory.SearchRequest{
		AppName: ts.appName,
		UserID:  userID,
		Query:   args.Query,
	}

	if ts.extMemoryService != nil {
		results, err := ts.extMemoryService.SearchWithID(ctx, req)
		if err != nil {
			return SearchResult{}, fmt.Errorf("failed to search memory: %w", err)
		}

		entries := []Entry{}
		for _, mem := range results {
			text := ""
			if mem.Content != nil && len(mem.Content.Parts) > 0 {
				text = mem.Content.Parts[0].Text
			}
			entries = append(entries, Entry{
				ID:        mem.ID,
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

	resp, err := ts.memoryService.Search(ctx, req)
	if err != nil {
		return SearchResult{}, fmt.Errorf("failed to search memory: %w", err)
	}

	entries := []Entry{}
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
	Content  string `json:"content" jsonschema:"The information to save to memory. Write it as a clear, self-contained statement that will be meaningful when retrieved later."`
	Category string `json:"category,omitempty" jsonschema:"Optional label to categorize the memory (e.g. 'preference', 'fact', 'task'). Helps organize and retrieve related memories."`
}

// SaveResult is the result of the save_to_memory tool.
type SaveResult struct {
	Success bool   `json:"success"`
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

// UpdateArgs are the arguments for the update_memory tool.
type UpdateArgs struct {
	ID      int    `json:"id" jsonschema:"Numeric ID of the memory entry to update, obtained from the id field in search_memory results."`
	Content string `json:"content" jsonschema:"The new content to replace the existing memory entry. Write it as a clear, self-contained statement."`
}

// UpdateResult is the result of the update_memory tool.
type UpdateResult struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// updateMemory updates an existing memory entry.
func (ts *Toolset) updateMemory(ctx tool.Context, args UpdateArgs) (UpdateResult, error) {
	if args.ID == 0 {
		return UpdateResult{
			Success: false,
			Message: "id is required",
		}, nil
	}
	if args.Content == "" {
		return UpdateResult{
			Success: false,
			Message: "content cannot be empty",
		}, nil
	}

	err := ts.extMemoryService.UpdateMemory(ctx, ts.appName, ctx.UserID(), args.ID, args.Content)
	if err != nil {
		return UpdateResult{
			Success: false,
			Message: fmt.Sprintf("failed to update: %v", err),
		}, nil
	}

	return UpdateResult{
		Success: true,
		Message: "Memory updated successfully",
	}, nil
}

// DeleteArgs are the arguments for the delete_memory tool.
type DeleteArgs struct {
	ID int `json:"id" jsonschema:"Numeric ID of the memory entry to delete, obtained from the id field in search_memory results."`
}

// DeleteResult is the result of the delete_memory tool.
type DeleteResult struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// deleteMemory deletes a memory entry.
func (ts *Toolset) deleteMemory(ctx tool.Context, args DeleteArgs) (DeleteResult, error) {
	if args.ID == 0 {
		return DeleteResult{
			Success: false,
			Message: "id is required",
		}, nil
	}

	err := ts.extMemoryService.DeleteMemory(ctx, ts.appName, ctx.UserID(), args.ID)
	if err != nil {
		return DeleteResult{
			Success: false,
			Message: fmt.Sprintf("failed to delete: %v", err),
		}, nil
	}

	return DeleteResult{
		Success: true,
		Message: "Memory deleted successfully",
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
