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

package filesystem

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"google.golang.org/adk/artifact"
	"google.golang.org/genai"
)

const userScopedArtifactKey = "user"

// FilesystemService implements artifact.Service using the local filesystem.
//
// Artifacts are stored as JSON files under:
//
//	{BasePath}/{appName}/{userID}/{sessionID}/{fileName}/{version}.json
//
// User-scoped artifacts (filenames prefixed with "user:") are stored under
// the "user" session key, making them accessible across all sessions for a
// given (appName, userID) pair.
type FilesystemService struct {
	basePath string
	mu       sync.RWMutex
}

// FilesystemServiceConfig holds configuration for FilesystemService.
type FilesystemServiceConfig struct {
	// BasePath is the root directory for artifact storage.
	BasePath string
}

// NewFilesystemService creates a new filesystem-backed artifact service.
// The base directory is created if it does not exist.
func NewFilesystemService(cfg FilesystemServiceConfig) (*FilesystemService, error) {
	if cfg.BasePath == "" {
		return nil, fmt.Errorf("BasePath is required")
	}

	if err := os.MkdirAll(cfg.BasePath, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create base directory: %w", err)
	}

	return &FilesystemService{
		basePath: cfg.BasePath,
	}, nil
}

func (s *FilesystemService) artifactDir(appName, userID, sessionID, fileName string) string {
	if fileHasUserNamespace(fileName) {
		sessionID = userScopedArtifactKey
	}
	return filepath.Join(s.basePath, appName, userID, sessionID, fileName)
}

func (s *FilesystemService) versionPath(appName, userID, sessionID, fileName string, version int64) string {
	return filepath.Join(s.artifactDir(appName, userID, sessionID, fileName), fmt.Sprintf("%d.json", version))
}

func (s *FilesystemService) sessionDir(appName, userID, sessionID string) string {
	return filepath.Join(s.basePath, appName, userID, sessionID)
}

func fileHasUserNamespace(filename string) bool {
	return strings.HasPrefix(filename, "user:")
}

// Save implements artifact.Service.
func (s *FilesystemService) Save(_ context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	dir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create artifact directory: %w", err)
	}

	nextVersion := int64(1)
	if latest, err := s.latestVersion(dir); err == nil {
		nextVersion = latest + 1
	}

	data, err := json.Marshal(req.Part)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal artifact: %w", err)
	}

	path := s.versionPath(req.AppName, req.UserID, req.SessionID, req.FileName, nextVersion)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return nil, fmt.Errorf("failed to write artifact: %w", err)
	}

	return &artifact.SaveResponse{Version: nextVersion}, nil
}

// Load implements artifact.Service.
func (s *FilesystemService) Load(_ context.Context, req *artifact.LoadRequest) (*artifact.LoadResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	version := req.Version
	if version <= 0 {
		dir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
		latest, err := s.latestVersion(dir)
		if err != nil {
			return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
		}
		version = latest
	}

	path := s.versionPath(req.AppName, req.UserID, req.SessionID, req.FileName, version)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
		}
		return nil, fmt.Errorf("failed to read artifact: %w", err)
	}

	var part genai.Part
	if err := json.Unmarshal(data, &part); err != nil {
		return nil, fmt.Errorf("failed to unmarshal artifact: %w", err)
	}

	return &artifact.LoadResponse{Part: &part}, nil
}

// Delete implements artifact.Service.
func (s *FilesystemService) Delete(_ context.Context, req *artifact.DeleteRequest) error {
	if err := req.Validate(); err != nil {
		return fmt.Errorf("request validation failed: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if req.Version != 0 {
		path := s.versionPath(req.AppName, req.UserID, req.SessionID, req.FileName, req.Version)
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to delete artifact version: %w", err)
		}
		s.cleanEmptyDirs(s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName))
		return nil
	}

	dir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err := os.RemoveAll(dir); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete artifact: %w", err)
	}

	return nil
}

// List implements artifact.Service.
func (s *FilesystemService) List(_ context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	files := make(map[string]bool)

	s.collectArtifactNames(s.sessionDir(req.AppName, req.UserID, req.SessionID), files)

	s.collectArtifactNames(s.sessionDir(req.AppName, req.UserID, userScopedArtifactKey), files)

	fileNames := make([]string, 0, len(files))
	for name := range files {
		fileNames = append(fileNames, name)
	}
	sort.Strings(fileNames)

	return &artifact.ListResponse{FileNames: fileNames}, nil
}

// Versions implements artifact.Service.
func (s *FilesystemService) Versions(_ context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	dir := s.artifactDir(req.AppName, req.UserID, req.SessionID, req.FileName)
	versions, err := s.listVersions(dir)
	if err != nil || len(versions) == 0 {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}

	return &artifact.VersionsResponse{Versions: versions}, nil
}

func (s *FilesystemService) latestVersion(dir string) (int64, error) {
	versions, err := s.listVersions(dir)
	if err != nil || len(versions) == 0 {
		return 0, fmt.Errorf("no versions found")
	}
	return versions[0], nil
}

func (s *FilesystemService) listVersions(dir string) ([]int64, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var versions []int64
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".json") {
			continue
		}
		numStr := strings.TrimSuffix(name, ".json")
		v, err := strconv.ParseInt(numStr, 10, 64)
		if err != nil {
			continue
		}
		versions = append(versions, v)
	}

	sort.Slice(versions, func(i, j int) bool {
		return versions[i] > versions[j]
	})

	return versions, nil
}

func (s *FilesystemService) collectArtifactNames(sessionDir string, files map[string]bool) {
	entries, err := os.ReadDir(sessionDir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		versionDir := filepath.Join(sessionDir, entry.Name())
		vEntries, err := os.ReadDir(versionDir)
		if err != nil {
			continue
		}
		for _, ve := range vEntries {
			if !ve.IsDir() && strings.HasSuffix(ve.Name(), ".json") {
				files[entry.Name()] = true
				break
			}
		}
	}
}

func (s *FilesystemService) cleanEmptyDirs(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	if len(entries) == 0 {
		os.Remove(dir)
	}
}

var _ artifact.Service = (*FilesystemService)(nil)
