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
	"testing"

	"google.golang.org/adk/artifact"
	"google.golang.org/genai"
)

func newTestService(t *testing.T) *FilesystemService {
	t.Helper()
	dir := t.TempDir()
	svc, err := NewFilesystemService(FilesystemServiceConfig{BasePath: dir})
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}
	return svc
}

func TestSaveAndLoadText(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	saveResp, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "hello.txt",
		Part:      genai.NewPartFromText("hello world"),
	})
	if err != nil {
		t.Fatalf("save failed: %v", err)
	}
	if saveResp.Version != 1 {
		t.Fatalf("expected version 1, got %d", saveResp.Version)
	}

	loadResp, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "hello.txt",
	})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if loadResp.Part.Text != "hello world" {
		t.Fatalf("expected 'hello world', got %q", loadResp.Part.Text)
	}
}

func TestSaveAndLoadBinary(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	binaryData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	part := &genai.Part{
		InlineData: &genai.Blob{
			MIMEType: "image/png",
			Data:     binaryData,
		},
	}

	_, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "image.png",
		Part:      part,
	})
	if err != nil {
		t.Fatalf("save failed: %v", err)
	}

	loadResp, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "image.png",
	})
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if loadResp.Part.InlineData == nil {
		t.Fatal("expected InlineData, got nil")
	}
	if loadResp.Part.InlineData.MIMEType != "image/png" {
		t.Fatalf("expected image/png, got %s", loadResp.Part.InlineData.MIMEType)
	}
	if len(loadResp.Part.InlineData.Data) != len(binaryData) {
		t.Fatalf("data length mismatch: expected %d, got %d", len(binaryData), len(loadResp.Part.InlineData.Data))
	}
}

func TestVersioning(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	for i := 1; i <= 3; i++ {
		resp, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			FileName:  "doc.txt",
			Part:      genai.NewPartFromText("version " + string(rune('0'+i))),
		})
		if err != nil {
			t.Fatalf("save %d failed: %v", i, err)
		}
		if resp.Version != int64(i) {
			t.Fatalf("expected version %d, got %d", i, resp.Version)
		}
	}

	loadResp, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "doc.txt",
	})
	if err != nil {
		t.Fatalf("load latest failed: %v", err)
	}
	if loadResp.Part.Text != "version 3" {
		t.Fatalf("expected 'version 3', got %q", loadResp.Part.Text)
	}

	loadResp, err = svc.Load(ctx, &artifact.LoadRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "doc.txt",
		Version:   1,
	})
	if err != nil {
		t.Fatalf("load version 1 failed: %v", err)
	}
	if loadResp.Part.Text != "version 1" {
		t.Fatalf("expected 'version 1', got %q", loadResp.Part.Text)
	}
}

func TestVersionsList(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	for i := 1; i <= 3; i++ {
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			FileName:  "doc.txt",
			Part:      genai.NewPartFromText("v"),
		})
		if err != nil {
			t.Fatalf("save failed: %v", err)
		}
	}

	versResp, err := svc.Versions(ctx, &artifact.VersionsRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "doc.txt",
	})
	if err != nil {
		t.Fatalf("versions failed: %v", err)
	}
	if len(versResp.Versions) != 3 {
		t.Fatalf("expected 3 versions, got %d", len(versResp.Versions))
	}
	if versResp.Versions[0] != 3 {
		t.Fatalf("expected newest first (3), got %d", versResp.Versions[0])
	}
}

func TestList(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	for _, name := range []string{"b.txt", "a.txt", "c.txt"} {
		_, err := svc.Save(ctx, &artifact.SaveRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			FileName:  name,
			Part:      genai.NewPartFromText("data"),
		})
		if err != nil {
			t.Fatalf("save %s failed: %v", name, err)
		}
	}

	listResp, err := svc.List(ctx, &artifact.ListRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
	})
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(listResp.FileNames) != 3 {
		t.Fatalf("expected 3 files, got %d", len(listResp.FileNames))
	}
	if listResp.FileNames[0] != "a.txt" {
		t.Fatalf("expected sorted, first should be 'a.txt', got %q", listResp.FileNames[0])
	}
}

func TestDeleteAllVersions(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		_, _ = svc.Save(ctx, &artifact.SaveRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			FileName:  "doc.txt",
			Part:      genai.NewPartFromText("v"),
		})
	}

	err := svc.Delete(ctx, &artifact.DeleteRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "doc.txt",
	})
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	_, err = svc.Load(ctx, &artifact.LoadRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "doc.txt",
	})
	if err == nil {
		t.Fatal("expected error loading deleted artifact")
	}
}

func TestDeleteSingleVersion(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		_, _ = svc.Save(ctx, &artifact.SaveRequest{
			AppName:   "app1",
			UserID:    "user1",
			SessionID: "sess1",
			FileName:  "doc.txt",
			Part:      genai.NewPartFromText("v"),
		})
	}

	err := svc.Delete(ctx, &artifact.DeleteRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "doc.txt",
		Version:   2,
	})
	if err != nil {
		t.Fatalf("delete version 2 failed: %v", err)
	}

	versResp, err := svc.Versions(ctx, &artifact.VersionsRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "doc.txt",
	})
	if err != nil {
		t.Fatalf("versions failed: %v", err)
	}
	if len(versResp.Versions) != 2 {
		t.Fatalf("expected 2 versions, got %d", len(versResp.Versions))
	}
}

func TestUserScopedArtifacts(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	_, err := svc.Save(ctx, &artifact.SaveRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "user:prefs.json",
		Part:      genai.NewPartFromText(`{"theme":"dark"}`),
	})
	if err != nil {
		t.Fatalf("save user-scoped failed: %v", err)
	}

	listResp, err := svc.List(ctx, &artifact.ListRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
	})
	if err != nil {
		t.Fatalf("list from sess1 failed: %v", err)
	}
	if len(listResp.FileNames) != 1 || listResp.FileNames[0] != "user:prefs.json" {
		t.Fatalf("expected user:prefs.json in sess1 list, got %v", listResp.FileNames)
	}

	listResp2, err := svc.List(ctx, &artifact.ListRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess2",
	})
	if err != nil {
		t.Fatalf("list from sess2 failed: %v", err)
	}
	if len(listResp2.FileNames) != 1 || listResp2.FileNames[0] != "user:prefs.json" {
		t.Fatalf("expected user:prefs.json visible from sess2, got %v", listResp2.FileNames)
	}
}

func TestLoadNotFound(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	_, err := svc.Load(ctx, &artifact.LoadRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "nonexistent.txt",
	})
	if err == nil {
		t.Fatal("expected error for nonexistent artifact")
	}
}

func TestVersionsNotFound(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	_, err := svc.Versions(ctx, &artifact.VersionsRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "nonexistent.txt",
	})
	if err == nil {
		t.Fatal("expected error for nonexistent artifact versions")
	}
}

func TestDeleteNonexistent(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	err := svc.Delete(ctx, &artifact.DeleteRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "nonexistent.txt",
	})
	if err != nil {
		t.Fatalf("deleting nonexistent should not error, got: %v", err)
	}
}

func TestValidationErrors(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	_, err := svc.Save(ctx, &artifact.SaveRequest{
		FileName: "test.txt",
		Part:     genai.NewPartFromText("data"),
	})
	if err == nil {
		t.Fatal("expected validation error for missing fields")
	}

	_, err = svc.Save(ctx, &artifact.SaveRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "test.txt",
	})
	if err == nil {
		t.Fatal("expected validation error for nil Part")
	}
}

func TestSessionIsolation(t *testing.T) {
	svc := newTestService(t)
	ctx := context.Background()

	_, _ = svc.Save(ctx, &artifact.SaveRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess1",
		FileName:  "private.txt",
		Part:      genai.NewPartFromText("secret"),
	})

	listResp, err := svc.List(ctx, &artifact.ListRequest{
		AppName:   "app1",
		UserID:    "user1",
		SessionID: "sess2",
	})
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	for _, name := range listResp.FileNames {
		if name == "private.txt" {
			t.Fatal("sess2 should not see sess1's artifacts")
		}
	}
}
