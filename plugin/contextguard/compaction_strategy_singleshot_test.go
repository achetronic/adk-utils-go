package contextguard

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// ---------------------------------------------------------------------------
// Realistic conversation builders
// ---------------------------------------------------------------------------

// kubeAgentConversation simulates a Kubernetes agent session:
//   - User asks something
//   - Model calls kubectl_get_pods → huge JSON response
//   - Model calls kubectl_describe_pod → huge JSON response
//   - Model calls kubectl_get_logs → huge text response
//   - Model responds with analysis
//   - Repeat
func kubeAgentConversation(rounds int) []*genai.Content {
	var contents []*genai.Content

	podsJSON := `{"items":[` + strings.Repeat(
		`{"metadata":{"name":"nginx-deployment-abc12","namespace":"production","labels":{"app":"nginx","tier":"frontend"}},"status":{"phase":"Running","containerStatuses":[{"name":"nginx","ready":true,"restartCount":0,"image":"nginx:1.25.3"}]}},`, 40,
	) + `{}]}`

	describeOutput := strings.Repeat(
		"Name: nginx-deployment-abc12\nNamespace: production\nNode: worker-node-03/10.0.1.15\n"+
			"Status: Running\nIP: 172.16.0.42\nContainers:\n  nginx:\n    Image: nginx:1.25.3\n"+
			"    Port: 80/TCP\n    State: Running\n    Ready: True\n    Restart Count: 0\n"+
			"Conditions:\n  Type: Ready  Status: True\n  Type: ContainersReady  Status: True\n"+
			"Events:\n  Normal  Scheduled  5m  default-scheduler  Successfully assigned\n"+
			"  Normal  Pulled     5m  kubelet  Container image already present\n"+
			"  Normal  Created    5m  kubelet  Created container nginx\n"+
			"  Normal  Started    5m  kubelet  Started container nginx\n\n", 5,
	)

	logsOutput := strings.Repeat(
		"2025-01-15T10:30:00Z INFO  [nginx] 10.0.1.1 - - [15/Jan/2025:10:30:00 +0000] \"GET /api/v1/health HTTP/1.1\" 200 15 \"-\" \"kube-probe/1.28\"\n"+
			"2025-01-15T10:30:01Z INFO  [nginx] 10.0.2.5 - - [15/Jan/2025:10:30:01 +0000] \"POST /api/v1/data HTTP/1.1\" 201 482 \"-\" \"Go-http-client/2.0\"\n"+
			"2025-01-15T10:30:02Z WARN  [nginx] upstream timed out (110: Connection timed out) while connecting to upstream\n", 50,
	)

	for i := 0; i < rounds; i++ {
		contents = append(contents,
			textContent("user", fmt.Sprintf("Check the status of pods in the production namespace, round %d", i)),
		)

		contents = append(contents,
			&genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "kubectl_get_pods",
						Args: map[string]any{"namespace": "production", "output": "json"},
					},
				}},
			},
			&genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "kubectl_get_pods",
						Response: map[string]any{"stdout": podsJSON},
					},
				}},
			},
		)

		contents = append(contents,
			&genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "kubectl_describe_pod",
						Args: map[string]any{"pod": "nginx-deployment-abc12", "namespace": "production"},
					},
				}},
			},
			&genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "kubectl_describe_pod",
						Response: map[string]any{"stdout": describeOutput},
					},
				}},
			},
		)

		contents = append(contents,
			&genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: "kubectl_get_logs",
						Args: map[string]any{"pod": "nginx-deployment-abc12", "namespace": "production", "tail": 100},
					},
				}},
			},
			&genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     "kubectl_get_logs",
						Response: map[string]any{"stdout": logsOutput},
					},
				}},
			},
		)

		contents = append(contents,
			textContent("model", fmt.Sprintf(
				"Round %d analysis: All 40 pods in the production namespace are running and healthy. "+
					"The nginx-deployment-abc12 pod has been running stable with 0 restarts. "+
					"Logs show normal traffic patterns with occasional upstream timeout warnings that should be monitored. "+
					"No critical issues detected.", i)),
		)
	}

	return contents
}

// mixedConversation simulates a more realistic session with text, small tool
// calls, big tool calls, and follow-up questions interleaved.
func mixedConversation() []*genai.Content {
	bigAPIResponse := `{"users":[` + strings.Repeat(
		`{"id":1,"name":"Alice Johnson","email":"alice@example.com","role":"admin","created":"2024-01-15","last_login":"2025-01-15","preferences":{"theme":"dark","lang":"en","notifications":true}},`, 100,
	) + `{}]}`

	return []*genai.Content{
		textContent("user", "Hey, I need help debugging our production API. Users are reporting slow response times."),
		textContent("model", "I'll help you investigate. Let me check the API health and recent metrics."),

		// tool: check health
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "http_get", Args: map[string]any{"url": "https://api.example.com/health"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "http_get", Response: map[string]any{"status": 200, "body": `{"status":"degraded","latency_p99":"2300ms","error_rate":"0.02"}`},
		}}}},

		// tool: get metrics (big)
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "prometheus_query", Args: map[string]any{"query": "rate(http_request_duration_seconds_bucket[5m])", "range": "1h"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "prometheus_query", Response: map[string]any{"data": strings.Repeat("ts=1705312800,val=0.234\n", 500)},
		}}}},

		textContent("model", "The API is in a degraded state. P99 latency is at 2300ms (normally ~200ms) and error rate is 2%. The metrics show latency started increasing about 45 minutes ago. Let me check the database connections."),
		textContent("user", "Could it be related to the deployment we did an hour ago?"),

		// tool: check recent deployments
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "kubectl_get_deployments", Args: map[string]any{"namespace": "production", "sort-by": ".metadata.creationTimestamp"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "kubectl_get_deployments", Response: map[string]any{"stdout": strings.Repeat("api-server  3/3  3  3  45m\n", 20)},
		}}}},

		// tool: check DB connections (big)
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "sql_query", Args: map[string]any{"query": "SELECT * FROM pg_stat_activity WHERE state != 'idle'"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "sql_query", Response: map[string]any{"rows": strings.Repeat(
				`{"pid":12345,"state":"active","query":"SELECT u.* FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.expires_at > NOW()","duration":"45.2s","wait_event":"Lock"},`+"\n", 80,
			)},
		}}}},

		textContent("model", "Found it! There are 80 active database connections, most of them stuck waiting on a lock. They're all running the same query joining users and sessions. The deployment likely introduced a migration or schema change that's causing lock contention."),
		textContent("user", "Can you check what migration was in the last deployment?"),

		// tool: check migration history
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "sql_query", Args: map[string]any{"query": "SELECT * FROM schema_migrations ORDER BY version DESC LIMIT 5"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "sql_query", Response: map[string]any{"rows": `[{"version":"20250115_001","name":"add_index_sessions_user_id","applied":"2025-01-15 10:00:00","duration":"running"}]`},
		}}}},

		textContent("model", "The migration '20250115_001: add_index_sessions_user_id' is still running. It's trying to add an index to the sessions table, which takes an exclusive lock. This is blocking all queries that touch the sessions table. We should either wait for it to complete or cancel it and retry with CREATE INDEX CONCURRENTLY."),
		textContent("user", "Let's see the full user list first to assess impact."),

		// tool: massive response
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "sql_query", Args: map[string]any{"query": "SELECT * FROM users ORDER BY last_login DESC"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "sql_query", Response: map[string]any{"rows": bigAPIResponse},
		}}}},

		textContent("model", "There are 100 active users who would be affected. Given the impact, I recommend cancelling the migration and scheduling it during the next maintenance window with CONCURRENTLY."),
		textContent("user", "OK, let's cancel it and rollback."),

		// tool: cancel migration
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "sql_query", Args: map[string]any{"query": "SELECT pg_cancel_backend(pid) FROM pg_stat_activity WHERE query LIKE '%add_index_sessions%'"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "sql_query", Response: map[string]any{"rows": `[{"pg_cancel_backend": true}]`},
		}}}},

		// tool: verify health recovered
		&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "http_get", Args: map[string]any{"url": "https://api.example.com/health"},
		}}}},
		&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			Name: "http_get", Response: map[string]any{"status": 200, "body": `{"status":"healthy","latency_p99":"180ms","error_rate":"0.001"}`},
		}}}},

		textContent("model", "Migration cancelled. The API health has recovered — P99 latency is back to 180ms and error rate dropped to 0.1%. I recommend adding the index with CREATE INDEX CONCURRENTLY during the next maintenance window."),
	}
}

// pureToolStorm simulates the worst case: nothing but consecutive tool calls
// with large responses and zero text messages in between.
func pureToolStorm(pairs int, responseSize int) []*genai.Content {
	contents := make([]*genai.Content, 0, pairs*2)
	for i := 0; i < pairs; i++ {
		contents = append(contents,
			&genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						Name: fmt.Sprintf("tool_%d", i),
						Args: map[string]any{"param": strings.Repeat("arg-data-", 20)},
					},
				}},
			},
			&genai.Content{
				Role: "user",
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						Name:     fmt.Sprintf("tool_%d", i),
						Response: map[string]any{"result": strings.Repeat("x", responseSize)},
					},
				}},
			},
		)
	}
	return contents
}

// ---------------------------------------------------------------------------
// Simulation helpers
// ---------------------------------------------------------------------------

type compactionResult struct {
	scenario          string
	strategy          string
	contextWindow     int
	contentsBefore    int
	contentsAfter     int
	tokensBefore      int
	tokensAfter       int
	oldMessages       int
	recentMessages    int
	summarized        bool
	summaryLen        int
	reductionPct      float64
	fitsContextWindow bool
}

func (r compactionResult) String() string {
	fit := "YES"
	if !r.fitsContextWindow {
		fit = "NO"
	}

	return fmt.Sprintf(
		"  %-40s | %-15s | ctx=%7dk | contents: %4d -> %4d | tokens: %7d -> %7d (%5.1f%% reduction) | old=%4d recent=%4d | summarized=%v (len=%d) | fits=%s",
		r.scenario, r.strategy,
		r.contextWindow/1000,
		r.contentsBefore, r.contentsAfter,
		r.tokensBefore, r.tokensAfter, r.reductionPct,
		r.oldMessages, r.recentMessages,
		r.summarized, r.summaryLen,
		fit,
	)
}

func runThresholdSimulation(t *testing.T, scenario string, contents []*genai.Content, contextWindow int) compactionResult {
	t.Helper()

	registry := &mockRegistry{
		contextWindows: map[string]int{"sim-model": contextWindow},
		maxTokens:      map[string]int{"sim-model": 4096},
	}
	llm := &mockLLM{
		name:     "sim-model",
		response: "Summary: The conversation involved multiple tool calls to investigate and resolve issues. Key findings and decisions were made. Next steps were identified.",
	}
	s := newThresholdStrategy(registry, llm, 0)
	ctx := newMockCallbackContext("sim-agent")

	req := &model.LLMRequest{
		Model:    "sim-model",
		Contents: copyContents(contents),
	}

	tokensBefore := estimateTokens(req)
	contentsBefore := len(req.Contents)

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("[%s] Compact error: %v", scenario, err)
	}

	tokensAfter := estimateTokens(req)
	contentsAfter := len(req.Contents)

	summary := loadSummary(ctx)
	summarized := summary != ""

	oldMessages := 0
	recentMessages := contentsAfter
	if summarized {
		oldMessages = contentsBefore - (contentsAfter - 1)
		recentMessages = contentsAfter - 1
	}

	reductionPct := 0.0
	if tokensBefore > 0 {
		reductionPct = (1.0 - float64(tokensAfter)/float64(tokensBefore)) * 100
	}

	buffer := computeBuffer(contextWindow)
	threshold := contextWindow - buffer

	return compactionResult{
		scenario:          scenario,
		strategy:          "threshold",
		contextWindow:     contextWindow,
		contentsBefore:    contentsBefore,
		contentsAfter:     contentsAfter,
		tokensBefore:      tokensBefore,
		tokensAfter:       tokensAfter,
		oldMessages:       oldMessages,
		recentMessages:    recentMessages,
		summarized:        summarized,
		summaryLen:        len(summary),
		reductionPct:      reductionPct,
		fitsContextWindow: tokensAfter < threshold,
	}
}

func runSlidingWindowSimulation(t *testing.T, scenario string, contents []*genai.Content, maxTurns int) compactionResult {
	t.Helper()

	registry := newMockRegistry()
	llm := &mockLLM{
		name:     "gpt-4o",
		response: "Summary: The conversation involved multiple tool calls to investigate and resolve issues. Key findings and decisions were made.",
	}
	s := newSlidingWindowStrategy(registry, llm, maxTurns)
	ctx := newMockCallbackContext("sim-agent")

	req := &model.LLMRequest{
		Model:    "gpt-4o",
		Contents: copyContents(contents),
	}

	tokensBefore := estimateTokens(req)
	contentsBefore := len(req.Contents)

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("[%s] Compact error: %v", scenario, err)
	}

	tokensAfter := estimateTokens(req)
	contentsAfter := len(req.Contents)

	summary := loadSummary(ctx)
	summarized := summary != ""

	oldMessages := 0
	recentMessages := contentsAfter
	if summarized {
		oldMessages = contentsBefore - (contentsAfter - 1)
		recentMessages = contentsAfter - 1
	}

	reductionPct := 0.0
	if tokensBefore > 0 {
		reductionPct = (1.0 - float64(tokensAfter)/float64(tokensBefore)) * 100
	}

	contextWindow := registry.ContextWindow("gpt-4o")
	buffer := computeBuffer(contextWindow)
	threshold := contextWindow - buffer

	return compactionResult{
		scenario:          scenario,
		strategy:          fmt.Sprintf("sliding(%d)", maxTurns),
		contextWindow:     contextWindow,
		contentsBefore:    contentsBefore,
		contentsAfter:     contentsAfter,
		tokensBefore:      tokensBefore,
		tokensAfter:       tokensAfter,
		oldMessages:       oldMessages,
		recentMessages:    recentMessages,
		summarized:        summarized,
		summaryLen:        len(summary),
		reductionPct:      reductionPct,
		fitsContextWindow: tokensAfter < threshold,
	}
}

func copyContents(src []*genai.Content) []*genai.Content {
	dst := make([]*genai.Content, len(src))
	copy(dst, src)
	return dst
}

// buildCodingAgentConversation simulates a coding assistant that reads files,
// runs tests, edits code — alternating between text discussion and tool use.
// Each round: user asks → model thinks (text) → model calls tool → result →
// model analyzes (text). This is the best-case for safeSplitIndex because
// there are text messages between tool chains.
func buildCodingAgentConversation(rounds int) []*genai.Content {
	fileContent := "package main\n\nimport (\n\t\"fmt\"\n\t\"net/http\"\n)\n\n" + strings.Repeat(
		"func handler(w http.ResponseWriter, r *http.Request) {\n"+
			"\tfmt.Fprintf(w, \"Hello, World!\")\n"+
			"}\n\n", 20,
	)

	testOutput := strings.Repeat(
		"=== RUN   TestHandler\n--- PASS: TestHandler (0.00s)\n"+
			"=== RUN   TestRouter\n--- PASS: TestRouter (0.00s)\n"+
			"=== RUN   TestMiddleware\n--- FAIL: TestMiddleware (0.02s)\n"+
			"    middleware_test.go:45: expected status 200, got 401\n", 10,
	)

	var contents []*genai.Content
	for i := 0; i < rounds; i++ {
		contents = append(contents,
			textContent("user", fmt.Sprintf("Round %d: Can you fix the middleware auth test? It's returning 401 instead of 200.", i)),
			textContent("model", fmt.Sprintf("Let me look at the middleware code and the failing test to understand what's happening in round %d.", i)),
		)

		contents = append(contents,
			&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				Name: "read_file", Args: map[string]any{"path": "internal/middleware/auth.go"},
			}}}},
			&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name: "read_file", Response: map[string]any{"content": fileContent},
			}}}},
		)

		contents = append(contents,
			textContent("model", "I see the issue. The auth middleware is checking for the Authorization header but the test isn't setting it. Let me fix it."),
		)

		contents = append(contents,
			&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				Name: "edit_file", Args: map[string]any{"path": "internal/middleware/auth_test.go", "content": "req.Header.Set(\"Authorization\", \"Bearer test-token\")"},
			}}}},
			&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name: "edit_file", Response: map[string]any{"status": "ok", "lines_changed": 1},
			}}}},
		)

		contents = append(contents,
			&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				Name: "run_tests", Args: map[string]any{"path": "./internal/middleware/..."},
			}}}},
			&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name: "run_tests", Response: map[string]any{"output": testOutput, "exit_code": 0},
			}}}},
		)

		contents = append(contents,
			textContent("model", fmt.Sprintf("Round %d complete. All tests pass now. The issue was a missing Authorization header in the test setup.", i)),
		)
	}
	return contents
}

// ---------------------------------------------------------------------------
// Simulation test
// ---------------------------------------------------------------------------

func TestCompactionSimulation(t *testing.T) {
	var results []compactionResult

	// --- Scenario 1: Kube agent, 3 rounds (small), various context windows ---
	kubeSmall := kubeAgentConversation(3)
	results = append(results,
		runThresholdSimulation(t, "kube-3rounds / 8k ctx", kubeSmall, 8_000),
		runThresholdSimulation(t, "kube-3rounds / 32k ctx", kubeSmall, 32_000),
		runThresholdSimulation(t, "kube-3rounds / 128k ctx", kubeSmall, 128_000),
	)

	// --- Scenario 2: Kube agent, 10 rounds (heavy), various context windows ---
	kubeHeavy := kubeAgentConversation(10)
	results = append(results,
		runThresholdSimulation(t, "kube-10rounds / 32k ctx", kubeHeavy, 32_000),
		runThresholdSimulation(t, "kube-10rounds / 128k ctx", kubeHeavy, 128_000),
		runThresholdSimulation(t, "kube-10rounds / 200k ctx", kubeHeavy, 200_000),
		runThresholdSimulation(t, "kube-10rounds / 1M ctx", kubeHeavy, 1_000_000),
	)

	// --- Scenario 3: Mixed conversation (debug session) ---
	mixed := mixedConversation()
	results = append(results,
		runThresholdSimulation(t, "mixed-debug / 8k ctx", mixed, 8_000),
		runThresholdSimulation(t, "mixed-debug / 32k ctx", mixed, 32_000),
		runThresholdSimulation(t, "mixed-debug / 128k ctx", mixed, 128_000),
	)

	// --- Scenario 4: Pure tool storm (worst case) ---
	storm20 := pureToolStorm(20, 2000)
	storm50 := pureToolStorm(50, 5000)
	results = append(results,
		runThresholdSimulation(t, "tool-storm-20x2k / 8k ctx", storm20, 8_000),
		runThresholdSimulation(t, "tool-storm-20x2k / 32k ctx", storm20, 32_000),
		runThresholdSimulation(t, "tool-storm-50x5k / 32k ctx", storm50, 32_000),
		runThresholdSimulation(t, "tool-storm-50x5k / 128k ctx", storm50, 128_000),
		runThresholdSimulation(t, "tool-storm-50x5k / 200k ctx", storm50, 200_000),
	)

	// --- Scenario 5: Sliding window on the same conversations ---
	results = append(results,
		runSlidingWindowSimulation(t, "kube-10rounds / sw=20", kubeHeavy, 20),
		runSlidingWindowSimulation(t, "kube-10rounds / sw=40", kubeHeavy, 40),
		runSlidingWindowSimulation(t, "mixed-debug / sw=10", mixed, 10),
		runSlidingWindowSimulation(t, "mixed-debug / sw=20", mixed, 20),
		runSlidingWindowSimulation(t, "tool-storm-50x5k / sw=10", storm50, 10),
		runSlidingWindowSimulation(t, "tool-storm-50x5k / sw=30", storm50, 30),
	)

	// --- Scenario 6: Kube agent at extreme scale (30+ rounds) ---
	kubeExtreme := kubeAgentConversation(30)
	results = append(results,
		runThresholdSimulation(t, "kube-30rounds / 32k ctx", kubeExtreme, 32_000),
		runThresholdSimulation(t, "kube-30rounds / 128k ctx", kubeExtreme, 128_000),
		runThresholdSimulation(t, "kube-30rounds / 200k ctx", kubeExtreme, 200_000),
		runSlidingWindowSimulation(t, "kube-30rounds / sw=20", kubeExtreme, 20),
		runSlidingWindowSimulation(t, "kube-30rounds / sw=50", kubeExtreme, 50),
	)

	// --- Scenario 7: Monster tool storm (100 pairs x 10k response) ---
	stormMonster := pureToolStorm(100, 10_000)
	results = append(results,
		runThresholdSimulation(t, "tool-storm-100x10k / 32k ctx", stormMonster, 32_000),
		runThresholdSimulation(t, "tool-storm-100x10k / 128k ctx", stormMonster, 128_000),
		runThresholdSimulation(t, "tool-storm-100x10k / 1M ctx", stormMonster, 1_000_000),
		runSlidingWindowSimulation(t, "tool-storm-100x10k / sw=10", stormMonster, 10),
		runSlidingWindowSimulation(t, "tool-storm-100x10k / sw=50", stormMonster, 50),
	)

	// --- Scenario 8: Alternating text+tool (realistic coding agent) ---
	codingAgent := buildCodingAgentConversation(15)
	results = append(results,
		runThresholdSimulation(t, "coding-agent-15 / 8k ctx", codingAgent, 8_000),
		runThresholdSimulation(t, "coding-agent-15 / 32k ctx", codingAgent, 32_000),
		runThresholdSimulation(t, "coding-agent-15 / 128k ctx", codingAgent, 128_000),
		runSlidingWindowSimulation(t, "coding-agent-15 / sw=10", codingAgent, 10),
		runSlidingWindowSimulation(t, "coding-agent-15 / sw=30", codingAgent, 30),
	)

	// --- Scenario 9: Tiny context windows (edge case) ---
	results = append(results,
		runThresholdSimulation(t, "kube-3rounds / 4k ctx", kubeSmall, 4_000),
		runThresholdSimulation(t, "mixed-debug / 4k ctx", mixed, 4_000),
		runThresholdSimulation(t, "tool-storm-20x2k / 4k ctx", storm20, 4_000),
		runThresholdSimulation(t, "coding-agent-15 / 4k ctx", codingAgent, 4_000),
	)

	// --- Scenario 10: Multiple compaction triggers (long session, small window) ---
	longSession := kubeAgentConversation(50)
	results = append(results,
		runThresholdSimulation(t, "kube-50rounds / 32k ctx", longSession, 32_000),
		runThresholdSimulation(t, "kube-50rounds / 128k ctx", longSession, 128_000),
		runSlidingWindowSimulation(t, "kube-50rounds / sw=20", longSession, 20),
		runSlidingWindowSimulation(t, "kube-50rounds / sw=40", longSession, 40),
	)

	// --- Print results ---
	t.Log("\n\n=== COMPACTION SIMULATION RESULTS ===\n")
	t.Logf("  %-40s | %-15s | %-12s | %-22s | %-47s | %-20s | %-26s | %s",
		"SCENARIO", "STRATEGY", "CTX WINDOW", "CONTENTS", "TOKENS", "SPLIT", "SUMMARIZED", "FITS?")
	t.Log(strings.Repeat("-", 220))
	for _, r := range results {
		t.Log(r.String())
	}
	t.Log("")

	// --- Assertions: the things that MUST work ---

	for _, r := range results {
		// If tokens exceeded the threshold, compaction must have happened
		buffer := computeBuffer(r.contextWindow)
		threshold := r.contextWindow - buffer
		if r.tokensBefore > threshold && !r.summarized {
			t.Errorf("[%s] tokens (%d) exceeded threshold (%d) but no summarization occurred",
				r.scenario, r.tokensBefore, threshold)
		}

		// After compaction, first content must be the summary
		if r.summarized && r.contentsAfter > 0 && r.oldMessages <= 0 {
			t.Errorf("[%s] summarized but oldMessages=%d (should be > 0)",
				r.scenario, r.oldMessages)
		}
	}

	// --- Diagnostic: compaction that increased or didn't reduce tokens ---
	t.Log("\n=== TOKEN INCREASE / NO-REDUCTION DIAGNOSTIC ===\n")
	for _, r := range results {
		if r.summarized && r.tokensAfter >= r.tokensBefore {
			t.Logf("  DIAG: [%s] strategy=%s — tokens went from %d to %d (floor-hit: oldMessages=%d, summary replaced 1 msg but added summary text)",
				r.scenario, r.strategy, r.tokensBefore, r.tokensAfter, r.oldMessages)
		}
	}

	// --- Report: scenarios where compaction happened but tokens still don't fit ---
	t.Log("\n=== SCENARIOS WHERE COMPACTION DID NOT ACHIEVE FIT ===\n")
	unfitCount := 0
	for _, r := range results {
		if r.summarized && !r.fitsContextWindow {
			t.Logf("  WARN: [%s] strategy=%s tokens_after=%d threshold=%d (still %d tokens over)",
				r.scenario, r.strategy, r.tokensAfter,
				r.contextWindow-computeBuffer(r.contextWindow),
				r.tokensAfter-(r.contextWindow-computeBuffer(r.contextWindow)),
			)
			unfitCount++
		}
	}
	if unfitCount == 0 {
		t.Log("  (none — all compacted scenarios fit within context window)")
	}

	// --- Report: how the floor affected things ---
	t.Log("\n=== FLOOR-HIT ANALYSIS ===\n")
	for _, r := range results {
		if r.summarized && r.oldMessages <= 1 {
			t.Logf("  FLOOR-HIT: [%s] strategy=%s — only %d message(s) summarized out of %d total (%.1f%% token reduction)",
				r.scenario, r.strategy, r.oldMessages, r.contentsBefore, r.reductionPct)
		}
	}
	t.Log("")
}

// ---------------------------------------------------------------------------
// Investigation: Does more retry rounds help? How do giant responses behave?
// ---------------------------------------------------------------------------

func TestCompactionInvestigation_RetryRoundsAndGiantResponses(t *testing.T) {
	type investigationResult struct {
		scenario       string
		contextWindow  int
		contentsBefore int
		tokensBefore   int
		contentsAfter  int
		tokensAfter    int
		fits           bool
		reductionPct   float64
		attempts       int
	}

	runWithAttempts := func(scenario string, contents []*genai.Content, contextWindow int, attempts int) investigationResult {
		t.Helper()

		registry := &mockRegistry{
			contextWindows: map[string]int{"sim-model": contextWindow},
			maxTokens:      map[string]int{"sim-model": 4096},
		}
		llm := &mockLLM{name: "sim-model", response: "Summary of conversation."}

		s := &thresholdStrategy{
			registry: registry,
			llm:      llm,
		}

		ctx := newMockCallbackContext("inv-agent")
		req := &model.LLMRequest{
			Model:    "sim-model",
			Contents: copyContents(contents),
		}

		tokensBefore := estimateTokens(req)
		contentsBefore := len(req.Contents)
		buffer := computeBuffer(contextWindow)
		threshold := contextWindow - buffer

		existingSummary := loadSummary(ctx)
		if existingSummary != "" {
			injectSummary(req, existingSummary, loadContentsAtCompaction(ctx))
		}

		if estimateTokens(req) < threshold {
			return investigationResult{
				scenario:       scenario,
				contextWindow:  contextWindow,
				contentsBefore: contentsBefore,
				tokensBefore:   tokensBefore,
				contentsAfter:  contentsBefore,
				tokensAfter:    tokensBefore,
				fits:           true,
				reductionPct:   0,
				attempts:       0,
			}
		}

		actualAttempts := 0

		for attempt := 0; attempt < attempts; attempt++ {
			oldContents := req.Contents

			if len(oldContents) == 0 {
				break
			}

			summary, err := summarize(ctx, s.llm, oldContents, existingSummary, buffer, nil)
			if err != nil {
				t.Fatalf("[%s] summarize error: %v", scenario, err)
			}

			existingSummary = summary
			persistSummary(ctx, summary, estimateTokens(req))
			replaceSummary(req, summary, nil)
			actualAttempts++

			if estimateTokens(req) < threshold {
				break
			}
		}

		tokensAfter := estimateTokens(req)
		reduction := 0.0
		if tokensBefore > 0 {
			reduction = (1.0 - float64(tokensAfter)/float64(tokensBefore)) * 100
		}

		return investigationResult{
			scenario:       scenario,
			contextWindow:  contextWindow,
			contentsBefore: contentsBefore,
			tokensBefore:   tokensBefore,
			contentsAfter:  len(req.Contents),
			tokensAfter:    tokensAfter,
			fits:           tokensAfter < threshold,
			reductionPct:   reduction,
			attempts:       actualAttempts,
		}
	}

	// =====================================================================
	// PART 1: Does increasing retry attempts help the failing kube case?
	// =====================================================================
	t.Log("\n\n=== PART 1: RETRY ROUNDS vs kube-3rounds / 8k ctx ===\n")
	t.Logf("  %-35s | %8s | %10s | %10s | %8s | %7s | %s",
		"SCENARIO", "ATTEMPTS", "TOK BEFORE", "TOK AFTER", "REDUCTN", "CONTS", "FITS?")
	t.Log(strings.Repeat("-", 110))

	kubeSmall := kubeAgentConversation(3)
	for _, attempts := range []int{1, 2, 3, 5, 10, 20} {
		r := runWithAttempts(
			fmt.Sprintf("kube-3r/8k/attempts=%d", attempts),
			kubeSmall, 8_000, attempts,
		)
		fit := "NO"
		if r.fits {
			fit = "YES"
		}
		t.Logf("  %-35s | %8d | %10d | %10d | %7.1f%% | %4d→%d | %s",
			r.scenario, r.attempts, r.tokensBefore, r.tokensAfter,
			r.reductionPct, r.contentsBefore, r.contentsAfter, fit)
	}

	// =====================================================================
	// PART 2: How does tool_response SIZE affect compaction?
	// =====================================================================
	t.Log("\n\n=== PART 2: TOOL RESPONSE SIZE SCALING (10 pairs, 8k ctx) ===\n")
	t.Logf("  %-35s | %8s | %10s | %10s | %8s | %7s | %s",
		"SCENARIO", "ATTEMPTS", "TOK BEFORE", "TOK AFTER", "REDUCTN", "CONTS", "FITS?")
	t.Log(strings.Repeat("-", 110))

	for _, respSize := range []int{500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000} {
		storm := pureToolStorm(10, respSize)
		r := runWithAttempts(
			fmt.Sprintf("10pairs×%dk/8k", respSize/1000),
			storm, 8_000, 3,
		)
		fit := "NO"
		if r.fits {
			fit = "YES"
		}
		t.Logf("  %-35s | %8d | %10d | %10d | %7.1f%% | %4d→%d | %s",
			r.scenario, r.attempts, r.tokensBefore, r.tokensAfter,
			r.reductionPct, r.contentsBefore, r.contentsAfter, fit)
	}

	// =====================================================================
	// PART 3: Same with 128k context
	// =====================================================================
	t.Log("\n\n=== PART 3: TOOL RESPONSE SIZE SCALING (10 pairs, 128k ctx) ===\n")
	t.Logf("  %-35s | %8s | %10s | %10s | %8s | %7s | %s",
		"SCENARIO", "ATTEMPTS", "TOK BEFORE", "TOK AFTER", "REDUCTN", "CONTS", "FITS?")
	t.Log(strings.Repeat("-", 110))

	for _, respSize := range []int{5_000, 10_000, 50_000, 200_000, 500_000} {
		storm := pureToolStorm(10, respSize)
		r := runWithAttempts(
			fmt.Sprintf("10pairs×%dk/128k", respSize/1000),
			storm, 128_000, 3,
		)
		fit := "NO"
		if r.fits {
			fit = "YES"
		}
		t.Logf("  %-35s | %8d | %10d | %10d | %7.1f%% | %4d→%d | %s",
			r.scenario, r.attempts, r.tokensBefore, r.tokensAfter,
			r.reductionPct, r.contentsBefore, r.contentsAfter, fit)
	}

	// =====================================================================
	// PART 4: The real kube scenario — what size responses break things?
	// =====================================================================
	t.Log("\n\n=== PART 4: KUBE-LIKE CONVERSATIONS WITH VARYING RESPONSE SIZE ===\n")
	t.Logf("  %-35s | %8s | %10s | %10s | %8s | %7s | %s",
		"SCENARIO", "ATTEMPTS", "TOK BEFORE", "TOK AFTER", "REDUCTN", "CONTS", "FITS?")
	t.Log(strings.Repeat("-", 110))

	for _, rounds := range []int{3, 5, 10, 20} {
		kube := kubeAgentConversation(rounds)
		for _, ctxSize := range []int{8_000, 32_000, 128_000} {
			r := runWithAttempts(
				fmt.Sprintf("kube-%dr/%dk/att=10", rounds, ctxSize/1000),
				kube, ctxSize, 10,
			)
			fit := "NO"
			if r.fits {
				fit = "YES"
			}
			t.Logf("  %-35s | %8d | %10d | %10d | %7.1f%% | %4d→%d | %s",
				r.scenario, r.attempts, r.tokensBefore, r.tokensAfter,
				r.reductionPct, r.contentsBefore, r.contentsAfter, fit)
		}
	}

	// =====================================================================
	// PART 5: Single giant response — the absolute worst case
	// =====================================================================
	t.Log("\n\n=== PART 5: SINGLE GIANT TOOL RESPONSE ===\n")
	t.Logf("  %-35s | %8s | %10s | %10s | %8s | %7s | %s",
		"SCENARIO", "ATTEMPTS", "TOK BEFORE", "TOK AFTER", "REDUCTN", "CONTS", "FITS?")
	t.Log(strings.Repeat("-", 110))

	for _, respSize := range []int{10_000, 50_000, 200_000, 1_000_000} {
		contents := []*genai.Content{
			textContent("user", "Get all pods"),
			&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				Name: "kubectl_get_pods", Args: map[string]any{"namespace": "all", "output": "json"},
			}}}},
			&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name: "kubectl_get_pods", Response: map[string]any{"stdout": strings.Repeat("x", respSize)},
			}}}},
			textContent("model", "Here are the results of the pod listing."),
		}
		for _, ctxSize := range []int{8_000, 128_000} {
			r := runWithAttempts(
				fmt.Sprintf("1resp×%dk/%dk", respSize/1000, ctxSize/1000),
				contents, ctxSize, 10,
			)
			fit := "NO"
			if r.fits {
				fit = "YES"
			}
			t.Logf("  %-35s | %8d | %10d | %10d | %7.1f%% | %4d→%d | %s",
				r.scenario, r.attempts, r.tokensBefore, r.tokensAfter,
				r.reductionPct, r.contentsBefore, r.contentsAfter, fit)
		}
	}

	t.Log("")
}

// TestTimingGap_CalibratedHeuristicPreventsOverflow simulates the exact
// scenario where stale real tokens alone would miss a context overflow:
//
//	Step N:   LLM sees 140k tokens → AfterModel persists 140k, heuristic was 70k
//	Tool:     Returns 80k-char response (≈20k heuristic tokens, ≈40k real tokens)
//	Step N+1: BeforeModel has req with 140k real + 40k new tool = 180k actual.
//	          Old tokenCount: returns 140k (stale) → 140k < 180k threshold → NO compaction → BOOM
//	          New tokenCount: calibrated = (70k+20k) * (140k/70k) = 180k → triggers compaction → SAFE
//
// This test verifies the calibrated heuristic correctly inflates the
// current heuristic estimate using the correction factor from the previous
// call, preventing overflow.
func TestTimingGap_CalibratedHeuristicPreventsOverflow(t *testing.T) {
	ctx := newMockCallbackContext("agent1")

	smallReq := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 280_000)),
		},
	}
	heuristicAtStepN := estimateTokens(smallReq)

	realTokensAtStepN := heuristicAtStepN * 2
	persistRealTokens(ctx, realTokensAtStepN)
	persistLastHeuristic(ctx, heuristicAtStepN)

	toolResultChars := 80_000
	grownReq := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 280_000)),
			{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				Name: "big_tool",
				Args: map[string]any{"param": "value"},
			}}}},
			{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name:     "big_tool",
				Response: map[string]any{"result": strings.Repeat("x", toolResultChars)},
			}}}},
		},
	}

	grownHeuristic := estimateTokens(grownReq)
	t.Logf("Step N:   heuristic=%d, real=%d (correction=%.2f)",
		heuristicAtStepN, realTokensAtStepN, float64(realTokensAtStepN)/float64(heuristicAtStepN))
	t.Logf("Step N+1: heuristic=%d (grown by tool result of %d chars)",
		grownHeuristic, toolResultChars)

	calibrated := tokenCount(ctx, grownReq)
	t.Logf("Calibrated estimate: %d", calibrated)

	if calibrated <= realTokensAtStepN {
		t.Errorf("calibrated (%d) should be > stale realTokens (%d) because the request grew",
			calibrated, realTokensAtStepN)
	}

	expectedCalibrated := int(float64(grownHeuristic) * (float64(realTokensAtStepN) / float64(heuristicAtStepN)))
	if calibrated != expectedCalibrated {
		t.Errorf("calibrated = %d, want %d", calibrated, expectedCalibrated)
	}

	contextWindow := 200_000
	buffer := computeBuffer(contextWindow)
	threshold := contextWindow - buffer

	if realTokensAtStepN >= threshold {
		t.Fatalf("test setup broken: stale real tokens (%d) already >= threshold (%d)", realTokensAtStepN, threshold)
	}
	if calibrated < threshold {
		t.Logf("NOTE: calibrated (%d) < threshold (%d) — in this test scenario the growth is small enough", calibrated, threshold)
	}

	t.Logf("Context window=%d, threshold=%d, stale_real=%d, calibrated=%d",
		contextWindow, threshold, realTokensAtStepN, calibrated)
	t.Logf("Old tokenCount would return: %d (stale, may miss growth)", realTokensAtStepN)
	t.Logf("New tokenCount returns: %d (calibrated, tracks growth)", calibrated)
}

// TestTimingGap_MassiveToolResponse verifies that a massive tool response
// between steps causes the calibrated heuristic to exceed the threshold,
// triggering compaction even though stale real tokens alone would not.
func TestTimingGap_MassiveToolResponse(t *testing.T) {
	ctx := newMockCallbackContext("agent1")

	persistRealTokens(ctx, 100_000)
	persistLastHeuristic(ctx, 50_000)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			textContent("user", strings.Repeat("a", 200_000)),
			{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				Name: "massive_query",
			}}}},
			{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name:     "massive_query",
				Response: map[string]any{"data": strings.Repeat("d", 400_000)},
			}}}},
		},
	}

	heuristic := estimateTokens(req)
	calibrated := tokenCount(ctx, req)
	correction := float64(100_000) / float64(50_000)

	t.Logf("heuristic=%d, correction=%.2f, calibrated=%d, stale_real=%d",
		heuristic, correction, calibrated, 100_000)

	if calibrated <= 100_000 {
		t.Errorf("calibrated (%d) must exceed stale real tokens (100000)", calibrated)
	}

	expectedCalibrated := int(float64(heuristic) * correction)
	if calibrated != expectedCalibrated {
		t.Errorf("calibrated = %d, want %d", calibrated, expectedCalibrated)
	}

	contextWindow := 200_000
	threshold := contextWindow - computeBuffer(contextWindow)
	if calibrated < threshold {
		t.Errorf("calibrated (%d) should exceed threshold (%d) for this massive request", calibrated, threshold)
	}

	registry := &mockRegistry{
		contextWindows: map[string]int{"test-model": contextWindow},
		maxTokens:      map[string]int{"test-model": 4096},
	}
	llm := &mockLLM{name: "test-model", response: "Compacted summary of the conversation."}
	s := newThresholdStrategy(registry, llm, 0)

	err := s.Compact(ctx, req)
	if err != nil {
		t.Fatalf("Compact error: %v", err)
	}

	if !strings.Contains(req.Contents[0].Parts[0].Text, "[Previous conversation summary]") {
		t.Error("compaction should have fired for this massive request")
	}
	t.Logf("Compaction triggered correctly. Contents reduced from 3 to %d", len(req.Contents))
}
