// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Pi coding agent extension: register a running `invar serve` as a provider, so every model call
// the agent makes — including tool calls — lands in the INVAR worldline as a re-executable receipt.
//
//   invar serve --model <gguf> [--device CUDA0 --ngl 99]          # port 8577
//   pi -e invar_provider.js --provider invar --model invar/local "fix the failing test"
//
// Env: INVAR_BASE_URL (default http://127.0.0.1:8577/v1). The model id is cosmetic — `invar serve`
// answers with the one model it was started with (see /v1/models). Tools work with any model that
// can emit a tool call (Qwen2.5-style <tool_call> JSON, or a bare/fenced JSON object).
// Note: with tools enabled Pi reads stdin; in scripts run it with `< /dev/null`.
export default function (pi) {
  pi.registerProvider("invar", {
    name: "INVAR (receipted local inference)",
    baseUrl: process.env.INVAR_BASE_URL || "http://127.0.0.1:8577/v1",
    apiKey: "none",
    api: "openai-completions",
    models: [
      {
        id: "local",
        name: "INVAR-served model (whatever `invar serve` loaded)",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: parseInt(process.env.INVAR_CONTEXT || "8192", 10),
        maxTokens: parseInt(process.env.INVAR_MAX_TOKENS || "1024", 10),
      },
    ],
  });
}
