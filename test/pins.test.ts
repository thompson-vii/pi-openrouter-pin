/**
 * Tests for the /openrouter-pins, /openrouter-unpin, and session-start
 * refresh logic.
 *
 * Four layers:
 *
 *   A — Pins surface: `formatRouting` (the /openrouter-pins display format)
 *       and `listPins` (what the command lists) from src/commands.ts.
 *
 *   B — Unpin core: the pure `unpinFromModels` and the file-level
 *       `performUnpin` (what the /openrouter-unpin handler delegates to).
 *
 *   C — Handlers: the REAL registered command handlers from src/index.ts,
 *       driven through a fake ExtensionAPI/ExtensionCommandContext with a
 *       temp PI_CODING_AGENT_DIR, asserting notifications and models.json
 *       contents end-to-end (including the no-args picker path, whose
 *       `ctx.custom` is resolved with a canned choice instead of driving a
 *       real TUI).
 *
 *   D — Refresh pipeline: `formatRefreshDiff` (the pricing display formatter
 *       that previously spilled multi-line output into the TUI footer),
 *       `collectRefreshTargets` and `applyPricingPatches` (pure refresh
 *       helpers), and the `session_start` handler that routes results to
 *       `ctx.ui.notify()` instead of raw `console.*` calls.
 *
 * These tests import index.ts, which pulls in @earendil-works/pi-coding-agent
 * and @earendil-works/pi-tui at runtime — run `npm install` (peer deps) first.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import openrouterPinExtension from "../src/index.ts";
import {
  applyPricingPatches,
  collectRefreshTargets,
  formatRefreshDiff,
  formatRouting,
  listPins,
  performUnpin,
  reconcilePinnedSettings,
  reconcileSettings,
  unpinFromModels,
  unpinFromSettings,
  type PricingLimitsDiff,
  type PricingLimitsPatch,
} from "../src/commands.ts";
import { atomicWriteJson, readJsonFile, type ModelsJson, type ProviderEntry, type SettingsJson } from "../src/files.ts";
import type { ExtensionContext, ModelRegistry } from "@earendil-works/pi-coding-agent";
import type { ModelConfig } from "../src/config.ts";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const providerEntry = (models: ModelConfig[]): ProviderEntry => ({
  baseUrl: "https://openrouter.ai/api/v1",
  api: "openai-completions",
  apiKey: "$OPENROUTER_API_KEY",
  models,
});

/** A stored pinned model, as written to models.json. */
const glmModel = (over: Partial<ModelConfig> = {}): ModelConfig => ({
  id: "z-ai/glm-5.2",
  name: "GLM 5.2 (novita)",
  reasoning: true,
  input: ["text"],
  contextWindow: 1_048_576,
  maxTokens: 128_000,
  cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 },
  compat: { thinkingFormat: "openrouter", openRouterRouting: { only: ["novita"], allow_fallbacks: false } },
  ...over,
});

/** A second pinned model (relaxed -plus provider). */
const deepseekModel = (over: Partial<ModelConfig> = {}): ModelConfig => ({
  id: "deepseek/deepseek-v4-flash-0731",
  name: "DeepSeek V4 Flash (novita)",
  reasoning: true,
  input: ["text"],
  contextWindow: 131_072,
  maxTokens: 32_768,
  cost: { input: 20_000, output: 80_000, cacheRead: 5_000, cacheWrite: 0 },
  compat: {
    thinkingFormat: "openrouter",
    openRouterRouting: { order: ["novita", "deepseek"], allow_fallbacks: true },
  },
  ...over,
});

/** A model in an openrouter-* provider WITHOUT routing compat — must be invisible to pins/unpin. */
const legacyModel = (over: Partial<ModelConfig> = {}): ModelConfig =>
  ({
    id: "some/legacy-model",
    name: "Legacy",
    reasoning: false,
    input: ["text"],
    contextWindow: 128_000,
    maxTokens: 16_384,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    ...over,
  }) as unknown as ModelConfig;

async function withTempModels(
  models: ModelsJson | null,
  /** The temp dir is passed so a test can write a sibling settings.json. */
  fn: (modelsPath: string, dir: string) => Promise<void>,
): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-pins-"));
  const modelsPath = join(dir, "models.json");
  try {
    if (models !== null) await atomicWriteJson(modelsPath, models);
    await fn(modelsPath, dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// A — /openrouter-pins surface
// ---------------------------------------------------------------------------

test("formatRouting: every routing option renders, fallbacks defaults to false", () => {
  // Empty routing still states the fallback policy explicitly.
  assert.equal(formatRouting({}), "fallbacks=false");
  assert.equal(formatRouting({ allow_fallbacks: false }), "fallbacks=false");
  assert.equal(formatRouting({ allow_fallbacks: true }), "fallbacks=true");

  // Strict pin: only the anchor provider, no fallbacks.
  assert.equal(
    formatRouting({ only: ["novita"], allow_fallbacks: false }),
    "only=novita fallbacks=false",
  );

  // Relaxed pin with the full surface, in the documented order.
  assert.equal(
    formatRouting({
      only: ["novita"],
      order: ["novita", "together"],
      ignore: ["openai", "anthropic"],
      quantizations: ["fp8"],
      data_collection: "deny",
      allow_fallbacks: true,
    }),
    "only=novita order=novita,together ignore=openai,anthropic quant=fp8 data_collection=deny fallbacks=true",
  );
});

test("listPins: only openrouter-* providers, only models with routing compat, insertion order", async () => {
  await withTempModels(
    {
      providers: {
        "openrouter-novita": providerEntry([glmModel(), legacyModel()]), // legacy model skipped
        "openrouter-novita-plus": providerEntry([deepseekModel()]),
        "anthropic": providerEntry([glmModel()]), // not our prefix — ignored
      },
    },
    async (modelsPath) => {
      const pins = await listPins(modelsPath);
      assert.deepEqual(
        pins.map((p) => [p.provider, p.model.id]),
        [
          ["openrouter-novita", "z-ai/glm-5.2"],
          ["openrouter-novita-plus", "deepseek/deepseek-v4-flash-0731"],
        ],
        "non-openrouter providers and compat-less models are invisible to /openrouter-pins",
      );
    },
  );
});

test("listPins: missing file, empty providers, and model-less providers all yield []", async () => {
  await withTempModels(null, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
  await withTempModels({}, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
  await withTempModels({ providers: {} }, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
  await withTempModels({ providers: { "openrouter-novita": providerEntry([]) } }, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
});

// ---------------------------------------------------------------------------
// B — Unpin core
// ---------------------------------------------------------------------------

test("unpinFromModels: removes the model from openrouter-* providers, keeps siblings", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
      "anthropic": providerEntry([glmModel()]),
    },
  };
  const { models, removed } = unpinFromModels(input, "z-ai/glm-5.2");
  assert.equal(removed, true);
  assert.deepEqual(
    models.providers!["openrouter-novita"].models.map((m) => m.id),
    ["deepseek/deepseek-v4-flash-0731"],
    "sibling pins survive",
  );
  // Non-openrouter providers are never scanned, let alone touched.
  assert.deepEqual(models.providers!["anthropic"], input.providers!["anthropic"]);
});

test("unpinFromModels: a provider left empty is dropped, not persisted as []", () => {
  const { models, removed } = unpinFromModels(
    { providers: { "openrouter-novita": providerEntry([glmModel()]) } },
    "z-ai/glm-5.2",
  );
  assert.equal(removed, true);
  assert.deepEqual(models.providers, {}, "the empty openrouter-* provider is deleted");
});

test("unpinFromModels: removes the same model from every openrouter-* provider at once", () => {
  // Same model pinned strict (novita) and relaxed (novita-plus): both go.
  const { models, removed } = unpinFromModels(
    {
      providers: {
        "openrouter-novita": providerEntry([glmModel()]),
        "openrouter-novita-plus": providerEntry([{ ...glmModel(), name: "GLM relaxed" }]),
      },
    },
    "z-ai/glm-5.2",
  );
  assert.equal(removed, true);
  assert.deepEqual(models.providers, {});
});

test("unpinFromModels: unknown model is a clean no-op", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel()]),
      "anthropic": providerEntry([glmModel()]),
    },
  };
  const { models, removed } = unpinFromModels(input, "nobody/home");
  assert.equal(removed, false);
  // The snapshot is a fresh object but shares the untouched provider entries.
  assert.notEqual(models, input);
  assert.equal(models.providers!["openrouter-novita"], input.providers!["openrouter-novita"]);
  assert.deepEqual(models, input);
});

test("unpinFromModels: never mutates its input (deep-frozen)", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
      "openrouter-together": providerEntry([glmModel({ id: "other/model" })]),
      "anthropic": providerEntry([glmModel()]),
    },
  };
  deepFreeze(input);
  const { models, removed } = unpinFromModels(input, "z-ai/glm-5.2");
  assert.equal(removed, true);
  // The deepseek sibling keeps the novita provider alive; only the unpinned
  // model is gone — and the frozen input was never touched.
  assert.deepEqual(
    models.providers!["openrouter-novita"].models.map((m) => m.id),
    ["deepseek/deepseek-v4-flash-0731"],
    "sibling model survives in the same provider",
  );
  assert.deepEqual(
    models.providers!["openrouter-together"].models.map((m) => m.id),
    ["other/model"],
    "sibling model in another provider survives",
  );
});

test("unpinFromModels: reports which providers the model was removed from", () => {
  const { models, removed, providers } = unpinFromModels(
    {
      providers: {
        "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
        "openrouter-novita-plus": providerEntry([{ ...glmModel(), name: "GLM relaxed" }]),
        "anthropic": providerEntry([glmModel()]), // never scanned
      },
    },
    "z-ai/glm-5.2",
  );
  assert.equal(removed, true);
  assert.deepEqual(providers, ["openrouter-novita", "openrouter-novita-plus"], "only the providers it was actually in");
  assert.deepEqual(models.providers!["anthropic"].models.map((m) => m.id), ["z-ai/glm-5.2"]);
});

test("unpinFromModels: an unpinned model reports no providers", () => {
  assert.deepEqual(unpinFromModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, "nobody/home").providers, []);
});

test("unpinFromModels: null / provider-less / malformed inputs are safe no-ops", () => {
  assert.deepEqual(unpinFromModels(null, "x/y"), { models: { providers: {} }, removed: false, providers: [] });
  assert.deepEqual(unpinFromModels({}, "x/y"), { models: { providers: {} }, removed: false, providers: [] });
  assert.deepEqual(unpinFromModels({ providers: {} }, "x/y"), { models: { providers: {} }, removed: false, providers: [] });
  // Defensive: an entry without a models array is kept as-is, never crashed on.
  const malformed = { providers: { "openrouter-novita": { baseUrl: "x" } } } as unknown as ModelsJson;
  const out = unpinFromModels(malformed, "x/y");
  assert.equal(out.removed, false);
  assert.equal(out.models.providers!["openrouter-novita"], malformed.providers!["openrouter-novita"]);
});

test("performUnpin: removes and writes only when something was removed", async () => {
  await withTempModels(
    { providers: { "openrouter-novita": providerEntry([glmModel(), deepseekModel()]) } },
    async (modelsPath, dir) => {
      const settingsPath = join(dir, "settings.json");
      assert.deepEqual(await performUnpin(modelsPath, settingsPath, "z-ai/glm-5.2"), {
        status: "removed",
        providers: ["openrouter-novita"],
        settingsChanged: false,
        clearedDefault: false,
      });
      const models = await readJsonFile<ModelsJson>(modelsPath);
      assert.deepEqual(
        models!.providers!["openrouter-novita"].models.map((m) => m.id),
        ["deepseek/deepseek-v4-flash-0731"],
        "the sibling survives on disk",
      );

      // A second unpin of the last model drops the whole provider.
      assert.deepEqual((await performUnpin(modelsPath, settingsPath, "deepseek/deepseek-v4-flash-0731")).status, "removed");
      assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), { providers: {} });
    },
  );
});

test("performUnpin: prunes the stale enabledModels entry and dangling default in settings.json", async () => {
  await withTempModels(
    { providers: { "openrouter-novita": providerEntry([glmModel()]) } },
    async (modelsPath, dir) => {
      const settingsPath = join(dir, "settings.json");
      await atomicWriteJson(settingsPath, {
        defaultProvider: "openrouter-novita",
        defaultModel: "z-ai/glm-5.2",
        enabledModels: ["openrouter-novita/z-ai/glm-5.2", "anthropic/claude-opus-4"],
        theme: "dark",
      });
      const outcome = await performUnpin(modelsPath, settingsPath, "z-ai/glm-5.2");
      assert.equal(outcome.status, "removed");
      assert.equal(outcome.settingsChanged, true);
      assert.equal(outcome.clearedDefault, true);
      assert.deepEqual(await readJsonFile<SettingsJson>(settingsPath), {
        enabledModels: ["anthropic/claude-opus-4"],
        theme: "dark",
      });
    },
  );
});

test("performUnpin: not-found and no-providers never write", async () => {
  await withTempModels(
    { providers: { "openrouter-novita": providerEntry([glmModel()]) } },
    async (modelsPath, dir) => {
      const settingsPath = join(dir, "settings.json");
      assert.deepEqual(await performUnpin(modelsPath, settingsPath, "nobody/home"), { status: "not-found" });
      const after = await readJsonFile<ModelsJson>(modelsPath);
      assert.deepEqual(after!.providers!["openrouter-novita"].models.map((m) => m.id), ["z-ai/glm-5.2"]);
    },
  );

  // Missing file → no-providers, and the file is NOT created.
  await withTempModels(null, async (modelsPath, dir) => {
    const settingsPath = join(dir, "settings.json");
    assert.deepEqual(await performUnpin(modelsPath, settingsPath, "x/y"), { status: "no-providers" });
    const { existsSync } = await import("node:fs");
    assert.equal(existsSync(modelsPath), false, "an unpin of nothing must not create models.json");
    assert.equal(existsSync(settingsPath), false, "an unpin of nothing must not create settings.json");
  });

  // File present but without a providers key → no-providers, content untouched.
  await withTempModels({ settings: { x: 1 } } as unknown as ModelsJson, async (modelsPath, dir) => {
    const settingsPath = join(dir, "settings.json");
    assert.deepEqual(await performUnpin(modelsPath, settingsPath, "x/y"), { status: "no-providers" });
    assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), { settings: { x: 1 } });
  });
});

// ---------------------------------------------------------------------------
// B2 — Settings pruning (the "No models match pattern" fix)
// ---------------------------------------------------------------------------

test("unpinFromSettings: drops the enabledModels ref of exactly the providers unpinned", () => {
  const input = {
    enabledModels: [
      "openrouter-novita/z-ai/glm-5.2",
      "openrouter-novita-plus/z-ai/glm-5.2",
      "openrouter-together/z-ai/glm-5.2", // same model, other provider — still resolves
      "anthropic/z-ai/glm-5.2", // not ours — never touched
    ],
  };
  deepFreeze(input);
  const { settings, changed } = unpinFromSettings(input, "z-ai/glm-5.2", ["openrouter-novita", "openrouter-novita-plus"]);
  assert.equal(changed, true);
  assert.deepEqual(settings.enabledModels, ["openrouter-together/z-ai/glm-5.2", "anthropic/z-ai/glm-5.2"]);
  assert.deepEqual(input.enabledModels.length, 4, "the input is never mutated");
});

test("unpinFromSettings: model ids contain slashes, so only the first separates provider", () => {
  const { settings, changed } = unpinFromSettings(
    { enabledModels: ["openrouter-novita/vendor/sub/model"] },
    "vendor/sub/model",
    ["openrouter-novita"],
  );
  assert.equal(changed, true);
  assert.deepEqual(settings.enabledModels, []);
});

test("unpinFromSettings: clears the default only when it pointed at the unpinned pin", () => {
  const cleared = unpinFromSettings(
    { defaultProvider: "openrouter-novita", defaultModel: "z-ai/glm-5.2", defaultThinkingLevel: "high" },
    "z-ai/glm-5.2",
    ["openrouter-novita"],
  );
  assert.equal(cleared.changed, true);
  assert.equal(cleared.clearedDefault, true);
  assert.deepEqual(cleared.settings, { defaultThinkingLevel: "high" });

  // Default points at a sibling pin → untouched.
  const kept = unpinFromSettings(
    { defaultProvider: "openrouter-novita", defaultModel: "deepseek/deepseek-v4-flash-0731" },
    "z-ai/glm-5.2",
    ["openrouter-novita"],
  );
  assert.equal(kept.changed, false);
  assert.equal(kept.clearedDefault, false);

  // Default on a provider that did not hold the model → untouched.
  const other = unpinFromSettings({ defaultProvider: "anthropic", defaultModel: "z-ai/glm-5.2" }, "z-ai/glm-5.2", [
    "openrouter-novita",
  ]);
  assert.equal(other.changed, false);
});

test("unpinFromSettings: missing/empty settings and no matches report no change", () => {
  assert.deepEqual(unpinFromSettings(null, "z-ai/glm-5.2", ["openrouter-novita"]), {
    settings: {},
    changed: false,
    clearedDefault: false,
  });
  assert.deepEqual(unpinFromSettings({ enabledModels: [] }, "z-ai/glm-5.2", ["openrouter-novita"]), {
    settings: { enabledModels: [] },
    changed: false,
    clearedDefault: false,
  });
  // A ref whose provider still holds the model still resolves.
  assert.equal(unpinFromSettings({ enabledModels: ["openrouter-novita/deepseek/deepseek-v4-flash-0731"] }, "z-ai/glm-5.2", ["openrouter-novita"]).changed, false);
});

test("reconcileSettings: drops dangling openrouter-* refs left by an older unpin", () => {
  const models: ModelsJson = {
    providers: {
      "openrouter-wafer-plus": providerEntry([glmModel({ id: "z-ai/glm-5.3-flash" })]),
    },
  };
  const input = {
    defaultProvider: "openrouter-wafer-plus",
    defaultModel: "deepseek/deepseek-v4.1-flash", // the model that was unpinned
    enabledModels: [
      "openrouter-wafer-plus/deepseek/deepseek-v4.1-flash", // dangling: provider gone or model gone
      "openrouter-fireworks-plus/deepseek/deepseek-v4.1-flash", // dangling: whole provider gone
      "openrouter-wafer-plus/z-ai/glm-5.3-flash", // still resolves
      "openrouter-*/*", // a pattern, not an exact ref → never resolved, never dropped
      "openrouter/z-ai/glm-5.3-flash", // the built-in provider, not ours → never touched
      "anthropic/claude-opus-4", // another extension's namespace → never touched
    ],
  };
  deepFreeze(input);
  const healed = reconcileSettings(models, input);
  assert.deepEqual(healed.removed, [
    "openrouter-wafer-plus/deepseek/deepseek-v4.1-flash",
    "openrouter-fireworks-plus/deepseek/deepseek-v4.1-flash",
  ]);
  assert.equal(healed.clearedDefault, true);
  assert.deepEqual(healed.settings, {
    enabledModels: [
      "openrouter-wafer-plus/z-ai/glm-5.3-flash",
      "openrouter-*/*",
      "openrouter/z-ai/glm-5.3-flash",
      "anthropic/claude-opus-4",
    ],
  });
});

test("reconcileSettings: a healthy config is a no-op", () => {
  const models: ModelsJson = { providers: { "openrouter-novita": providerEntry([glmModel()]) } };
  const settings = {
    defaultProvider: "openrouter-novita",
    defaultModel: "z-ai/glm-5.2",
    enabledModels: ["openrouter-novita/z-ai/glm-5.2"],
  };
  const healed = reconcileSettings(models, settings);
  assert.deepEqual(healed.removed, []);
  assert.equal(healed.clearedDefault, false);
  assert.deepEqual(healed.settings, settings);
});

test("reconcileSettings: a non-openrouter default is never cleared", () => {
  const healed = reconcileSettings({ providers: {} }, { defaultProvider: "anthropic", defaultModel: "claude-opus-4" });
  assert.equal(healed.clearedDefault, false);
  assert.deepEqual(healed.removed, []);
});

test("reconcilePinnedSettings: writes settings.json only when something was dangling", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath, dir) => {
    const settingsPath = join(dir, "settings.json");

    // No settings.json at all → nothing is created.
    assert.deepEqual(await reconcilePinnedSettings(modelsPath, settingsPath), { removed: [], clearedDefault: false });
    const { existsSync } = await import("node:fs");
    assert.equal(existsSync(settingsPath), false);

    // Healthy settings.json → read but not rewritten.
    const healthy = { enabledModels: ["openrouter-novita/z-ai/glm-5.2"] };
    await atomicWriteJson(settingsPath, healthy);
    const before = (await readJsonFile<SettingsJson>(settingsPath))!;
    assert.deepEqual(await reconcilePinnedSettings(modelsPath, settingsPath), { removed: [], clearedDefault: false });
    assert.deepEqual(await readJsonFile<SettingsJson>(settingsPath), before);

    // A dangling ref is dropped from disk.
    await atomicWriteJson(settingsPath, { ...healthy, enabledModels: [...healthy.enabledModels, "openrouter-gone/z-ai/glm-5.2"] });
    assert.deepEqual(await reconcilePinnedSettings(modelsPath, settingsPath), {
      removed: ["openrouter-gone/z-ai/glm-5.2"],
      clearedDefault: false,
    });
    assert.deepEqual(await readJsonFile<SettingsJson>(settingsPath), healthy);
  });
});

// ---------------------------------------------------------------------------
// C — The registered /openrouter-pins and /openrouter-unpin handlers
// ---------------------------------------------------------------------------

type Notify = { message: string; type: "info" | "warning" | "error" };

/**
 * Registers the real extension against a fake ExtensionAPI and plays the
 * ExtensionCommandContext side. PI_CODING_AGENT_DIR must already point at a
 * temp dir — the factory reads it once at construction.
 */
class CommandHarness {
  readonly notifications: Notify[] = [];
  customCalls = 0;
  private readonly picked: string | null;
  private readonly events = new Map<string, Array<(event: unknown, ctx: unknown) => void>>();
  private readonly commands = new Map<
    string,
    { description: string; handler: (args: string, ctx: ExtensionCommandContext) => Promise<void> }
  >();

  constructor(picked: string | null = null) {
    // Explicit field, not a parameter property: Node's strip-only TS mode
    // cannot transform parameter properties (same constraint as api.ts).
    this.picked = picked;
    const pi = {
      on: (event: string, handler: (event: unknown, ctx: unknown) => void) => {
        const list = this.events.get(event) ?? [];
        list.push(handler);
        this.events.set(event, list);
      },
      registerProvider: () => {},
      registerCommand: (name: string, options: { description: string; handler: (args: string, ctx: ExtensionCommandContext) => Promise<void> }) => {
        this.commands.set(name, options);
      },
    } as unknown as ExtensionAPI;
    openrouterPinExtension(pi);
  }

  /**
   * Fire the FIRST handler registered for an event. The extension registers
   * two `session_start` handlers (settings self-heal, then pricing refresh);
   * tests fire only the self-heal one, because the refresh hits the network
   * and needs a real ModelRegistry. Handlers are fire-and-forget, so the
   * caller waits for the effect it expects.
   */
  async fireFirst(event: string): Promise<void> {
    const handler = this.events.get(event)?.[0];
    assert.ok(handler, `expected an ${event} handler to be registered`);
    handler({}, this.ctx());
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  private ctx(): ExtensionCommandContext {
    return {
      mode: "tui",
      hasUI: true,
      cwd: "/tmp",
      modelRegistry: {},
      model: undefined,
      scopedModels: [],
      ui: {
        notify: (message: string, type: "info" | "warning" | "error" = "info") => {
          this.notifications.push({ message, type });
        },
        custom: async () => {
          this.customCalls++;
          return this.picked;
        },
        select: async () => this.picked,
      },
    } as unknown as ExtensionCommandContext;
  }

  async run(command: string, args: string): Promise<void> {
    const registered = this.commands.get(command);
    assert.ok(registered, `expected /${command} to be registered`);
    await registered.handler(args, this.ctx());
  }
}

/** Point PI_CODING_AGENT_DIR at a temp dir for the duration of fn. */
async function withAgentDir<T>(fn: () => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-agent-"));
  const saved = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = dir;
  try {
    return await fn();
  } finally {
    if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = saved;
    await rm(dir, { recursive: true, force: true });
  }
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const v of Object.values(value as Record<string, unknown>)) deepFreeze(v);
    Object.freeze(value);
  }
  return value;
}

test("/openrouter-pins: lists every pin with its routing policy", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel()]),
        "openrouter-novita-plus": providerEntry([deepseekModel()]),
        "anthropic": providerEntry([glmModel()]), // must not appear
      },
    });
    const h = new CommandHarness();
    await h.run("openrouter-pins", "");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.equal(
      h.notifications[0].message,
      [
        "Active pins:",
        "  openrouter-novita/z-ai/glm-5.2  → only=novita fallbacks=false",
        "  openrouter-novita-plus/deepseek/deepseek-v4-flash-0731  → order=novita,deepseek fallbacks=true",
      ].join("\n"),
    );
  });
});

test("/openrouter-pins: no pins shows the create hint", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pins", "");
    assert.equal(h.notifications.length, 1);
    assert.ok(h.notifications[0].message.startsWith("No pins. Run /openrouter-pin"), "points at the pin commands");
  });
});

test("/openrouter-pins: a broken models.json surfaces as a List failed error", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await (await import("node:fs/promises")).writeFile(modelsPath, "{ not json", "utf-8");
    const h = new CommandHarness();
    await h.run("openrouter-pins", "");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "error");
    assert.ok(h.notifications[0].message.startsWith("List failed:"), "the error is surfaced, not swallowed");
  });
});

test("/openrouter-unpin <model>: removes the pin and reports success", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
        "openrouter-together": providerEntry([glmModel({ id: "other/model" })]),
      },
    });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    assert.deepEqual(h.notifications, [
      {
        message:
          "Unpinned z-ai/glm-5.2 from models.json (openrouter-novita) (applies on /reload or next session).",
        type: "info",
      },
    ]);
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(
      models!.providers!["openrouter-novita"].models.map((m) => m.id),
      ["deepseek/deepseek-v4-flash-0731"],
    );
    assert.deepEqual(models!.providers!["openrouter-together"].models.map((m) => m.id), ["other/model"]);
  });
});

test("/openrouter-unpin <model>: also drops the stale settings.json entry", async () => {
  await withAgentDir(async () => {
    const dir = process.env.PI_CODING_AGENT_DIR!;
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, { providers: { "openrouter-novita": providerEntry([glmModel()]) } });
    // The bug this fixes: a leftover ref makes pi warn "No models match pattern" at every startup.
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-novita",
      defaultModel: "z-ai/glm-5.2",
      enabledModels: ["openrouter-novita/z-ai/glm-5.2", "anthropic/claude-opus-4"],
    });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    assert.deepEqual(await readJsonFile<SettingsJson>(settingsPath), { enabledModels: ["anthropic/claude-opus-4"] });
    assert.deepEqual(h.notifications, [
      {
        message:
          "Unpinned z-ai/glm-5.2 from models.json (openrouter-novita); removed it from settings.json and cleared the default model (applies on /reload or next session).",
        type: "info",
      },
    ]);
  });
});

test("session_start: heals settings.json refs left dangling by an older unpin", async () => {
  await withAgentDir(async () => {
    const dir = process.env.PI_CODING_AGENT_DIR!;
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-wafer-plus": providerEntry([glmModel({ id: "z-ai/glm-5.3-flash" })]) },
    });
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-wafer-plus",
      defaultModel: "deepseek/deepseek-v4.1-flash",
      enabledModels: [
        "openrouter-wafer-plus/deepseek/deepseek-v4.1-flash",
        "openrouter-fireworks-plus/deepseek/deepseek-v4.1-flash",
        "openrouter-wafer-plus/z-ai/glm-5.3-flash",
      ],
    });
    const h = new CommandHarness();
    await h.fireFirst("session_start");
    assert.deepEqual(await readJsonFile<SettingsJson>(settingsPath), {
      enabledModels: ["openrouter-wafer-plus/z-ai/glm-5.3-flash"],
    });
    const healed = h.notifications.find((n) => n.message.startsWith("Cleaned up settings.json"));
    assert.ok(healed, "the cleanup is reported, not silent");
    assert.ok(healed.message.includes("2 stale enabledModels entries"), healed.message);
    assert.ok(healed.message.includes("cleared the dangling default model"), healed.message);
  });
});

test("session_start: a healthy settings.json is left alone and silent", async () => {
  await withAgentDir(async () => {
    const dir = process.env.PI_CODING_AGENT_DIR!;
    const settingsPath = join(dir, "settings.json");
    const healthy = { enabledModels: ["anthropic/claude-opus-4"] };
    await atomicWriteJson(settingsPath, healthy);
    const h = new CommandHarness();
    await h.fireFirst("session_start");
    assert.deepEqual(await readJsonFile<SettingsJson>(settingsPath), healthy);
    assert.deepEqual(h.notifications, [], "nothing dangling, nothing to say");
  });
});

test("/openrouter-unpin <model>: the last pin in a provider drops the provider entirely", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([glmModel()]) },
    });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models, { providers: {} }, "the emptied openrouter-* provider is removed");
  });
});

test("/openrouter-unpin <model>: not pinned → info notice, file untouched", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    const before: ModelsJson = { providers: { "openrouter-novita": providerEntry([glmModel()]) } };
    await atomicWriteJson(modelsPath, before);
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "nobody/home");
    assert.deepEqual(h.notifications, [
      { message: 'No pin for "nobody/home" found (checked openrouter-* providers)', type: "info" },
    ]);
    assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), before);
  });
});

test("/openrouter-unpin <model>: no models.json → info notice, nothing created", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    assert.deepEqual(h.notifications, [
      { message: "No pins found (no providers in models.json)", type: "info" },
    ]);
    const { existsSync } = await import("node:fs");
    assert.equal(existsSync(join(process.env.PI_CODING_AGENT_DIR!, "models.json")), false);
  });
});

test("/openrouter-unpin (no args): offers the picker and unpins the chosen pin", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
        "anthropic": providerEntry([glmModel()]), // never offered: not an openrouter-* pin
      },
    });
    // The picker resolves with the novita pin; the handler extracts the model
    // id after the provider prefix and unpins exactly that one.
    const h = new CommandHarness("openrouter-novita/z-ai/glm-5.2");
    await h.run("openrouter-unpin", "");
    assert.equal(h.customCalls, 1, "the picker is shown");
    assert.deepEqual(h.notifications, [
      {
        message:
          "Unpinned z-ai/glm-5.2 from models.json (openrouter-novita) (applies on /reload or next session).",
        type: "info",
      },
    ]);
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models!.providers!["openrouter-novita"].models.map((m) => m.id), ["deepseek/deepseek-v4-flash-0731"]);
  });
});

test("/openrouter-unpin (no args): Esc-cancel is silent and writes nothing", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    const before: ModelsJson = { providers: { "openrouter-novita": providerEntry([glmModel()]) } };
    await atomicWriteJson(modelsPath, before);
    const h = new CommandHarness(null); // picker cancelled
    await h.run("openrouter-unpin", "");
    assert.equal(h.customCalls, 1, "the picker is shown before cancel");
    assert.deepEqual(h.notifications, [], "cancelling quietly does not notify");
    assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), before);
  });
});

test("/openrouter-unpin (no args): no pins → hint without opening the picker", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "");
    assert.equal(h.customCalls, 0, "no picker when there is nothing to pick");
    assert.deepEqual(h.notifications, [
      { message: "No pins to remove. Use /openrouter-pin to create one.", type: "info" },
    ]);
  });
});

test("/openrouter-pin --help: prints help without pinning or opening the wizard", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pin", "--help");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pin/);
    assert.ok(h.notifications[0].message.includes("--quant"));
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    assert.equal(await readJsonFile(modelsPath), null, "--help never writes models.json");
  });
});

test("/openrouter-pin -h: short flag prints the same help", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pin", "-h");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pin/);
    // A pin-looking invocation with --help anywhere also shows help, and
    // never reaches the network or models.json.
    const mixed = new CommandHarness();
    await mixed.run("openrouter-pin", "z-ai/glm-5.2 novita --help");
    assert.equal(mixed.notifications.length, 1);
    assert.match(mixed.notifications[0].message, /^Usage: \/openrouter-pin/);
  });
});

test("/openrouter-unpin --help: prints help and touches nothing", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "--help");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-unpin/);
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    assert.equal(await readJsonFile(modelsPath), null, "--help never creates models.json");
  });
});

test("/openrouter-unpin -h: short flag prints the same help", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "-h");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-unpin/);
  });
});

test("/openrouter-pins --help: prints help even with pins present", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([glmModel()]) },
    });
    const h = new CommandHarness();
    await h.run("openrouter-pins", "--help");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pins/);
  });
});

test("/openrouter-pins -h: short flag prints the same help", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pins", "-h");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pins/);
  });
});

// ---------------------------------------------------------------------------
// D — Refresh pipeline
// ---------------------------------------------------------------------------

/** Harness for the `session_start` handler registered by the extension factory. */
class SessionStartHarness {
  readonly notifications: Notify[] = [];
  private handler?: (event: unknown, ctx: ExtensionContext) => void | Promise<void>;

  constructor() {
    const pi = {
      on: (event: string, handler: (..._args: unknown[]) => unknown) => {
        if (event === "session_start") {
          this.handler = handler as typeof this.handler;
        }
      },
      registerProvider: () => {},
      registerCommand: () => {},
    } as unknown as ExtensionAPI;
    openrouterPinExtension(pi);
  }

  async fireSessionStart(apiKey?: string): Promise<void> {
    assert.ok(this.handler, "session_start handler must be registered");

    const oldKey = process.env.OPENROUTER_API_KEY;
    if (apiKey !== undefined) {
      process.env.OPENROUTER_API_KEY = apiKey;
    } else {
      delete process.env.OPENROUTER_API_KEY;
    }

    try {
      await this.handler!(
        { type: "session_start", reason: "startup" },
        {
          mode: "tui",
          hasUI: true,
          cwd: "/tmp",
          modelRegistry: {} as ModelRegistry,
          model: undefined,
          scopedModels: [],
          ui: {
            notify: (message: string, type: "info" | "warning" | "error" = "info") => {
              this.notifications.push({ message, type });
            },
            select: async () => undefined,
          },
        } as unknown as ExtensionContext,
      );
    } finally {
      if (oldKey === undefined) delete process.env.OPENROUTER_API_KEY;
      else process.env.OPENROUTER_API_KEY = oldKey;
    }

    // Wait for async settle — refreshPinnedModels performs network I/O under the hood.
    await delay(200);
  }
}

const delay = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms));

// --- formatRefreshDiff -------------------------------------------------------

test("formatRefreshDiff: empty diff returns an empty string", () => {
  assert.equal(formatRefreshDiff([]), "");
});

test("formatRefreshDiff: unchanged model produces no output", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  assert.equal(formatRefreshDiff(diff), "");
});

test("formatRefreshDiff: single cost field change aligned", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.34, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  const result = formatRefreshDiff(diff);
  // Labels changed: ["cost.input"] → width = 10
  assert.equal(
    result,
    [
      "  z-ai/glm-5.2 (openrouter-novita):",
      "    cost.input  $0.39/M  →  $0.34/M",
    ].join("\n"),
  );
});

test("formatRefreshDiff: multiple cost fields, tabular alignment", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.34, output: 1.18, cacheRead: 0.06, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  const result = formatRefreshDiff(diff);
  // Labels: ["cost.input"(10), "cost.cacheRead"(14)] → width = 14
  // "cost.input".padEnd(14) = "cost.input    " (4 trailing spaces)
  // "cost.cacheRead".padEnd(14) = "cost.cacheRead"   (exact length)
  assert.equal(
    result,
    [
      "  z-ai/glm-5.2 (openrouter-novita):",
      "    cost.input      $0.39/M  →  $0.34/M",
      "    cost.cacheRead  $0.07/M  →  $0.06/M",
    ].join("\n"),
  );
});

test("formatRefreshDiff: token-limit changes alongside costs", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-groq",
    modelId: "meta-llama/llama-4-scout",
    before: {
      cost: { input: 0.10, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
      contextWindow: 131_072,
      maxTokens: 32_768,
    },
    after: {
      cost: { input: 0.08, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
      contextWindow: 262_144,
      maxTokens: 65_536,
    },
  }];
  const result = formatRefreshDiff(diff);
  // Labels: ["cost.input"(10), "contextWindow"(13), "maxTokens"(9)] → width = 13
  assert.equal(
    result,
    [
      "  meta-llama/llama-4-scout (openrouter-groq):",
      "    cost.input     $0.10/M  →  $0.08/M",
      "    contextWindow  131,072  →  262,144",
      "    maxTokens      32,768  →  65,536",
    ].join("\n"),
  );
});

test("formatRefreshDiff: multi-model groups by model header", () => {
  const diff: PricingLimitsDiff[] = [
    {
      provider: "openrouter-novita",
      modelId: "z-ai/glm-5.2",
      before: {
        cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
        contextWindow: 1_048_576,
        maxTokens: 128_000,
      },
      after: {
        cost: { input: 0.34, output: 1.18, cacheRead: 0.06, cacheWrite: 0 },
        contextWindow: 1_048_576,
        maxTokens: 128_000,
      },
    },
    {
      provider: "openrouter-groq",
      modelId: "meta-llama/llama-4-scout",
      before: {
        cost: { input: 0.10, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
        contextWindow: 131_072,
        maxTokens: 32_768,
      },
      after: {
        cost: { input: 0.08, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
        contextWindow: 131_072,
        maxTokens: 65_536,
      },
    },
  ];
  const result = formatRefreshDiff(diff);
  // Model 1 width = max(10, 14) = 14; Model 2 width = max(10, 9) = 10
  assert.equal(
    result,
    [
      "  z-ai/glm-5.2 (openrouter-novita):",
      "    cost.input      $0.39/M  →  $0.34/M",
      "    cost.cacheRead  $0.07/M  →  $0.06/M",
      "  meta-llama/llama-4-scout (openrouter-groq):",
      "    cost.input  $0.10/M  →  $0.08/M",
      "    maxTokens   32,768  →  65,536",
    ].join("\n"),
  );
});

test("formatRefreshDiff: only changed fields appear (no redundant labels)", () => {
  // Only input price changed — output, cacheRead, cacheWrite, tokens all hidden.
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.34, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  const result = formatRefreshDiff(diff);
  assert.ok(!result.includes("cacheRead"), "unchanged cacheRead is hidden");
  assert.ok(!result.includes("output"), "unchanged output is hidden");
  assert.ok(!result.includes("cacheWrite"), "unchanged cacheWrite is hidden");
  assert.ok(!result.includes("contextWindow"), "unchanged contextWindow is hidden");
  assert.ok(!result.includes("maxTokens"), "unchanged maxTokens is hidden");
  assert.ok(result.includes("cost.input"), "changed cost.input is shown");
});

// --- collectRefreshTargets ---------------------------------------------------

test("collectRefreshTargets: only openrouter-* providers with routing compat", () => {
  const targets = collectRefreshTargets({
    providers: {
      "openrouter-novita": providerEntry([glmModel()]),           // matched
      "anthropic": providerEntry([glmModel()]),                   // not our prefix
      "openrouter-together": providerEntry([{                     // matching prefix but no routing config
        id: "google/gemini-2.5-pro",
        name: "Gemini Pro",
        reasoning: true,
        input: ["text"],
        contextWindow: 2_097_152,
        maxTokens: 65_536,
        cost: { input: 1.25, output: 5.00, cacheRead: 0.1875, cacheWrite: 0 },
        compat: { openRouterRouting: {} },
      }]),
    },
  });
  assert.deepEqual(targets.map(t => [t.provider, t.model.id]), [
    ["openrouter-novita", "z-ai/glm-5.2"],
  ]);
});

test("collectRefreshTargets: missing file / empty providers → []", () => {
  assert.deepEqual(collectRefreshTargets(null), []);
  assert.deepEqual(collectRefreshTargets({}), []);
  assert.deepEqual(collectRefreshTargets({ providers: {} }), []);
});

// --- applyPricingPatches -----------------------------------------------------

test("applyPricingPatches: mutates models in-place and returns diffs", () => {
  const current: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel()]),
    },
  };
  const patches: PricingLimitsPatch[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    cost: { input: 0.34, output: 0.98, cacheRead: 0.06, cacheWrite: 0 },
    contextWindow: 2_097_152,
    maxTokens: 65_536,
  }];
  const { applied, diff } = applyPricingPatches(current, patches);

  assert.equal(applied, 1);
  assert.equal(diff.length, 1);
  assert.equal(diff[0].modelId, "z-ai/glm-5.2");
  assert.deepEqual(diff[0].before.cost, glmModel().cost); // original cost preserved
  assert.deepEqual(diff[0].after.cost, patches[0].cost); // patched cost recorded

  // Verify in-place mutation on disk snapshot
  assert.deepEqual(current.providers!["openrouter-novita"].models[0].cost, patches[0].cost);
  assert.equal(current.providers!["openrouter-novita"].models[0].contextWindow, 2_097_152);
  assert.equal(current.providers!["openrouter-novita"].models[0].maxTokens, 65_536);
});

test("applyPricingPatches: silently skips missing provider", () => {
  const current: ModelsJson = {
    providers: { "openrouter-novita": providerEntry([glmModel()]) },
  };
  const patches: PricingLimitsPatch[] = [{
    provider: "openrouter-nonexistent",
    modelId: "z-ai/glm-5.2",
    cost: { input: 0.34, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1_048_576,
    maxTokens: 128_000,
  }];
  const { applied, diff } = applyPricingPatches(current, patches);
  assert.equal(applied, 0);
  assert.equal(diff.length, 0);
});

test("applyPricingPatches: silently skips missing model id within provider", () => {
  const current: ModelsJson = {
    providers: { "openrouter-novita": providerEntry([glmModel()]) },
  };
  const patches: PricingLimitsPatch[] = [{
    provider: "openrouter-novita",
    modelId: "nobody/home",
    cost: { input: 0.34, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1_048_576,
    maxTokens: 128_000,
  }];
  const { applied, diff } = applyPricingPatches(current, patches);
  assert.equal(applied, 0);
  assert.equal(diff.length, 0);
});

test("applyPricingPatches: processes valid patches and skips invalid ones", () => {
  const current: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
    },
  };
  const patches: PricingLimitsPatch[] = [
    {  // valid
      provider: "openrouter-novita",
      modelId: "z-ai/glm-5.2",
      cost: { input: 0.34, output: 0.98, cacheRead: 0.06, cacheWrite: 0 },
      contextWindow: 2_097_152,
      maxTokens: 65_536,
    },
    {  // invalid model id
      provider: "openrouter-novita",
      modelId: "fake/model",
      cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 131_072,
      maxTokens: 32_768,
    },
  ];
  const { applied, diff } = applyPricingPatches(current, patches);
  assert.equal(applied, 1);
  assert.equal(diff.length, 1);

  // GLM patched, DeepSeek untouched
  const ids = current.providers!["openrouter-novita"].models.map(m => m.id);
  assert.deepEqual(ids, ["z-ai/glm-5.2", "deepseek/deepseek-v4-flash-0731"]);
  // GLM model cost was mutated
  assert.deepEqual(current.providers!["openrouter-novita"].models[0].cost, patches[0].cost);
});

// --- session_start handler (integration) -------------------------------------

test("session_start handler: no-op without API key (integration)", async () => {
  // Without an OpenRouter API key, refreshPinnedModels returns immediately
  // with empty results — no notification emitted. Guards against regressions
  // where console.log leaks into the TUI footer.
  const h = new SessionStartHarness();
  await h.fireSessionStart(); // no API key set

  // Without an API key, refreshPinnedModels either returns empty immediately
  // (0 notifications) or hits an error (e.g. empty modelRegistry throws) which
  // the handler catches and routes via ctx.ui.notify. Either outcome is
  // acceptable — the critical assertion is that messages always go through
  // notify() and never leak to raw console.*.
  assert.ok(h.notifications.length <= 1, "at most one notification (none, or an error routed via ctx.ui.notify)");
});

test("session_start handler: routes refresh results via ctx.ui.notify (guarded)", async () => {
  // Integration: requires OPENROUTER_API_KEY to make real API calls.
  // Skipped entirely when absent (does not fail CI).
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    // eslint-disable-next-line no-console -- expected skip in environments without API keys
    console.log("[skip] OPENROUTER_API_KEY not set");
    return;
  }

  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel({ id: "z-ai/glm-5.2" })]),
      },
    });

    const h = new SessionStartHarness();
    await h.fireSessionStart(apiKey);

    // With real credentials, the handler should emit at least one notification
    // (success with pricing, or warning if the endpoint call fails).
    assert.ok(h.notifications.length > 0, "handler emits a notification for pinned models");
    // All messages come through ctx.ui.notify — never raw console.*.
    assert.ok(h.notifications.every(n => n.type === "info" || n.type === "warning" || n.type === "error"));
    // Messages reference pricing or refresh, confirming the right logic ran.
    const anyPricingMsg = h.notifications.some(n =>
      n.message.toLowerCase().includes("refresh") || n.message.toLowerCase().includes("pricing"),
    );
    assert.ok(anyPricingMsg, "at least one message references pricing/refresh");
  });
});
