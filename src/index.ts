/**
 * pi-openrouter-pin — entry point.
 *
 * Pin OpenRouter models to a specific provider (e.g. novita) as a dedicated,
 * persistent pi provider. Persistence is pi-native: the pin writes a provider
 * `openrouter-<provider>` (strict) or `openrouter-<provider>-plus` (relaxed:
 * fallbacks/order/ignore) into `~/.pi/agent/models.json`, which pi loads
 * itself at every startup — no in-memory state, no plugin dependency.
 *
 * A pin always means "at least prefer this provider": strict is the default;
 * --fallback, --order, and --ignore are explicit relaxations.
 *
 * Commands:
 *   /openrouter-pin <model> <provider> [--quant fp8] [--name "Display"] [--default]
 *                  [--order a,b,c] [--ignore a,b] [--fallback]
 *                  [--data-collection allow|deny]     one-shot, scriptable
 *   /openrouter-pin                                       interactive wizard
 *   /openrouter-unpin <model>                             one-shot
 *   /openrouter-unpin                                     pick from existing pins
 *   /openrouter-pins                                      list pins (verbose view)
 */
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CATALOG_CACHE_TTL_MS, ENDPOINT_CACHE_TTL_MS, OpenRouterClient, PROVIDER_PREFIX } from "./api.ts";
import { isHelpRequest, parsePinArgs, PIN_HELP, PINS_HELP, UNPIN_HELP } from "./args.ts";
import { makePinCompletions } from "./completions.ts";
import { formatRouting, formatRefreshDiff, listPins, performPin, performUnpin, refreshPinnedModels } from "./commands.ts";
import { providerNameFor } from "./config.ts";
import { stripJsonComments, type ModelsJson } from "./files.ts";
import { pickFromList } from "./ui.ts";
import { runWizard } from "./wizard.ts";
import { resolveOpenRouterApiKey } from "./api.ts";

/**
 * The OpenRouter API key from the environment or pi's stored `openrouter`
 * credential (`~/.pi/agent/auth.json`). Read directly because the extension
 * factory runs before any ModelRegistry exists; pin-time validation and the
 * startup refresh still use registry-based resolution.
 *
 * Pins are separate providers (`openrouter-<slug>`) whose models.json entry
 * only carries `apiKey: "$OPENROUTER_API_KEY"`. A key saved with `/login` is
 * stored under the built-in `openrouter` provider, so without this the pinned
 * provider would have no auth, its models would never become available, and
 * `enabledModels` would warn "No models match pattern".
 */
function readOpenRouterApiKey(agentDir: string): string | undefined {
  const env = process.env.OPENROUTER_API_KEY?.trim();
  if (env) return env;
  try {
    const raw = readFileSync(join(agentDir, "auth.json"), "utf-8");
    const auth = JSON.parse(raw) as Record<string, { key?: unknown } | undefined>;
    const key = auth?.openrouter?.key;
    return typeof key === "string" && key.trim() ? key.trim() : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Re-register every `openrouter-*` pin from models.json with the resolved
 * OpenRouter credential so pinned providers inherit the same auth as the
 * built-in `openrouter` provider. Only synchronous file reads happen here (no
 * network, no writes), so it is safe in the extension factory for invocations
 * that never start a session (`pi --list-models`, `--help`, RPC health
 * checks). Staying synchronous keeps command/event registration from being
 * delayed behind an await.
 */
function registerPinnedProviders(
  pi: ExtensionAPI,
  modelsPath: string,
  apiKey: string | undefined,
): void {
  let models: ModelsJson | null;
  try {
    models = JSON.parse(stripJsonComments(readFileSync(modelsPath, "utf-8"))) as ModelsJson;
  } catch {
    return; // a missing or malformed models.json must not break the whole extension
  }
  for (const [name, entry] of Object.entries(models?.providers ?? {})) {
    if (!name.startsWith(PROVIDER_PREFIX)) continue;
    if (!Array.isArray(entry?.models) || entry.models.length === 0) continue;
    pi.registerProvider(name, apiKey ? { ...entry, apiKey } : entry);
  }
}

export default function openrouterPinExtension(pi: ExtensionAPI) {
  const agentDir = getAgentDir();
  const modelsPath = join(agentDir, "models.json");
  const settingsPath = join(agentDir, "settings.json");
  const client = new OpenRouterClient(CATALOG_CACHE_TTL_MS, ENDPOINT_CACHE_TTL_MS);

  // Pinned providers need the `openrouter` credential injected; see
  // readOpenRouterApiKey. Done synchronously in the factory so the models are
  // registered before startup continues and before `pi --list-models` prints.
  registerPinnedProviders(pi, modelsPath, readOpenRouterApiKey(agentDir));

  // Refresh pinned model pricing & limits (cost, contextWindow, maxTokens) at
  // session start. Deliberately NOT in the factory: factories run in
  // invocations that never start a session (pi --list-models, --help, RPC
  // health checks) and must not hit the network or rewrite models.json.
  // Fire-and-forget so startup is never blocked; failures are logged, not
  // swallowed. New values apply on the next /reload, which re-fires
  // session_start.
  pi.on("session_start", (_event, ctx) => {
    void refreshPinnedModels(modelsPath, client, () => resolveOpenRouterApiKey(ctx.modelRegistry))
      .then((r) => {
        if (r.refreshed > 0) {
          const diff = formatRefreshDiff(r.diff);
          if (diff) {
            ctx.ui.notify(
              `Refreshed ${r.refreshed} pinned model pricing & limits\n${diff}\n/reload to apply`,
              "info",
            );
          } else {
            ctx.ui.notify(`Refreshed ${r.refreshed} pinned model pricing & limits — /reload to apply`, "info");
          }
        } else if (r.failed.length > 0) {
          ctx.ui.notify(
            `Pricing & limits refresh unavailable for ${r.failed.length} model(s): ${r.failed.join(", ")}`,
            "warning",
          );
        }
      })
      .catch((err) => {
        ctx.ui.notify(
          `Pricing & limits refresh failed: ${err instanceof Error ? err.message : String(err)}`,
          "error",
        );
      });
  });

  const pinCompletions = makePinCompletions(client, modelsPath);

  pi.registerCommand("openrouter-pin", {
    description:
      "Pin an OpenRouter model to a specific provider (persistent, models.json). " +
      "No args opens an interactive wizard. With args: /openrouter-pin <model-id> <provider> " +
      "[--quant q] [--name 'Display'] [--default] [--order a,b,c] [--ignore a,b] [--fallback] [--data-collection allow|deny]",
    getArgumentCompletions: (prefix) => pinCompletions(prefix),
    handler: async (args, ctx: ExtensionCommandContext) => {
      if (isHelpRequest(args)) {
        ctx.ui.notify(PIN_HELP, "info");
        return;
      }
      if (!args.trim()) {
        await runWizard(modelsPath, settingsPath, pi, ctx.ui, client, ctx.modelRegistry);
        return;
      }
      const parsed = parsePinArgs(args);
      if ("error" in parsed) {
        ctx.ui.notify(parsed.error, "error");
        return;
      }
      await performPin(modelsPath, settingsPath, pi, ctx.ui, client, () =>
        resolveOpenRouterApiKey(ctx.modelRegistry, providerNameFor(parsed.slug!, parsed)), {
        modelId: parsed.modelId!,
        slug: parsed.slug!,
        quant: parsed.quant,
        // Empty/quoted-whitespace names fall back to the generated one, same
        // as the wizard ("Z.ai: GLM 5.2 (novita)") — never a blank picker entry.
        name: parsed.name?.trim() || undefined,
        isDefault: parsed.isDefault,
        allowFallbacks: parsed.allowFallbacks,
        order: parsed.order,
        ignore: parsed.ignore,
        dataCollection: parsed.dataCollection,
      });
    },
  });

  pi.registerCommand("openrouter-unpin", {
    description:
      "Remove an OpenRouter provider pin from models.json. No args picks from existing pins. " +
      "With args: /openrouter-unpin <model-id> (applies on /reload or next session)",
    handler: async (args, ctx: ExtensionCommandContext) => {
      if (isHelpRequest(args)) {
        ctx.ui.notify(UNPIN_HELP, "info");
        return;
      }
      try {
        let modelId = args.trim().split(/\s+/)[0];
        if (!modelId) {
          const pins = await listPins(modelsPath);
          if (pins.length === 0) {
            ctx.ui.notify("No pins to remove. Use /openrouter-pin to create one.", "info");
            return;
          }
          const chosen = await pickFromList(
            ctx.ui,
            "Pick a pin to remove",
            pins.map((p) => `${p.provider}/${p.model.id}`),
          );
          if (!chosen) {
            return; // Esc: cancel quietly, no notification
          }
          modelId = chosen.slice(chosen.indexOf("/") + 1);
        }
        const outcome = await performUnpin(modelsPath, modelId);
        if (outcome.status === "no-providers") {
          ctx.ui.notify("No pins found (no providers in models.json)", "info");
        } else if (outcome.status === "not-found") {
          ctx.ui.notify(`No pin for "${modelId}" found (checked openrouter-* providers)`, "info");
        } else {
          ctx.ui.notify(`Unpinned ${modelId} from models.json (applies on /reload or next session).`, "info");
        }
      } catch (err) {
        ctx.ui.notify(`Unpin failed: ${err instanceof Error ? err.message : String(err)}`, "error");
      }
    },
  });

  pi.registerCommand("openrouter-pins", {
    description: "List all pinned OpenRouter provider routes from models.json",
    handler: async (args, ctx: ExtensionCommandContext) => {
      if (isHelpRequest(args)) {
        ctx.ui.notify(PINS_HELP, "info");
        return;
      }
      try {
        const pins = await listPins(modelsPath);
        ctx.ui.notify(
          pins.length
            ? `Active pins:\n${pins
                .map((p) => {
                  const r = p.model.compat!.openRouterRouting;
                  return `  ${p.provider}/${p.model.id}  → ${formatRouting(r)}`;
                })
                .join("\n")}`
            : "No pins. Run /openrouter-pin (wizard) or /openrouter-pin <model> <provider> [--quant q] [--default] [--fallback]",
          "info",
        );
      } catch (err) {
        ctx.ui.notify(`List failed: ${err instanceof Error ? err.message : String(err)}`, "error");
      }
    },
  });
}
