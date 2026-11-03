import type { EngineState, Model } from "../types.ts";

/**
 * Fitting and prediction require an initialized family-specific model. Keeping
 * this invariant at a single boundary avoids non-null assertions throughout
 * numerical code and produces an actionable runtime error for invalid calls.
 */
export function requireModel(state: EngineState): Model {
  if (!state.model)
    throw new Error("Initialize the model before using its parameters.");
  return state.model;
}

/**
 * Mark a model-family branch as exhaustive. If a new Model member is introduced,
 * TypeScript rejects callers until they handle it explicitly. The runtime throw
 * remains a defensive boundary for values supplied by untyped integrations.
 */
export function assertNever(_model: never): never {
  throw new Error("Unhandled model family.");
}

/**
 * Previous-step explanations must use the same family as the current model.
 * A kind check both protects that invariant at runtime and narrows the union
 * for TypeScript. The sole return assertion bridges a compiler limitation:
 * TypeScript cannot reduce a generic Extract from the runtime kind comparison.
 * The preceding check establishes that invariant before the assertion.
 */
export function modelOfKind<K extends Model["kind"]>(
  model: Model | null,
  kind: K,
): Extract<Model, { kind: K }> {
  if (!model || model.kind !== kind)
    throw new Error(`Expected a ${kind} model.`);
  return model as Extract<Model, { kind: K }>;
}
