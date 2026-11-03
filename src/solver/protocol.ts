import type {
  Configuration,
  Dataset,
  LearningTrace,
  LearningVisualization,
  NeuralDiagnostics,
  NeuralVisualization,
  PageId,
  Snapshot,
  Matrix,
  Vector,
} from "../types.ts";

/**
 * The worker owns the mutable engine. Only serializable commands and read-only
 * presentation frames cross this boundary; React never executes a solver step.
 * Control IDs order run/pause/step commands independently of render frame IDs.
 */
export type SolverCommand =
  | {
      type: "initialize";
      dataset: Dataset;
      config: Configuration;
      page: PageId;
    }
  | { type: "run" | "pause" | "step"; controlId: number }
  | { type: "page"; page: PageId }
  | { type: "inspect"; index: number }
  | { type: "predict"; requestId: number; input: Vector }
  | { type: "ack"; frameId: number };

/** World-space samples let pan/zoom redraw immediately without rerunning prediction. */
export interface PredictionGrid {
  cells: { input: Vector; value: number }[];
  stepX: number;
  stepY: number;
  three: boolean;
}

export interface SolverFrame {
  frameId: number;
  controlId: number;
  running: boolean;
  snapshot: Snapshot;
  /** Row indices avoid duplicating datasets in messages and restore identity in React. */
  trainIndices: number[];
  testIndices: number[];
  predictions: Vector;
  parameters: [string, number][];
  centers: Matrix;
  plot: PredictionGrid | null;
  trace: LearningTrace;
  neural: NeuralVisualization | null;
  diagnostics: NeuralDiagnostics | null;
  diagnosticsIteration: number | null;
  inspection: LearningVisualization | null;
  inspectionIndex: number;
}

export type SolverEvent =
  | { type: "frame"; frame: SolverFrame }
  | { type: "prediction"; requestId: number; value: number | null }
  | { type: "error"; controlId: number; message: string };

/** A new experiment renders its data immediately while its worker starts up. */
export function emptyFrame(): SolverFrame {
  return {
    frameId: 0,
    controlId: 0,
    running: false,
    snapshot: {
      iteration: 0,
      fitted: false,
      loss: NaN,
      losses: [],
      trainHistory: [],
      testHistory: [],
      metrics: {},
      model: null,
      train: [],
      test: [],
      regression: false,
      clustering: false,
      total: 1,
      issue: "",
    },
    trainIndices: [],
    testIndices: [],
    predictions: [],
    parameters: [],
    centers: [],
    plot: null,
    trace: { status: "Preparazione dei calcoli…", steps: [] },
    neural: null,
    diagnostics: null,
    diagnosticsIteration: null,
    inspection: null,
    inspectionIndex: 0,
  };
}
