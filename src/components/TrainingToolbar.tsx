import type { PlaygroundProps } from "../types.ts";

/** Shared run controls reflect the hook state, so every page controls the same experiment. */
export default function TrainingToolbar({ playground: p }: PlaygroundProps) {
  const s = p.snapshot;
  return (
    <section className="training-toolbar" aria-label="Controlli simulazione">
      <button
        id="train"
        onClick={p.train}
        disabled={!p.ready || p.busy || p.running || Boolean(s.issue)}
      >
        {s.iteration >= s.total
          ? "↻ Riavvia simulazione"
          : "▶ Avvia / Continua"}
      </button>
      <button
        id="step"
        className="secondary"
        disabled={!p.ready || p.busy || p.running || Boolean(s.issue)}
        onClick={p.step}
      >
        ⏭ Un passo alla volta
      </button>
      <button
        id="trainingPause"
        className="secondary"
        disabled={!p.running}
        onClick={p.pause}
      >
        ⏸ Pausa
      </button>
      <div
        className="progress training-toolbar-progress"
        role="progressbar"
        aria-label="Training"
        aria-valuemin={0}
        aria-valuemax={s.total}
        aria-valuenow={s.iteration}
      >
        <div
          style={{ width: `${Math.min(100, (s.iteration / s.total) * 100)}%` }}
        />
      </div>
      <div
        id="status"
        className="training-toolbar-status small muted"
        role="status"
      >
        {(!p.ready ? "Preparazione solver?" : s.issue) ||
          `${p.running ? "Training in corso" : s.iteration >= s.total ? "Simulazione completata" : s.iteration ? "In pausa" : "Pronto"} · passo ${s.iteration} / ${s.total}`}
      </div>
      {p.error && (
        <div role="alert" className="error">
          {p.error}
        </div>
      )}
    </section>
  );
}
