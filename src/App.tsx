import { useTranslation } from "./i18n/Language.tsx";
/** Application composition: React conditionally mounts each page instead of moving DOM nodes. */
import { useState } from "react";
import DatasetPanel from "./components/DatasetPanel.tsx";
import ThemeColors from "./components/ThemeColors.tsx";
import LanguageSelector from "./components/LanguageSelector.tsx";
import {
  DataChart,
  Histogram,
  LossCharts,
  NetworkPanel,
} from "./components/LearningCharts.tsx";
import MethodReference from "./components/MethodReference.tsx";
import ModelPanel from "./components/ModelPanel.tsx";
import {
  PredictionPanel,
  SamplesPanel,
  WeightsPanel,
} from "./components/ResultsPanels.tsx";
import TrainingToolbar from "./components/TrainingToolbar.tsx";
import Tutorial from "./components/Tutorial.tsx";
import { NAVIGATION, TASKS } from "./data/catalog.ts";
import { usePlayground } from "./hooks/usePlayground.ts";
import type { PageId, TaskId } from "./types.ts";

export default function App() {
  const t = useTranslation();

  const [page, setPage] = useState<PageId>("home");
  const p = usePlayground(page);
  const settings = ["home", "data", "train", "method", "tutorial"].includes(
    page,
  );
  const charts = ["home", "train", "visual"].includes(page);
  return (
    <>
      <header className="top">
        <div className="logo" aria-hidden="true">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
            <path
              d="M5 4v19h19M8 19l5-7 5 3 6-9"
              stroke="currentColor"
              strokeWidth="1.5"
            />
            <circle cx="13" cy="12" r="2" fill="currentColor" />
            <circle cx="18" cy="15" r="2" fill="currentColor" />
          </svg>
        </div>
        <div>
          <div className="brand">{t("ML Playground")}</div>
          <div className="sub">{t("Sperimenta · Addestra · Visualizza")}</div>
        </div>
        <div className="header-description muted small">
          {t("Laboratorio interattivo di Machine Learning")}
        </div>
        <ThemeColors />
        <LanguageSelector />
      </header>
      <div className="layout">
        <aside>
          <nav aria-label={t("Navigazione principale")}>
            {NAVIGATION.map(([id, label]) => (
              <button
                key={id}
                className={page === id ? "active" : ""}
                aria-current={page === id ? "page" : undefined}
                onClick={() => setPage(id)}
              >
                {t(label)}
              </button>
            ))}
          </nav>
          <div className="asidehint">{t("Come iniziare")}</div>
          <p className="small muted">
            {t(
              "Scegli i dati, configura un algoritmo e avvia l’apprendimento. Osserva pesi, bias e grafici.",
            )}
          </p>
          <div className="note">
            {t(
              "Il modello modifica i parametri per ridurre una funzione di costo.",
            )}
          </div>
        </aside>
        <main className="main">
          <div className="tabs" aria-label={t("Tipo di problema")}>
            {(Object.entries(TASKS) as [TaskId, (typeof TASKS)[TaskId]][]).map(
              ([id, task]) => (
                <button
                  key={id}
                  className={p.config.task === id ? "active" : ""}
                  aria-pressed={p.config.task === id}
                  onClick={() => p.selectTask(id)}
                >
                  {t(task.label)}
                </button>
              ),
            )}
          </div>
          <TrainingToolbar playground={p} />
          {settings && (
            <div className="settings-grid">
              {["home", "data"].includes(page) && (
                <DatasetPanel playground={p} />
              )}
              <ModelPanel playground={p} />
              {page === "home" && <DataChart playground={p} />}
            </div>
          )}
          {page === "data" && (
            <div className="content-grid">
              <DataChart playground={p} />
              <SamplesPanel playground={p} />
            </div>
          )}
          {charts && (
            <>
              <div className="content-grid">
                <LossCharts playground={p} />
                {page === "visual" && <DataChart playground={p} />}
                <WeightsPanel playground={p} />
                <Histogram playground={p} />
                <NetworkPanel playground={p} />
              </div>
              <div className="content-grid">
                <SamplesPanel playground={p} />
                <PredictionPanel playground={p} />
              </div>
            </>
          )}
          {page === "method" && <MethodReference playground={p} />}
          {page === "tutorial" && (
            <Tutorial
              key={`${p.config.method}-${p.dataset.data.length}`}
              playground={p}
            />
          )}
          <footer className="footer">
            {t(
              "ML Playground · laboratorio didattico locale. Gli algoritmi funzionano nel browser: nessun dato viene inviato a un server.",
            )}
          </footer>
        </main>
      </div>
    </>
  );
}
