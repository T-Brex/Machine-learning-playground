/** Controlled dataset inputs. Generating rows and importing files belong to usePlayground. */
import { DATASETS, MULTICLASS_DATASETS } from "../data/catalog.ts";
import type { DatasetId, PlaygroundProps } from "../types.ts";
import { Field, Panel } from "./Panel.tsx";

export default function DatasetPanel({ playground: p }: PlaygroundProps) {
  const { config: c, dataset: d } = p;
  const options = DATASETS.filter(
    ([id]) => id.startsWith("regression") === (c.task === "regression"),
  );
  return (
    <Panel
      title="Dataset"
      className="dataset-panel"
      headerAside={
        <div className="dataset-summary">
          <span>
            <b>{d.data.length}</b> campioni
          </span>
          <span>
            <b>{d.data[0]?.x.length}</b> feature
          </span>
          <span>
            <b>{d.classes}</b> classi
          </span>
        </div>
      }
    >
      <div className="dataset-source">
        <Field id="dataset" label="Dataset di esempio">
          <select
            id="dataset"
            value={c.dataset}
            onChange={(e) =>
              p.update({ dataset: e.target.value as DatasetId }, true)
            }
          >
            {options.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </Field>
      </div>
      {MULTICLASS_DATASETS.includes(c.dataset) && (
        <Field id="classCount" label="Numero di classi">
          <select
            id="classCount"
            value={c.classCount}
            onChange={(e) =>
              p.update({ classCount: Number(e.target.value) }, true)
            }
          >
            {[2, 3, 4, 5, 6, 7, 8].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </Field>
      )}
      {d.imported && (
        <p className="dataset-description small muted">
          CSV importato · ultima colonna = target
        </p>
      )}
      <div className="dataset-parameters">
        <Field id="sampleCount" label="Campioni">
          <input
            id="sampleCount"
            type="number"
            min="20"
            max="1000"
            disabled={d.imported}
            value={c.sampleCount}
            onChange={(e) => p.update({ sampleCount: e.target.value }, true)}
          />
        </Field>
        <Field id="featureCount" label="Feature">
          <select
            id="featureCount"
            disabled={d.imported}
            value={c.featureCount}
            onChange={(e) =>
              p.update({ featureCount: Number(e.target.value) }, true)
            }
          >
            {[2, 3, 4, 5, 6, 7, 8].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </Field>
        <Field
          id="noise"
          label={`Rumore dei dati · ${Number(c.noise).toFixed(2)}`}
        >
          <input
            id="noise"
            type="range"
            min="0"
            max="1.5"
            step=".05"
            value={c.noise}
            onChange={(e) => p.update({ noise: Number(e.target.value) }, true)}
          />
        </Field>
      </div>
      <p className="dataset-help small muted">
        I modelli usano tutte le feature. La vista spaziale proietta le prime
        due o tre.
      </p>
      <div className="dataset-actions">
        <button
          className="secondary full"
          id="regen"
          onClick={() => p.update({}, true)}
        >
          Rigenera dataset
        </button>
        <Field id="csvFile" label="Carica CSV (opzionale)">
          <input
            id="csvFile"
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => {
              p.importCsv(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </Field>
      </div>
      <p className="small muted">
        Almeno due feature + target numerico. Intestazione facoltativa;
        separatori: virgola, punto e virgola, tab.
      </p>
    </Panel>
  );
}
