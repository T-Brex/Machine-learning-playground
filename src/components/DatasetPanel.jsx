/** Controlled dataset inputs. Generating rows and importing files belong to usePlayground. */
import { DATASETS, MULTICLASS_DATASETS } from "../data/catalog.js";
import { Field, Panel, Stat } from "./Panel.jsx";

export default function DatasetPanel({ playground: p }) {
  const { config: c, dataset: d } = p;
  const options = DATASETS.filter(
    ([id]) => id.startsWith("regression") === (c.task === "regression"),
  );
  return (
    <Panel title="Dataset">
      <Field id="dataset" label="Dataset di esempio">
        <select
          id="dataset"
          value={c.dataset}
          onChange={(e) => p.update({ dataset: e.target.value }, true)}
        >
          {options.map(([id, label]) => (
            <option key={id} value={id}>
              {label}
            </option>
          ))}
        </select>
      </Field>
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
      <p className="note">
        {d.imported
          ? "CSV importato. Ultima colonna = target; fino a 8 feature numeriche."
          : options.find(([id]) => id === c.dataset)?.[1]}
      </p>
      <div className="statrow">
        <Stat label="Campioni" value={d.data.length} />
        <Stat label="Feature" value={d.data[0]?.x.length} />
        <Stat label="Classi" value={d.classes} />
      </div>
      <Field id="sampleCount" label="Numero di campioni nel dataset">
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
      <Field id="featureCount" label="Numero di feature (parametri di input)">
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
      <p className="small muted">
        I modelli usano tutte le feature. La vista spaziale proietta le prime
        due o tre.
      </p>
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
      <button
        className="secondary full"
        id="regen"
        onClick={() => p.update({}, true)}
      >
        ↻ Rigenera dataset
      </button>
      <Field id="csvFile" label="Carica CSV (opzionale)">
        <input
          id="csvFile"
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => {
            p.importCsv(e.target.files[0]);
            e.target.value = "";
          }}
        />
      </Field>
      <p className="small muted">
        Almeno due feature + target numerico. Intestazione facoltativa;
        separatori: virgola, punto e virgola, tab.
      </p>
    </Panel>
  );
}
