import {
  METHOD_INFO,
  METHOD_SYMBOLS,
  COMMON_METHOD_SYMBOLS,
} from "../data/methods.js";
import { Panel, Stat, format } from "./Panel.jsx";

/** Formula strings are local reference data; render text and line breaks as React nodes. */
function Formula({ text, symbols }) {
  const glossary = new Map(
    [...COMMON_METHOD_SYMBOLS, ...symbols].flatMap(([term, meaning]) =>
      term.split(/,\s*/).map((t) => [t, meaning]),
    ),
  );
  return (
    <div className="formula">
      {text.split(/<br\s*\/?\s*>/i).map((line, index) => (
        <div key={index}>
          {line.split(/(\s+|[=+←→])/).map((token, i) =>
            glossary.has(token) ? (
              <span
                key={i}
                className="formula-symbol"
                tabIndex={0}
                title={glossary.get(token)}
              >
                {token}
              </span>
            ) : (
              token
            ),
          )}
        </div>
      ))}
    </div>
  );
}
export default function MethodReference({ playground: p }) {
  const info = METHOD_INFO[p.config.method],
    symbols = METHOD_SYMBOLS[p.config.method] || [];
  return (
    <Panel title="Metodo di apprendimento · formule e valutazioni">
      <p className="note">{info.status}</p>
      <p>{info.explain}</p>
      <div className="reference-grid">
        {[
          ["Formula della previsione / modello", info.model],
          ["Funzione obiettivo / costo", info.cost],
          ["Aggiornamento dei parametri", info.update],
        ].map(([title, text]) => (
          <Panel key={title} title={title}>
            <Formula text={text} symbols={symbols} />
          </Panel>
        ))}
      </div>
      <details open>
        <summary>Legenda delle formule · simboli e funzione</summary>
        <ul className="symbol-legend">
          {symbols.map(([symbol, meaning], i) => (
            <li key={i}>
              <strong tabIndex={0} title={meaning}>
                {symbol}
              </strong>
              <span>{meaning}</span>
            </li>
          ))}
        </ul>
      </details>
      <h3>Metriche sul set di validazione</h3>
      <div className="statrow metrics">
        {Object.entries(p.snapshot.metrics)
          .filter(
            ([, value]) => typeof value !== "number" || Number.isFinite(value),
          )
          .map(([key, value]) => (
            <Stat key={key} label={key} value={format(value)} />
          ))}
      </div>
      {!p.snapshot.fitted && (
        <p className="small muted">
          Avvia il training per calcolare le metriche sul set separato di
          validazione.
        </p>
      )}
    </Panel>
  );
}
