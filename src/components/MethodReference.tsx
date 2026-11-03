import { useTranslation } from "../i18n/Language.tsx";
import {
  COMMON_METHOD_SYMBOLS,
  METHOD_INFO,
  METHOD_SYMBOLS,
} from "../data/methods.ts";
import type { PlaygroundProps, SymbolDefinition } from "../types.ts";
import { Panel, Stat, format } from "./Panel.tsx";

/** Formula strings are local reference data; render text and line breaks as React nodes. */
function Formula({
  text,
  symbols,
}: {
  text: string;
  symbols: SymbolDefinition[];
}) {
  const t = useTranslation();

  const glossary = new Map<string, string>(
    [...COMMON_METHOD_SYMBOLS, ...symbols].flatMap(([term, meaning]) =>
      term.split(/,\s*/).map((t): [string, string] => [t, meaning]),
    ),
  );
  return (
    <div className="formula">
      {t(text)
        .split(/<br\s*\/?\s*>/i)
        .map((line, index) => (
          <div key={index}>
            {line.split(/(\s+|[=+←→])/).map((token, i) =>
              glossary.has(token) ? (
                <span
                  key={i}
                  className="formula-symbol"
                  tabIndex={0}
                  title={t(glossary.get(token))}
                >
                  {t(token)}
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
export default function MethodReference({ playground: p }: PlaygroundProps) {
  const t = useTranslation();

  const info = METHOD_INFO[p.config.method],
    symbols = METHOD_SYMBOLS[p.config.method] || [];
  return (
    <Panel title={t("Metodo di apprendimento · formule e valutazioni")}>
      <p className="note">{t(info.status)}</p>
      <p>{t(info.explain)}</p>
      <div className="reference-grid">
        {[
          ["Formula della previsione / modello", info.model],
          ["Funzione obiettivo / costo", info.cost],
          ["Aggiornamento dei parametri", info.update],
        ].map(([title, text]) => (
          <Panel key={title} title={t(title)}>
            <Formula text={text} symbols={symbols} />
          </Panel>
        ))}
      </div>
      <details open>
        <summary>{t("Legenda delle formule · simboli e funzione")}</summary>
        <ul className="symbol-legend">
          {symbols.map(([symbol, meaning], i) => (
            <li key={i}>
              <strong tabIndex={0} title={t(meaning)}>
                {t(symbol)}
              </strong>
              <span>{t(meaning)}</span>
            </li>
          ))}
        </ul>
      </details>
      <h3>{t("Metriche sul set di validazione")}</h3>
      <div className="statrow metrics">
        {Object.entries(p.snapshot.metrics)
          .filter(
            ([, value]) => typeof value !== "number" || Number.isFinite(value),
          )
          .map(([key, value]) => (
            <Stat key={key} label={t(key)} value={format(value)} />
          ))}
      </div>
      {!p.snapshot.fitted && (
        <p className="small muted">
          {t(
            "Avvia il training per calcolare le metriche sul set separato di validazione.",
          )}
        </p>
      )}
    </Panel>
  );
}
