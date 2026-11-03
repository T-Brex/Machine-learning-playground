import type { PredictionGrid } from "../solver/protocol.ts";
import type {
  ChartView,
  Matrix,
  Dataset,
  NeuralDiagnostics,
  NeuralVisualization,
  Snapshot,
  Vector,
} from "../types.ts";

// Canvas drawing stays imperative, but receives data and never queries or changes the DOM.
export const COLORS = [
  "#2995ff",
  "#ff9d37",
  "#35d58a",
  "#d879ff",
  "#f5cf4b",
  "#49d8d0",
  "#ff657a",
  "#a7a7ff",
];
export const classColor = (label: number) =>
  label < 0 ? "#8192a8" : COLORS[label % COLORS.length];

/** Return data-to-pixel scales and draw labeled ticks inside a consistent plotting margin. */

function axes(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  xLabel: string,
  yLabel: string,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
) {
  const left = 58,
    right = width - 18,
    top = 24,
    bottom = height - 44;
  ctx.font = "10px system-ui";
  ctx.fillStyle = "#9db6d4";
  ctx.strokeStyle = "#234565";
  ctx.lineWidth = 1;
  const x = (v: number) =>
      left + ((v - xMin) / (xMax - xMin || 1)) * (right - left),
    y = (v: number) =>
      bottom - ((v - yMin) / (yMax - yMin || 1)) * (bottom - top);
  for (let i = 0; i < 5; i++) {
    const a = xMin + ((xMax - xMin) * i) / 4,
      b = yMin + ((yMax - yMin) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x(a), top);
    ctx.lineTo(x(a), bottom);
    ctx.moveTo(left, y(b));
    ctx.lineTo(right, y(b));
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(String(Number(a.toFixed(2))), x(a), bottom + 16);
    ctx.textAlign = "right";
    ctx.fillText(String(Number(b.toFixed(3))), left - 7, y(b) + 3);
  }
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (left + right) / 2, height - 7);
  ctx.save();
  ctx.translate(12, (top + bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.textAlign = "left";
  return { x, y, left, right, top, bottom };
}

/** Plot finite history values; logarithmic loss uses log10 and keeps zeros above a small floor. */

export function drawLines(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  series: { values: Vector; color: string; label: string }[],
  { log = false, yLabel = "Loss" } = {},
) {
  const clean = series.flatMap((s) => s.values).filter(Number.isFinite);
  const transform = (v: number) => (log ? Math.log10(Math.max(1e-8, v)) : v);
  const values = clean.map(transform),
    lo = Math.min(0, ...values),
    hi = Math.max(log ? lo + 1 : 1, ...values);
  const count = Math.max(2, ...series.map((s) => s.values.length));
  const a = axes(
    ctx,
    w,
    h,
    "Epoca",
    log ? "log₁₀(loss)" : yLabel,
    1,
    count,
    lo,
    hi,
  );
  for (const { values: line, color, label } of series) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    line.forEach((v: number, i: number) => {
      if (!Number.isFinite(v)) {
        started = false;
        return;
      }
      const px = a.x(i + 1),
        py = a.y(transform(v));
      if (started) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      started = true;
    });
    ctx.stroke();
    if (line.length === 1 && Number.isFinite(line[0])) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(a.x(1), a.y(transform(line[0])), 3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = color;
    ctx.fillText(
      label,
      a.left + series.indexOf(series.find((s) => s.values === line)!) * 110,
      14,
    );
  }
  if (!clean.length) {
    ctx.fillStyle = "#9db6d4";
    ctx.fillText(
      "Avvia il training per vedere le curve",
      a.left + 8,
      a.top + 22,
    );
  }
}

/** Bin completed training losses to show how frequently each loss range occurred. */

export function drawHistogram(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  losses: Vector,
) {
  const values = losses.filter(Number.isFinite),
    lo = Math.min(...values),
    hi = Math.max(...values),
    bins = Array<number>(12).fill(0);
  for (const value of values)
    bins[Math.min(11, Math.floor(((value - lo) / (hi - lo || 1)) * 12))]++;
  const a = axes(
    ctx,
    w,
    h,
    "Loss",
    "Numero di epoche",
    values.length ? lo : 0,
    values.length ? hi + (hi === lo ? 1 : 0) : 1,
    0,
    Math.max(1, ...bins),
  );
  bins.forEach((n, i: number) => {
    ctx.fillStyle = COLORS[0];
    const bw = (a.right - a.left) / 12;
    ctx.fillRect(a.left + i * bw + 1, a.y(n), bw - 2, a.bottom - a.y(n));
  });
}

/** Spatial projection uses only visible dimensions; predictions retain every feature. */
export function drawScatter(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  {
    dataset,
    plot,
    predictions,
    centers,
    snapshot,
    view,
    selected = -1,
    hiddenClasses = [],
  }: {
    dataset: Dataset;
    plot: PredictionGrid | null;
    predictions: Vector;
    centers: Matrix;
    snapshot: Snapshot;
    view: ChartView;
    selected?: number;
    hiddenClasses?: number[];
  },
) {
  const rows = dataset.data,
    three = (rows[0]?.x.length || 2) >= 3;
  const min = [0, 1, 2].map(
      (i: number) => Math.min(-2, ...rows.map((r) => r.x[i] || 0)) - 0.3,
    ),
    max = [0, 1, 2].map(
      (i: number) => Math.max(2, ...rows.map((r) => r.x[i] || 0)) + 0.3,
    );
  // The foreground, prediction grid, and axis ticks share the same inverse projection.
  const centerX = (w + 40) / 2,
    centerY = (h - 20) / 2;
  const plotWidth = Math.max(1, w - 76),
    plotHeight = Math.max(1, h - 68);
  const visibleX = (pixel: number) =>
    min[0] +
    ((pixel - centerX - view.panX) / (plotWidth * view.zoom) + 0.5) *
      (max[0] - min[0]);
  const visibleY = (pixel: number) =>
    min[1] +
    ((centerY + view.panY - pixel) / (plotHeight * view.zoom) + 0.5) *
      (max[1] - min[1]);
  const project = (x: Vector): [number, number] => {
    const n = [0, 1, 2].map(
      (i: number) => ((x[i] || 0) - min[i]) / (max[i] - min[i]) - 0.5,
    );
    if (!three)
      return [
        centerX + n[0] * plotWidth * view.zoom + view.panX,
        centerY - n[1] * plotHeight * view.zoom + view.panY,
      ];
    const u = n[0] * Math.cos(view.yaw) - n[1] * Math.sin(view.yaw),
      v = n[0] * Math.sin(view.yaw) + n[1] * Math.cos(view.yaw);
    return [
      w / 2 + u * w * 0.6 * view.zoom + view.panX,
      h * 0.55 +
        (v * Math.sin(view.pitch) - n[2] * Math.cos(view.pitch)) *
          h *
          0.65 *
          view.zoom +
        view.panY,
    ];
  };
  // Cells were evaluated in the worker. Reprojecting cached world-space data
  // keeps gestures independent of solver speed, with no inference during draw.
  if (snapshot.fitted && !three && plot) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(58, 24, plotWidth, plotHeight);
    ctx.clip();
    for (const cell of plot.cells) {
      const [x, y] = project(cell.input);
      const [right, top] = project([
        cell.input[0] + plot.stepX,
        cell.input[1] + plot.stepY,
      ]);
      ctx.fillStyle = snapshot.regression
        ? `hsl(${220 - Math.max(0, Math.min(1, (cell.value + 5) / 10)) * 180} 50% 23%)`
        : classColor(cell.value) + "35";
      ctx.fillRect(x, top, right - x + 1, y - top + 1);
    }
    ctx.restore();
  }
  if (three) {
    const origin = project([0, 0, 0]);
    ctx.strokeStyle = "#6b89a9";
    ctx.font = "11px system-ui";
    [0, 1, 2].forEach((i: number) => {
      const end = [0, 0, 0];
      end[i] = max[i];
      const p = project(end);
      ctx.beginPath();
      ctx.moveTo(...origin);
      ctx.lineTo(...p);
      ctx.stroke();
      ctx.fillStyle = "#b9d6f5";
      ctx.fillText(`x${i + 1}`, p[0], p[1]);
    });
    // The worker samples these slices once per published model, rather than
    // evaluating a fresh network/kernel for every pixel of every pointer move.
    if (snapshot.fitted && plot) {
      ctx.globalAlpha = 0.15;
      for (const cell of plot.cells) {
        if (hiddenClasses.includes(cell.value)) continue;
        const [x, y] = project(cell.input);
        ctx.fillStyle = snapshot.regression
          ? `hsl(${220 - Math.max(0, Math.min(1, (cell.value + 5) / 10)) * 180} 50% 40%)`
          : classColor(cell.value);
        ctx.fillRect(x, y, 7 * view.zoom, 7 * view.zoom);
      }
      ctx.globalAlpha = 1;
    }
  } else
    axes(
      ctx,
      w,
      h,
      "x₁",
      "x₂",
      visibleX(58),
      visibleX(w - 18),
      visibleY(h - 44),
      visibleY(24),
    );
  const points = rows.map((row, index) => ({ row, index, p: project(row.x) }));
  points
    .sort((a, b) => (a.row.x[2] || 0) - (b.row.x[2] || 0))
    .forEach(({ row, index, p }) => {
      const label =
        snapshot.clustering && snapshot.fitted
          ? (predictions[index] ?? -1)
          : row.y;
      ctx.fillStyle = snapshot.regression
        ? `hsl(${220 - Math.max(0, Math.min(1, (row.y + 5) / 10)) * 180} 80% 66%)`
        : classColor(label);
      ctx.beginPath();
      ctx.arc(...p, index === selected ? 6 : 3.5, 0, Math.PI * 2);
      ctx.fill();
      if (index === selected) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(...p, 9, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
  centers.forEach((center, i: number) => {
    const [x, y] = project(center);
    ctx.strokeStyle = classColor(i);
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(x - 7, y);
    ctx.lineTo(x + 7, y);
    ctx.moveTo(x, y - 7);
    ctx.lineTo(x, y + 7);
    ctx.stroke();
  });
  return points.map(({ index, p }) => ({ index, x: p[0], y: p[1] }));
}

/** Draw one connection per weight, with sign encoded by color and magnitude by thickness. */

export function drawNetwork(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  network: NeuralVisualization | null,
) {
  if (!network) return;
  const layers = [
      network.inputValues,
      network.hiddenValues,
      network.outputValues,
    ],
    weights = [network.weightsInputHidden, network.weightsHiddenOutput];
  const point = (layer: number, i: number) => ({
    x: 45 + (layer * (w - 90)) / 2,
    y: 40 + ((i + 0.5) * (h - 80)) / layers[layer].length,
  });
  weights.forEach((matrix, l: number) =>
    matrix.forEach((row, j) =>
      row.forEach((weight, i: number) => {
        const a = point(l, i),
          b = point(l + 1, j);
        ctx.strokeStyle = weight >= 0 ? "#2995ff99" : "#ff9d3799";
        ctx.lineWidth = Math.min(5, 0.3 + Math.abs(weight) * 2);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }),
    ),
  );
  layers.forEach((layer, l: number) => {
    ctx.fillStyle = "#b9d6f5";
    ctx.font = "12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(["Input", "Hidden · tanh", "Output"][l], point(l, 0).x, 20);
    layer.forEach((value, i: number) => {
      const { x, y } = point(l, i);
      ctx.beginPath();
      ctx.fillStyle = "#123a65";
      ctx.strokeStyle = "#80bbff";
      ctx.lineWidth = 1;
      ctx.arc(x, y, 15, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.font = "10px system-ui";
      ctx.fillText(Number(value).toFixed(2), x, y + 3);
    });
  });
  ctx.textAlign = "left";
}

/** Color a two-weight objective slice; the white marker is the current parameter pair. */

export function drawLossSurface(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  diagnostics: NeuralDiagnostics,
) {
  const { surface, xMin, xMax, yMin, yMax } = diagnostics;
  const values = surface.flat(),
    lo = Math.min(...values),
    hi = Math.max(...values);
  const a = axes(
    ctx,
    w,
    h,
    "Primo peso W1[0,0]",
    "Deuxième poids W1[0,1]",
    xMin,
    xMax,
    yMin,
    yMax,
  );
  const cw = (a.right - a.left) / surface[0].length,
    ch = (a.bottom - a.top) / surface.length;
  surface.forEach((row, y) =>
    row.forEach((loss, x: number) => {
      const t = (loss - lo) / (hi - lo || 1);
      ctx.fillStyle = `hsl(${230 - 210 * t} 75% ${28 + 25 * t}%)`;
      ctx.fillRect(a.left + x * cw, a.bottom - (y + 1) * ch, cw + 1, ch + 1);
    }),
  );
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc((a.left + a.right) / 2, (a.top + a.bottom) / 2, 4, 0, Math.PI * 2);
  ctx.fill();
}
