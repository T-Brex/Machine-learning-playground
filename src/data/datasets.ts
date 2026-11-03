import { randn } from "../engine/math.ts";
import type { Configuration, Dataset, Sample } from "../types.ts";

/** Generate synthetic rows; charts may project them, but models retain all features. */
export function generateDataset(config: Configuration): Dataset {
  let data: Sample[] = [],
    classes = 2;
  const kind = config.dataset,
    noise = Number(config.noise),
    n = Math.max(
      20,
      Math.min(1000, Math.round(Number(config.sampleCount) || 120)),
    );
  const requestedClasses = Math.max(
    2,
    Math.min(8, Number(config.classCount) || 3),
  );
  // Continuous targets share a feature domain, with different nonlinear relationships.
  if (kind.startsWith("regression")) {
    for (let i = 0; i < n; i++) {
      let x1 = Math.random() * 4 - 2,
        x2 = Math.random() * 4 - 2,
        y;
      if (kind === "regression_quadratic")
        y = 1.1 * x1 * x1 + 0.65 * x2 * x2 - 0.8 * x1 + 0.4;
      else if (kind === "regression_sine")
        y = 2 * Math.sin(1.35 * x1) + 0.55 * x2;
      else if (kind === "regression_interaction")
        y = 1.25 * x1 * x2 + 0.7 * x1 - 0.45 * x2;
      else y = 2.2 * x1 - 1.35 * x2 + 0.6;
      data.push({ x: [x1, x2], y: y + randn() * noise });
    }
    classes = 1;
  } else if (kind === "linear_multiclass") {
    classes = requestedClasses;
    for (let i = 0; i < n; i++) {
      const x = [Math.random() * 4 - 2, Math.random() * 4 - 2],
        scores = Array.from(
          { length: classes },
          (_, c) =>
            x[0] * Math.cos((c * 2 * Math.PI) / classes) +
            x[1] * Math.sin((c * 2 * Math.PI) / classes),
        ),
        label = scores.indexOf(Math.max(...scores));
      data.push({ x, y: label });
    }
  } else if (kind === "threeblobs" || kind === "multiclass_blobs") {
    classes = requestedClasses;
    for (let i = 0; i < n; i++) {
      const y = i % classes,
        angle = (y * 2 * Math.PI) / classes,
        center = [1.35 * Math.cos(angle), 1.35 * Math.sin(angle)];
      data.push({
        x: [
          center[0] + (randn() * 0.38 * noise) / 0.35,
          center[1] + (randn() * 0.38 * noise) / 0.35,
        ],
        y,
      });
    }
  } else if (kind === "spiral3" || kind === "multiclass_spiral") {
    classes = requestedClasses;
    for (let i = 0; i < n; i++) {
      const y = i % classes,
        t = (Math.floor(i / classes) / Math.ceil(n / classes)) * Math.PI * 4,
        r = 0.12 + 0.105 * t,
        angle = t + (y * 2 * Math.PI) / classes;
      data.push({
        x: [
          r * Math.cos(angle) + randn() * noise * 0.35,
          r * Math.sin(angle) + randn() * noise * 0.35,
        ],
        y,
      });
    }
  } else if (kind === "linear") {
    for (let i = 0; i < n; i++) {
      let x1 = Math.random() * 4 - 2,
        x2 = Math.random() * 4 - 2;
      data.push({
        x: [x1, x2],
        y: 1.4 * x1 - 0.7 * x2 + 0.25 + randn() * noise > 0 ? 1 : 0,
      });
    }
    classes = 2;
  } else if (kind === "xor") {
    for (let i = 0; i < n; i++) {
      let a = Math.random() < 0.5 ? -1 : 1,
        c = Math.random() < 0.5 ? -1 : 1;
      data.push({
        x: [a * 0.8 + randn() * noise, c * 0.8 + randn() * noise],
        y: a * c > 0 ? 1 : 0,
      });
    }
    classes = 2;
  } else if (kind === "circles") {
    for (let i = 0; i < n; i++) {
      let y = i % 2,
        ang = Math.random() * Math.PI * 2,
        r = (y ? 0.95 : 0.4) + randn() * noise * 0.35;
      data.push({ x: [r * Math.cos(ang), r * Math.sin(ang)], y });
    }
    classes = 2;
  } else if (kind === "moons") {
    for (let i = 0; i < n; i++) {
      let y = i % 2,
        ang = Math.random() * Math.PI,
        x = y
          ? [1 - Math.cos(ang), 0.45 - Math.sin(ang)]
          : [Math.cos(ang), Math.sin(ang)];
      data.push({ x: [x[0] + randn() * noise, x[1] + randn() * noise], y });
    }
    classes = 2;
  } else if (kind === "oneclass") {
    for (let i = 0; i < n; i++)
      data.push({ x: [randn() * 0.65, randn() * 0.65], y: 0 });
    classes = 1;
  } else {
    for (let i = 0; i < n; i++) {
      let y = i % 2,
        mu = y ? 1.1 : -1.1;
      data.push({
        x: [
          mu + (randn() * 0.55 * noise) / 0.35,
          mu * 0.5 + (randn() * 0.65 * noise) / 0.35,
        ],
        y,
      });
    }
    classes = 2;
  }
  // Additional dimensions keep the original two-dimensional pattern visible in the charts.
  const dims = Number(config.featureCount) || 2;
  data.forEach((d) => {
    for (let j = d.x.length; j < dims; j++) {
      let signal = classes > 1 ? (d.y - (classes - 1) / 2) * 0.45 : 0;
      d.x.push(signal + randn() * (0.35 + noise * 0.25));
    }
  });
  return { data, classes };
}
