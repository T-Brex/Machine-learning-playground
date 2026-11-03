import type { Dataset, Vector } from "../types.ts";

export type Point3 = [number, number, number];
/** A face is model-space geometry, not a screen-space tile or a prediction slice. */
export interface DecisionFace {
  vertices: Point3[];
  /** The adjacent class IDs drive visibility toggles and the boundary's color. */
  classes: number[];
}
export interface SurfaceField {
  predict(input: Vector): number;
  /** Scores must preserve argmax decisions; raw logits/log-likelihoods work well. */
  scores(input: Vector): Vector | null;
  linear: boolean;
  regression: boolean;
  clustering: boolean;
}

const EPS = 1e-8;
const subtract = (a: Point3, b: Point3): Point3 => [
  a[0] - b[0],
  a[1] - b[1],
  a[2] - b[2],
];
const dot = (a: Point3, b: Point3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Point3, b: Point3): Point3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const interpolate = (a: Point3, b: Point3, t: number): Point3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];
const unique = (points: Point3[]) =>
  points.filter(
    (p, i) =>
      !points.slice(0, i).some((q) => Math.hypot(...subtract(p, q)) < EPS),
  );

/** Keep projection and geometry extraction in the same padded feature-space box. */
export function spatialDomain(dataset: Dataset): [Point3, Point3] {
  return [
    [0, 1, 2].map(
      (i) => Math.min(-2, ...dataset.data.map((row) => row.x[i] || 0)) - 0.3,
    ) as Point3,
    [0, 1, 2].map(
      (i) => Math.max(2, ...dataset.data.map((row) => row.x[i] || 0)) + 0.3,
    ) as Point3,
  ];
}

/**
 * Put coplanar intersections in winding order before filling/clipping them.
 * Edge discovery order is not polygon order: using it directly produces bow-tie
 * faces, particularly when an isosurface crosses four edges of a tetrahedron.
 */
function orderFace(points: Point3[], normal?: Point3): Point3[] {
  const vertices = unique(points);
  if (vertices.length < 3) return [];
  const center: Point3 = [0, 0, 0];
  for (const point of vertices)
    for (let axis = 0; axis < 3; axis++)
      center[axis] += point[axis] / vertices.length;
  const u = subtract(vertices[0], center);
  const n =
    normal ??
    cross(
      subtract(vertices[1], vertices[0]),
      subtract(vertices[2], vertices[0]),
    );
  if (Math.hypot(...n) < EPS) return [];
  const v = cross(n, u),
    ul = Math.hypot(...u),
    vl = Math.hypot(...v);
  if (ul < EPS || vl < EPS) return [];
  return vertices.sort((a, b) => {
    const angle = (p: Point3) =>
      Math.atan2(
        dot(subtract(p, center), v) / vl,
        dot(subtract(p, center), u) / ul,
      );
    return angle(a) - angle(b);
  });
}

/** Sutherland-Hodgman clipping in 3D, retaining the half-space value(point)>=0. */
function clip(polygon: Point3[], value: (point: Point3) => number): Point3[] {
  const result: Point3[] = [];
  let previous = polygon.at(-1);
  if (!previous) return result;
  let before = value(previous);
  for (const point of polygon) {
    const after = value(point),
      inside = after >= -EPS,
      previousInside = before >= -EPS;
    if (inside !== previousInside)
      result.push(interpolate(previous, point, before / (before - after)));
    if (inside) result.push(point);
    previous = point;
    before = after;
  }
  return unique(result);
}

/**
 * Draw the actual multiclass hyperplane patches, not every pairwise score tie.
 * A tie score_a=score_b is a visible boundary only where those scores also beat
 * every other class. Clipping by those inequalities removes spurious planes
 * through regions won by a third class. Binary models naturally yield one face.
 */
function linearFaces(
  min: Point3,
  max: Point3,
  base: Vector,
  scores: NonNullable<SurfaceField["scores"]>,
): DecisionFace[] {
  const origin = scores(base);
  if (!origin || origin.length < 2) return [];
  const weights: Point3[] = origin.map(() => [0, 0, 0]);
  for (let axis = 0; axis < 3; axis++) {
    const input = base.slice();
    input[axis] = 1;
    const values = scores(input);
    if (!values) return [];
    origin.forEach((value, c) => (weights[c][axis] = values[c] - value));
  }
  const corners: Point3[] = Array.from({ length: 8 }, (_, i) => [
    i & 1 ? max[0] : min[0],
    i & 2 ? max[1] : min[1],
    i & 4 ? max[2] : min[2],
  ]);
  const faces: DecisionFace[] = [];
  for (let a = 0; a < origin.length; a++)
    for (let b = a + 1; b < origin.length; b++) {
      const normal = subtract(weights[a], weights[b]),
        offset = origin[a] - origin[b];
      if (Math.hypot(...normal) < EPS) continue; // Identical scores do not define a unique separating plane.
      const intersections: Point3[] = [];
      for (let i = 0; i < 8; i++)
        for (const bit of [1, 2, 4]) {
          const j = i ^ bit;
          if (j < i) continue;
          const first = dot(normal, corners[i]) + offset,
            second = dot(normal, corners[j]) + offset;
          if (Math.abs(first) < EPS) intersections.push(corners[i]);
          if (Math.abs(second) < EPS) intersections.push(corners[j]);
          if (first * second < 0)
            intersections.push(
              interpolate(corners[i], corners[j], first / (first - second)),
            );
        }
      let polygon = orderFace(intersections, normal);
      for (let c = 0; c < origin.length && polygon.length >= 3; c++) {
        if (c === a || c === b) continue;
        const direction = subtract(weights[a], weights[c]);
        polygon = clip(
          polygon,
          (p) => dot(direction, p) + origin[a] - origin[c],
        );
      }
      if (polygon.length >= 3)
        faces.push({ vertices: polygon, classes: [a, b] });
    }
  return faces;
}

/**
 * Extract zero sets with marching tetrahedra for curved/discontinuous models.
 * All cubes use the same six-tetrahedron split, so neighboring cells share the
 * same face diagonals. Interpolate zero crossings along tetrahedron edges and
 * triangulate their ordered polygon. This creates connected surfaces in space;
 * it does not draw point sprites on three arbitrarily chosen z layers.
 */
function sampledFaces(
  min: Point3,
  max: Point3,
  base: Vector,
  field: SurfaceField,
  classes: number,
  level: number,
): DecisionFace[] {
  const resolution = 10,
    stride = resolution + 1,
    faces: DecisionFace[] = [];
  const nodes: { point: Point3; values: Vector }[] = [];
  for (let z = 0; z <= resolution; z++)
    for (let y = 0; y <= resolution; y++)
      for (let x = 0; x <= resolution; x++) {
        const point: Point3 = [
          min[0] + ((max[0] - min[0]) * x) / resolution,
          min[1] + ((max[1] - min[1]) * y) / resolution,
          min[2] + ((max[2] - min[2]) * z) / resolution,
        ];
        const input = base.slice();
        point.forEach((value, i) => (input[i] = value));
        const scores = field.regression ? null : field.scores(input);
        const prediction = scores ? 0 : field.predict(input);
        const values = field.regression
          ? [prediction - level]
          : classes === 2
            ? [scores ? scores[1] - scores[0] : prediction === 1 ? 1 : -1]
            : Array.from({ length: classes }, (_, c) =>
                scores
                  ? scores[c] - Math.max(...scores.filter((_, i) => i !== c))
                  : prediction === c
                    ? 1
                    : -1,
              );
        nodes.push({ point, values });
      }
  const offsets = [
    0,
    1,
    stride,
    stride + 1,
    stride * stride,
    stride * stride + 1,
    stride * stride + stride,
    stride * stride + stride + 1,
  ];
  const tetrahedra = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
  ];
  const edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
  ];
  for (let z = 0; z < resolution; z++)
    for (let y = 0; y < resolution; y++)
      for (let x = 0; x < resolution; x++) {
        const start = x + y * stride + z * stride * stride,
          cube = offsets.map((offset) => nodes[start + offset]);
        for (
          let c = 0;
          c < (field.regression || classes === 2 ? 1 : classes);
          c++
        )
          for (const tetra of tetrahedra) {
            const vertices = tetra.map((i) => cube[i]),
              crossings: Point3[] = [];
            for (const [a, b] of edges) {
              const first = vertices[a].values[c],
                second = vertices[b].values[c];
              if (
                !Number.isFinite(first) ||
                !Number.isFinite(second) ||
                first < 0 === second < 0
              )
                continue;
              crossings.push(
                interpolate(
                  vertices[a].point,
                  vertices[b].point,
                  first / (first - second),
                ),
              );
            }
            const polygon = orderFace(crossings);
            for (let i = 1; i + 1 < polygon.length; i++)
              faces.push({
                vertices: [polygon[0], polygon[i], polygon[i + 1]],
                classes: field.regression ? [0] : classes === 2 ? [0, 1] : [c],
              });
          }
      }
  return faces;
}

/**
 * Higher-dimensional models are intersected with the first-three-feature view:
 * all remaining features stay at their dataset means, matching the old renderer.
 * This is a cross-section of the learned boundary, not a retrained 3D model.
 * Extraction is called by the worker; camera changes only reproject its faces.
 */
export function buildDecisionSurfaces(
  dataset: Dataset,
  field: SurfaceField,
): DecisionFace[] {
  const dimensions = dataset.data[0]?.x.length || 0;
  if (dimensions < 3 || field.clustering) return [];
  const [min, max] = spatialDomain(dataset);
  const base = Array.from({ length: dimensions }, (_, i) =>
    i < 3
      ? 0
      : dataset.data.reduce((sum, row) => sum + row.x[i], 0) /
        dataset.data.length,
  );
  if (field.linear && !field.regression)
    return linearFaces(min, max, base, field.scores);
  const targets = dataset.data.map((row) => row.y).sort((a, b) => a - b),
    middle = Math.floor(targets.length / 2);
  const median =
    targets.length % 2
      ? targets[middle]
      : (targets[middle - 1] + targets[middle]) / 2;
  return sampledFaces(min, max, base, field, dataset.classes, median);
}
