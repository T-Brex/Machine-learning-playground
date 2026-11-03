/**
 * CPU visibility queries and fallback compositing for the orthographic 3D canvas.
 * The GPU renderer handles normal drawing; these queries keep hit targets from
 * selecting hidden samples and preserve occlusion if a GPU context is unavailable.
 *
 * Painting every marker last makes even a 100% opaque plane look transparent.
 * Sorting a whole polygon by its center is insufficient too: the same tilted
 * plane can pass in front of one sample and behind another. Under an orthographic
 * camera, depth is affine in screen coordinates. Recover that depth equation,
 * clip each overlapping face to the part in front of the marker, and composite
 * only that part over the marker. Transparency then obeys ordinary alpha blending.
 *
 * A screen-space grid limits each point to nearby faces. Camera changes only
 * reproject cached geometry; this code never evaluates a model or changes a fit.
 */
export type ScreenPoint = [number, number];
export interface PaintedFace {
  vertices: ScreenPoint[];
  depths: number[];
  color: string;
  opacity: number;
  outline: boolean;
}
interface IndexedFace extends PaintedFace {
  equation: [number, number, number];
  bounds: [number, number, number, number];
}
const EPS = 1e-8;

/** Recover depth(x,y)=a*x+b*y+c from any non-collinear projected triangle. */
function depthEquation(face: PaintedFace): IndexedFace["equation"] | null {
  const [x0, y0] = face.vertices[0],
    z0 = face.depths[0];
  for (let i = 1; i + 1 < face.vertices.length; i++) {
    const [x1, y1] = face.vertices[i],
      [x2, y2] = face.vertices[i + 1];
    const dx1 = x1 - x0,
      dy1 = y1 - y0,
      dx2 = x2 - x0,
      dy2 = y2 - y0,
      det = dx1 * dy2 - dx2 * dy1;
    if (Math.abs(det) < EPS) continue; // Edge-on faces have no screen area to occlude.
    const dz1 = face.depths[i] - z0,
      dz2 = face.depths[i + 1] - z0;
    const a = (dz1 * dy2 - dz2 * dy1) / det,
      b = (dx1 * dz2 - dx2 * dz1) / det;
    return [a, b, z0 - a * x0 - b * y0];
  }
  return null;
}

/** Clip a convex face to pixels that are closer to the camera than the marker. */
function foreground(face: IndexedFace, pointDepth: number): ScreenPoint[] {
  const [a, b, c] = face.equation,
    result: ScreenPoint[] = [];
  let previous = face.vertices.at(-1)!;
  let before = a * previous[0] + b * previous[1] + c - pointDepth;
  for (const point of face.vertices) {
    const after = a * point[0] + b * point[1] + c - pointDepth;
    if (before > EPS !== after > EPS) {
      const t = before / (before - after);
      result.push([
        previous[0] + (point[0] - previous[0]) * t,
        previous[1] + (point[1] - previous[1]) * t,
      ]);
    }
    if (after > EPS) result.push(point);
    previous = point;
    before = after;
  }
  return result;
}
function contains(polygon: ScreenPoint[], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i],
      b = polygon[j];
    if (
      a[1] > y !== b[1] > y &&
      x < ((b[0] - a[0]) * (y - a[1])) / (b[1] - a[1]) + a[0]
    )
      inside = !inside;
  }
  return inside;
}

/** Build once per camera redraw; query the grid for each small point marker. */
export function createPointOcclusion(
  faces: PaintedFace[],
  width: number,
  height: number,
) {
  const size = 48,
    columns = Math.max(1, Math.ceil(width / size)),
    rows = Math.max(1, Math.ceil(height / size));
  const bins = new Map<number, IndexedFace[]>();
  for (const face of faces) {
    if (face.opacity <= 0 || face.vertices.length < 3) continue;
    const equation = depthEquation(face);
    if (!equation) continue;
    const bounds: IndexedFace["bounds"] = [
      Math.min(...face.vertices.map((p) => p[0])),
      Math.min(...face.vertices.map((p) => p[1])),
      Math.max(...face.vertices.map((p) => p[0])),
      Math.max(...face.vertices.map((p) => p[1])),
    ];
    const indexed = { ...face, equation, bounds };
    for (
      let y = Math.max(0, Math.floor(bounds[1] / size));
      y <= Math.min(rows - 1, Math.floor(bounds[3] / size));
      y++
    )
      for (
        let x = Math.max(0, Math.floor(bounds[0] / size));
        x <= Math.min(columns - 1, Math.floor(bounds[2] / size));
        x++
      ) {
        const key = x + y * columns,
          list = bins.get(key) || [];
        list.push(indexed);
        bins.set(key, list);
      }
  }
  return {
    /**
     * Resolve foreground fragments before painting a marker. Fully opaque
     * coverage can skip drawing it altogether, avoiding antialiased dot edges
     * leaking through a solid plane. Partially visible/transparent intersections
     * are composited over the marker after it is painted.
     */
    prepare(x: number, y: number, depth: number, radius: number) {
      const candidates = new Set<IndexedFace>();
      for (
        let row = Math.max(0, Math.floor((y - radius) / size));
        row <= Math.min(rows - 1, Math.floor((y + radius) / size));
        row++
      )
        for (
          let column = Math.max(0, Math.floor((x - radius) / size));
          column <= Math.min(columns - 1, Math.floor((x + radius) / size));
          column++
        )
          for (const face of bins.get(column + row * columns) || [])
            candidates.add(face);
      const front = [...candidates]
        .filter((face) => {
          const [left, top, right, bottom] = face.bounds,
            [a, b, c] = face.equation;
          return (
            left <= x + radius &&
            right >= x - radius &&
            top <= y + radius &&
            bottom >= y - radius &&
            a * x + b * y + c + Math.hypot(a, b) * radius > depth + EPS
          );
        })
        .map((face) => ({
          face,
          polygon: foreground(face, depth),
          depth: face.equation[0] * x + face.equation[1] * y + face.equation[2],
        }))
        .filter((item) => item.polygon.length >= 3)
        .sort((a, b) => a.depth - b.depth);
      const opaque = front.filter(
        ({ face, polygon }) => face.opacity === 1 && contains(polygon, x, y),
      );
      const fullyHidden = opaque.some(({ polygon }) =>
        polygon.every((a, i) => {
          const b = polygon[(i + 1) % polygon.length],
            length = Math.hypot(b[0] - a[0], b[1] - a[1]);
          return (
            length < EPS ||
            Math.abs((x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])) /
              length >=
              radius
          );
        }),
      );
      return {
        fullyHidden,
        visible: opaque.length === 0,
        composite(ctx: CanvasRenderingContext2D, selected = false) {
          if (!front.length || fullyHidden) return;
          ctx.save();
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, Math.PI * 2);
          if (selected) {
            // Selection rings have an empty gap between their outer circle and
            // the center dot. Do not blend the background twice inside that gap.
            ctx.moveTo(x + 8, y);
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.moveTo(x + 6, y);
            ctx.arc(x, y, 6, 0, Math.PI * 2);
          }
          ctx.clip(selected ? "evenodd" : "nonzero");
          for (const { face, polygon } of front) {
            ctx.beginPath();
            ctx.moveTo(...polygon[0]);
            for (const point of polygon.slice(1)) ctx.lineTo(...point);
            ctx.closePath();
            ctx.fillStyle = face.color;
            ctx.globalAlpha = face.opacity;
            ctx.fill();
          }
          ctx.restore();
        },
      };
    },
  };
}
