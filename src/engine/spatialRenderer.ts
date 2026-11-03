import type { PaintedFace, ScreenPoint } from "./sceneOcclusion.ts";

export interface SpatialPoint {
  position: ScreenPoint;
  depth: number;
  color: string;
  selected: boolean;
}
interface Renderer {
  canvas: OffscreenCanvas | HTMLCanvasElement;
  gl: WebGLRenderingContext;
  program: WebGLProgram;
  buffer: WebGLBuffer;
  mode: WebGLUniformLocation;
  scale: WebGLUniformLocation;
}
const renderers = new WeakMap<CanvasRenderingContext2D, Renderer | null>();
const vertexShader = `
attribute vec3 position;
attribute vec4 color;
attribute float size;
attribute float selection;
uniform float pixelScale;
varying vec4 rgba;
varying float selected;
varying float diameter;
void main(){
  gl_Position=vec4(position,1.0);
  gl_PointSize=size*pixelScale;
  rgba=color;selected=selection;diameter=size*pixelScale;
}`;
const fragmentShader = `
precision mediump float;
uniform bool points;
varying vec4 rgba;
varying float selected;
varying float diameter;
void main(){
  vec4 result=rgba;
  if(points){
    float r=length(gl_PointCoord*2.0-1.0);
    if(r>1.0)discard;
    if(selected>0.5){
      if(r>0.6&&r<0.8)discard;
      if(r>=0.8)result=vec4(1.0);
    }
    result.a*=1.0-smoothstep(1.0-2.0/diameter,1.0,r);
  }
  gl_FragColor=result;
}`;

/**
 * A GPU layer avoids Canvas's per-triangle antialiasing seams. Shared triangle
 * edges follow the rasterizer's coverage rule, so curved surfaces do not acquire
 * dark/light triangulation lines. The depth buffer also orders opaque surfaces
 * and point markers per pixel instead of painting every marker on top.
 *
 * This offscreen layer is composited into the existing React-owned 2D canvas;
 * axes, labels, camera gestures, and hit testing can keep their current API.
 */
function createRenderer(): Renderer | null {
  const canvas =
    typeof OffscreenCanvas === "function"
      ? new OffscreenCanvas(1, 1)
      : document.createElement("canvas");
  const gl = canvas.getContext("webgl", {
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
    preserveDrawingBuffer: true,
  }) as WebGLRenderingContext | null;
  if (!gl) return null;
  const compile = (type: number, source: string) => {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  };
  const vertex = compile(gl.VERTEX_SHADER, vertexShader),
    fragment = compile(gl.FRAGMENT_SHADER, fragmentShader);
  if (!vertex || !fragment) return null;
  const program = gl.createProgram(),
    buffer = gl.createBuffer();
  if (!program || !buffer) return null;
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    gl.deleteBuffer(buffer);
    return null;
  }
  const mode = gl.getUniformLocation(program, "points"),
    scale = gl.getUniformLocation(program, "pixelScale");
  if (!mode || !scale) return null;
  return { canvas, gl, program, buffer, mode, scale };
}

function rgb(color: string): [number, number, number] {
  // Surface/point colors are the local six-digit palette, except regression's
  // HSL ramp. Convert that ramp directly instead of consulting DOM/CSS per vertex.
  if (color.startsWith("#"))
    return [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16) / 255) as [
      number,
      number,
      number,
    ];
  const values = color.match(/[\d.]+/g)?.map(Number) || [0, 0, 50];
  const [h, s, l] = [values[0] / 360, values[1] / 100, values[2] / 100];
  const a = s * Math.min(l, 1 - l),
    f = (n: number) => {
      const k = (n + h * 12) % 12;
      return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    };
  return [f(0), f(8), f(4)];
}

/**
 * Render opaque geometry and markers with depth writes, then transparent faces
 * back-to-front with blending and read-only depth. A plane in front of a marker
 * tints/hides it; a plane behind that marker fails the depth test. Alpha uses a
 * separate ONE source factor so the offscreen canvas stores premultiplied RGB
 * and alpha consistently when drawImage composites it over the graph background.
 */
export function renderSpatialScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  faces: PaintedFace[],
  points: SpatialPoint[],
): boolean {
  if (!renderers.has(ctx)) renderers.set(ctx, createRenderer());
  const renderer = renderers.get(ctx);
  if (!renderer) return false;
  const { gl, canvas, program, buffer } = renderer;
  if (gl.isContextLost()) return false;
  const pixelScale = ctx.canvas.width / Math.max(1, width);
  const w = Math.max(1, Math.round(width * pixelScale)),
    h = Math.max(1, Math.round(height * pixelScale));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  gl.viewport(0, 0, w, h);
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  // Vertex records: position.xyz, rgba, point diameter, selected flag.
  for (const [name, count, offset] of [
    ["position", 3, 0],
    ["color", 4, 3],
    ["size", 1, 7],
    ["selection", 1, 8],
  ] as const) {
    const location = gl.getAttribLocation(program, name);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, count, gl.FLOAT, false, 9 * 4, offset * 4);
  }
  gl.uniform1f(renderer.scale, pixelScale);
  gl.clearColor(0, 0, 0, 0);
  gl.depthMask(true);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(
    gl.SRC_ALPHA,
    gl.ONE_MINUS_SRC_ALPHA,
    gl.ONE,
    gl.ONE_MINUS_SRC_ALPHA,
  );
  const vertex = (
    target: number[],
    point: ScreenPoint,
    depth: number,
    color: [number, number, number],
    opacity: number,
    size = 1,
    selected = false,
  ) => {
    target.push(
      (point[0] / width) * 2 - 1,
      1 - (point[1] / height) * 2,
      -depth,
      ...color,
      opacity,
      size,
      selected ? 1 : 0,
    );
  };
  const geometry = (input: PaintedFace[]) => {
    const triangles: number[] = [],
      lines: number[] = [];
    for (const face of input) {
      const color = rgb(face.color);
      for (let i = 1; i + 1 < face.vertices.length; i++)
        for (const j of [0, i, i + 1])
          vertex(
            triangles,
            face.vertices[j],
            face.depths[j],
            color,
            face.opacity,
          );
      // Only explicit polygon perimeters are outlined. Nonlinear triangles have
      // no wireframe stroke; shared edges must disappear into the smooth mesh.
      if (face.outline)
        for (let i = 0; i < face.vertices.length; i++)
          for (const j of [i, (i + 1) % face.vertices.length])
            vertex(
              lines,
              face.vertices[j],
              face.depths[j] + 1e-6,
              color,
              Math.min(1, face.opacity * 2.5),
            );
    }
    return { triangles, lines };
  };
  const draw = (data: number[], primitive: number, pointMode = false) => {
    if (!data.length) return;
    gl.uniform1i(renderer.mode, pointMode ? 1 : 0);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
    gl.drawArrays(primitive, 0, data.length / 9);
  };
  const opaque = geometry(faces.filter((face) => face.opacity === 1)),
    transparent = geometry(faces.filter((face) => face.opacity < 1));
  draw(opaque.triangles, gl.TRIANGLES);
  draw(opaque.lines, gl.LINES);
  const markers: number[] = [];
  for (const point of points)
    vertex(
      markers,
      point.position,
      point.depth,
      rgb(point.color),
      1,
      point.selected ? 20 : 7,
      point.selected,
    );
  draw(markers, gl.POINTS, true);
  gl.depthMask(false);
  draw(transparent.triangles, gl.TRIANGLES);
  draw(transparent.lines, gl.LINES);
  gl.depthMask(true);
  ctx.drawImage(canvas, 0, 0, width, height);
  return true;
}

/** Release GPU allocations promptly when a React canvas leaf unmounts. */
export function releaseSpatialRenderer(ctx: CanvasRenderingContext2D) {
  const renderer = renderers.get(ctx);
  if (renderer) {
    renderer.gl.deleteBuffer(renderer.buffer);
    renderer.gl.deleteProgram(renderer.program);
    renderer.gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
  renderers.delete(ctx);
}
