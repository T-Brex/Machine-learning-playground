/** Canvas draw callbacks use CSS-pixel coordinates; device-pixel scaling is applied by the effect. */
interface CanvasChartProps {
  label: string;
  draw: (
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    view: ChartView,
  ) => ChartPoint[] | void;
  interactive?: boolean;
  rotate?: boolean;
  onSelect?: (index: number) => void;
  height?: number;
  resetKey?: unknown;
}
interface Gesture {
  moved: boolean;
  distance?: number;
}
import type { PointerEvent } from "react";
import { useEffect, useRef, useState } from "react";
import type { ChartPoint, ChartView } from "../types.ts";

const INITIAL_VIEW = { zoom: 1, panX: 0, panY: 0, yaw: -0.65, pitch: 0.45 };

/**
 * Own the canvas element while keeping numerical drawing outside React markup.
 *
 * ResizeObserver reports CSS-pixel dimensions. The draw effect sizes the backing
 * store by devicePixelRatio and scales the context once, so renderers and hit
 * coordinates can continue using CSS pixels on both normal and high-DPI displays.
 * A canvas is a leaf: renderers draw pixels and never move React-owned DOM nodes.
 *
 * Pointer refs track transient gestures without rendering on every pointer-bookkeeping
 * change. View changes use React state because they must redraw. Every observer and
 * native wheel listener has effect cleanup, including React StrictMode remounts.
 *
 * Spatial renderers apply their own projection; ordinary charts use a context
 * transform. Applying both would double the zoom/pan and invalidate hit testing.
 */
export default function CanvasChart({
  label,
  draw,
  interactive = false,
  rotate = false,
  onSelect,
  height = 280,
  resetKey,
}: CanvasChartProps) {
  const ref = useRef<HTMLCanvasElement | null>(null),
    points = useRef<ChartPoint[]>([]),
    pointers = useRef(new Map<number, { x: number; y: number }>()),
    gesture = useRef<Gesture | null>(null);
  const [view, setView] = useState(INITIAL_VIEW);
  const [size, setSize] = useState({ width: 600, height });
  useEffect(() => setView(INITIAL_VIEW), [resetKey]);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const observer = new ResizeObserver(([entry]) =>
      setSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      }),
    );
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const canvas = ref.current,
      dpr = window.devicePixelRatio || 1;
    if (!canvas) return;
    canvas.width = Math.max(1, Math.round(size.width * dpr));
    canvas.height = Math.max(1, Math.round(size.height * dpr));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#091a2a";
    ctx.fillRect(0, 0, size.width, size.height);
    ctx.save();
    if (!interactive) {
      ctx.translate(size.width / 2 + view.panX, size.height / 2 + view.panY);
      ctx.scale(view.zoom, view.zoom);
      ctx.translate(-size.width / 2, -size.height / 2);
    }
    points.current = draw(ctx, size.width, size.height, view) || [];
    ctx.restore();
  }, [draw, size, view, interactive]);
  // A native non-passive listener lets pinch/scroll stay inside the chart.
  useEffect(() => {
    const canvas = ref.current;
    function wheel(event: WheelEvent) {
      event.preventDefault();
      setView((v) =>
        event.ctrlKey
          ? {
              ...v,
              zoom: Math.max(
                0.5,
                Math.min(8, v.zoom * Math.exp(-event.deltaY * 0.002)),
              ),
            }
          : { ...v, panX: v.panX - event.deltaX, panY: v.panY - event.deltaY },
      );
    }
    if (!canvas) return;
    canvas.addEventListener("wheel", wheel, { passive: false });
    return () => canvas.removeEventListener("wheel", wheel);
  }, []);
  /**
   * Interpret pointer motion without mixing gesture bookkeeping with model state.
   *
   * Two active pointers scale the view by their distance ratio. One mouse pointer
   * rotates a 3D projection unless Shift is held; other drags pan. Capture the prior
   * pinch distance before the state updater runs rather than reading a mutable ref
   * inside a deferred updater. None of these gestures changes training samples.
   */
  function move(event: PointerEvent<HTMLCanvasElement>) {
    const previous = pointers.current.get(event.pointerId);
    if (!previous) return;
    pointers.current.set(event.pointerId, {
      x: event.clientX,
      y: event.clientY,
    });
    const dx = event.clientX - previous.x,
      dy = event.clientY - previous.y;
    if (Math.abs(dx) + Math.abs(dy) > 1 && gesture.current)
      gesture.current.moved = true;
    if (pointers.current.size === 2) {
      const [a, b] = pointers.current.values(),
        distance = Math.hypot(a.x - b.x, a.y - b.y);
      const previousDistance = gesture.current?.distance;
      if (previousDistance)
        setView((v) => ({
          ...v,
          zoom: Math.max(
            0.5,
            Math.min(8, (v.zoom * distance) / previousDistance),
          ),
        }));
      if (gesture.current) gesture.current.distance = distance;
    } else
      setView((v) =>
        rotate && !event.shiftKey && event.pointerType !== "touch"
          ? {
              ...v,
              yaw: v.yaw + dx * 0.008,
              pitch: Math.max(-1.5, Math.min(1.5, v.pitch + dy * 0.008)),
              panX: v.panX,
              panY: v.panY,
            }
          : { ...v, panX: v.panX + dx, panY: v.panY + dy },
      );
  }
  /** Treat a gesture as selection only when the pointer did not move; choose the nearest plotted sample. */
  function release(event: PointerEvent<HTMLCanvasElement>) {
    if (!gesture.current?.moved && onSelect) {
      const rect = event.currentTarget.getBoundingClientRect(),
        x = event.clientX - rect.left,
        y = event.clientY - rect.top;
      let nearest: { index: number; distance: number } | null = null;
      for (const point of points.current) {
        const d = (x - point.x) ** 2 + (y - point.y) ** 2;
        if (!nearest || d < nearest.distance)
          nearest = { index: point.index, distance: d };
      }
      if (nearest) onSelect(nearest.index);
    }
    pointers.current.delete(event.pointerId);
    if (!pointers.current.size) gesture.current = null;
  }
  return (
    <div className="react-chart">
      <div className="chart-controls">
        <button
          type="button"
          className="secondary"
          aria-label={`Riduci zoom: ${label}`}
          onClick={() =>
            setView((v) => ({ ...v, zoom: Math.max(0.5, v.zoom * 0.8) }))
          }
        >
          −
        </button>
        <button
          type="button"
          className="secondary"
          aria-label={`Aumenta zoom: ${label}`}
          onClick={() =>
            setView((v) => ({ ...v, zoom: Math.min(8, v.zoom * 1.25) }))
          }
        >
          +
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => setView(INITIAL_VIEW)}
        >
          Ripristina vista
        </button>
        <span className="small muted">
          Ctrl + rotellina / pinch: zoom · trascina
          {rotate ? " per ruotare · Shift: sposta" : ": sposta"}
        </span>
      </div>
      <canvas
        ref={ref}
        role="img"
        aria-label={label}
        className="chart"
        style={{ height, touchAction: "none" }}
        onDoubleClick={() => setView(INITIAL_VIEW)}
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
          gesture.current = { moved: false };
        }}
        onPointerMove={move}
        onPointerUp={release}
        onPointerCancel={(e) => {
          pointers.current.delete(e.pointerId);
          gesture.current = null;
        }}
      />
    </div>
  );
}
