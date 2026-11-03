import { useEffect, useRef, useState } from "react";

const INITIAL_VIEW = { zoom: 1, panX: 0, panY: 0, yaw: -0.65, pitch: 0.45 };

/** The canvas is a React-owned leaf: effects draw pixels and clean up resize observers. */
export default function CanvasChart({
  label,
  draw,
  interactive = false,
  rotate = false,
  onSelect,
  height = 280,
  resetKey,
}) {
  const ref = useRef(null),
    points = useRef([]),
    pointers = useRef(new Map()),
    gesture = useRef(null);
  const [view, setView] = useState(INITIAL_VIEW);
  const [size, setSize] = useState({ width: 600, height });
  useEffect(() => setView(INITIAL_VIEW), [resetKey]);
  useEffect(() => {
    const canvas = ref.current;
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
    canvas.width = Math.max(1, Math.round(size.width * dpr));
    canvas.height = Math.max(1, Math.round(size.height * dpr));
    const ctx = canvas.getContext("2d");
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
    function wheel(event) {
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
    canvas.addEventListener("wheel", wheel, { passive: false });
    return () => canvas.removeEventListener("wheel", wheel);
  }, []);
  /** Track pointer deltas; two fingers zoom, while one pointer pans or rotates the spatial projection. */
  function move(event) {
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
      if (gesture.current?.distance)
        setView((v) => ({
          ...v,
          zoom: Math.max(
            0.5,
            Math.min(8, (v.zoom * distance) / gesture.current.distance),
          ),
        }));
      gesture.current.distance = distance;
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
  function release(event) {
    if (!gesture.current?.moved && onSelect) {
      const rect = ref.current.getBoundingClientRect(),
        x = event.clientX - rect.left,
        y = event.clientY - rect.top;
      let nearest = null;
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
          ref.current.setPointerCapture(e.pointerId);
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
