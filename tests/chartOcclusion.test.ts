import assert from "node:assert/strict";
import test from "node:test";
import { isBehindSurface } from "../src/engine/charts.ts";

test("a closer hyperplane hides points inside its projected area", () => {
  const triangle = [
    { x: 0, y: 0, depth: 1 },
    { x: 10, y: 0, depth: 1 },
    { x: 0, y: 10, depth: 1 },
  ] as const;
  assert.equal(isBehindSurface({ x: 2, y: 2, depth: 0 }, [triangle]), true);
  assert.equal(isBehindSurface({ x: 2, y: 2, depth: 2 }, [triangle]), false);
  assert.equal(isBehindSurface({ x: 8, y: 8, depth: 0 }, [triangle]), false);
});

test("occlusion interpolates depth across a tilted hyperplane", () => {
  const triangle = [
    { x: 0, y: 0, depth: 0 },
    { x: 10, y: 0, depth: 2 },
    { x: 0, y: 10, depth: 2 },
  ] as const;
  assert.equal(isBehindSurface({ x: 2, y: 2, depth: 0.7 }, [triangle]), true);
  assert.equal(isBehindSurface({ x: 2, y: 2, depth: 0.9 }, [triangle]), false);
});
