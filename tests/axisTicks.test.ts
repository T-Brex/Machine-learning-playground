import assert from "node:assert/strict";
import test from "node:test";
import { axisTicks } from "../src/engine/axisTicks.ts";

test("panning retains tick values in the overlapping visible domain", () => {
  assert.deepEqual(axisTicks(-2.3, 2.3), [-2, -1, 0, 1, 2]);
  assert.deepEqual(axisTicks(-1.7, 2.9), [-1, 0, 1, 2]);
  assert.deepEqual(axisTicks(0.3, 4.9), [1, 2, 3, 4]);
});

test("zoom uses regular intervals without floating point noise", () => {
  assert.deepEqual(axisTicks(-0.9, 0.9), [-0.5, 0, 0.5]);
  assert.deepEqual(axisTicks(-0.23, 0.23), [-0.2, -0.1, 0, 0.1, 0.2]);
  assert.deepEqual(axisTicks(0, 6 + 1e-12), [0, 1, 2, 3, 4, 5, 6]);
});
