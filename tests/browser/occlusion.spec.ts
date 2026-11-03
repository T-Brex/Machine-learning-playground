import { expect, test } from "@playwright/test";
import type { Dataset, Snapshot } from "../../src/types.ts";

test("curved 3D surfaces remain filled without a visible triangle wireframe", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("#dataset").selectOption("circles");
  await page.locator("#featureCount").selectOption("3");
  await page.locator("#algorithm").selectOption("naive");
  await page.locator("#train").click();
  await expect(page.locator("#status")).toContainText("Simulazione completata");
  const slider = page.getByRole("slider", { name: /Opacità confine/ }).first();
  await slider.focus();
  await slider.press("End");
  await page
    .getByRole("img", { name: "Dati e previsione del modello", exact: true })
    .screenshot({ path: "test-results/smooth-curved-surface.png" });
});

test("adjacent transparent mesh triangles share seamless rasterized edges", async ({
  page,
}) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const path = "/src/engine/spatialRenderer.ts";
    const { renderSpatialScene, releaseSpatialRenderer } = (await import(
      path
    )) as typeof import("../../src/engine/spatialRenderer.ts");
    const canvas = document.createElement("canvas");
    canvas.width = 200;
    canvas.height = 200;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#091a2a";
    ctx.fillRect(0, 0, 200, 200);
    const drawn = renderSpatialScene(
      ctx,
      200,
      200,
      [
        {
          vertices: [
            [20, 20],
            [180, 20],
            [180, 180],
          ],
          depths: [0, 0, 0],
          opacity: 0.5,
          color: "#bba7ff",
          outline: false,
        },
        {
          vertices: [
            [20, 20],
            [180, 180],
            [20, 180],
          ],
          depths: [0, 0, 0],
          opacity: 0.5,
          color: "#bba7ff",
          outline: false,
        },
      ],
      [],
    );
    const pixel = (x: number, y: number) =>
      Array.from(ctx.getImageData(x, y, 1, 1).data).slice(0, 3);
    const samples = {
      drawn,
      edge: pixel(100, 100),
      first: pixel(100, 70),
      second: pixel(70, 100),
    };
    releaseSpatialRenderer(ctx);
    return samples;
  });
  expect(result.drawn).toBe(true);
  for (let i = 0; i < 3; i++) {
    expect(Math.abs(result.edge[i] - result.first[i])).toBeLessThanOrEqual(1);
    expect(Math.abs(result.edge[i] - result.second[i])).toBeLessThanOrEqual(1);
  }
});

test("opaque tilted planes occlude rear points while front points and transparent-plane blending stay correct", async ({
  page,
}) => {
  await page.goto("/");
  const scenes = await page.evaluate(async () => {
    // Use the actual Vite-served renderer and browser canvas, so these are pixel
    // assertions rather than mocks of the draw order. The tilted plane y=x has
    // a point in front of its far side and a point behind its near side; sorting
    // whole faces by average depth would fail this arrangement.
    const modulePath = "/src/engine/charts.ts";
    const { drawScatter } = (await import(
      modulePath
    )) as typeof import("../../src/engine/charts.ts");
    const canvas = document.createElement("canvas");
    canvas.width = 200;
    canvas.height = 200;
    const ctx = canvas.getContext("2d")!;
    const dataset: Dataset = {
      classes: 2,
      data: [
        { x: [1, 0.5, 0], y: 0 },
        { x: [-1, -0.5, 0], y: 1 },
      ],
    };
    const snapshot: Snapshot = {
      iteration: 1,
      fitted: true,
      loss: 0,
      losses: [],
      trainHistory: [],
      testHistory: [],
      metrics: {},
      model: null,
      train: [],
      test: [],
      regression: false,
      clustering: false,
      total: 1,
      issue: "",
    };
    const scene = (
      opacity: number,
      hiddenPointSets: number[] = [],
      showPoints = true,
    ) => {
      ctx.fillStyle = "#091a2a";
      ctx.fillRect(0, 0, 200, 200);
      const hits = drawScatter(ctx, 200, 200, {
        dataset,
        snapshot,
        predictions: [0, 1],
        centers: [],
        view: { zoom: 1, panX: 0, panY: 0, yaw: 0, pitch: 0 },
        surfaceOpacity: { "0:1": opacity },
        hiddenPointSets,
        showPoints,
        plot: {
          cells: [],
          stepX: 0,
          stepY: 0,
          three: true,
          regions: [],
          regionsRequested: false,
          surfaces: [
            {
              classes: [0, 1],
              vertices: [
                [-2, -2, -2],
                [2, 2, -2],
                [2, 2, 2],
                [-2, -2, 2],
              ],
            },
          ],
        },
      });
      const pixel = (x: number) =>
        Array.from(ctx.getImageData(x, 100, 1, 1).data).slice(0, 3);
      return {
        back: pixel(126),
        front: pixel(74),
        hits: hits.map((hit) => hit.index).sort(),
      };
    };
    return {
      opaque: scene(1),
      transparent: scene(0.5),
      invisible: scene(0),
      filtered: scene(0, [1]),
      noPoints: scene(1, [], false),
    };
  });
  expect(scenes.opaque.back).toEqual([187, 167, 255]);
  expect(scenes.opaque.front).toEqual([255, 157, 55]);
  expect(scenes.opaque.hits).toEqual([1]);
  expect(scenes.transparent.front).toEqual([255, 157, 55]);
  for (const [i, value] of scenes.transparent.back.entries())
    expect(Math.abs(value - [114, 158, 255][i])).toBeLessThanOrEqual(1);
  expect(scenes.invisible.back).toEqual([41, 149, 255]);
  expect(scenes.invisible.hits).toEqual([0, 1]);
  expect(scenes.filtered.hits).toEqual([0]);
  expect(scenes.filtered.front).not.toEqual([255, 157, 55]);
  expect(scenes.noPoints.hits).toEqual([]);
  expect(scenes.noPoints.front).toEqual([187, 167, 255]);
});
