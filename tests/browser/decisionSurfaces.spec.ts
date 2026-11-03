import { expect, test } from "@playwright/test";
import type { SolverEvent, PredictionGrid } from "../../src/solver/protocol.ts";

test("3D multiclass graphs draw separating hyperplanes that can be rotated and hidden", async ({
  page,
}) => {
  // Observe the actual worker boundary, rather than testing a UI-only mock of a
  // surface. This catches regressions back to fixed-z prediction tiles.
  await page.addInitScript(() => {
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(url: string | URL, options?: WorkerOptions) {
        super(url, options);
        this.addEventListener("message", (event: MessageEvent<SolverEvent>) => {
          if (event.data.type === "frame" && event.data.frame.plot?.three)
            (
              window as typeof window & { __surfacePlot?: PredictionGrid }
            ).__surfacePlot = event.data.frame.plot;
        });
      }
    };
  });
  await page.goto("/");
  await page.locator("#dataset").selectOption("threeblobs");
  await page.locator("#featureCount").selectOption("3");
  await expect(page.locator("#algorithm")).toHaveValue("softmax");
  await page.locator("#train").click();
  await expect(page.locator("#status")).toContainText("Simulazione completata");
  const geometry = await page.evaluate(
    () =>
      (window as typeof window & { __surfacePlot?: PredictionGrid })
        .__surfacePlot,
  );
  expect(geometry?.cells).toEqual([]);
  expect(geometry?.surfaces.length).toBeGreaterThanOrEqual(2);
  expect(
    geometry?.surfaces.every(
      (face) => face.classes.length === 2 && face.vertices.length >= 3,
    ),
  ).toBe(true);
  const controls = page.locator('.surface-controls input[type="checkbox"]');
  await expect(controls.first()).toBeChecked();
  const canvas = page.getByRole("img", {
    name: "Dati e previsione del modello",
    exact: true,
  });
  const hash = () =>
    canvas.evaluate((element) => {
      const canvas = element as HTMLCanvasElement;
      const pixels = canvas
        .getContext("2d")!
        .getImageData(0, 0, canvas.width, canvas.height).data;
      let result = 2166136261;
      for (let i = 0; i < pixels.length; i += 4)
        result = Math.imul(
          result ^ pixels[i] ^ pixels[i + 1] ^ pixels[i + 2],
          16777619,
        );
      return result;
    });
  await canvas.screenshot({ path: "test-results/decision-hyperplanes.png" });
  const opacity = page.getByRole("slider", { name: /Opacità confine/ }).first();
  const otherOpacity = page
    .getByRole("slider", { name: /Opacità confine/ })
    .nth(1);
  const originalOpacityImage = await hash();
  await opacity.focus();
  await opacity.press("Home");
  await expect(opacity).toHaveValue("0");
  await expect(otherOpacity).toHaveValue("26");
  await expect.poll(hash).not.toBe(originalOpacityImage);
  await opacity.press("End");
  await expect(opacity).toHaveValue("100");
  for (const slider of await page
    .getByRole("slider", { name: /Opacità confine/ })
    .all()) {
    await slider.focus();
    await slider.press("End");
  }
  await canvas.screenshot({ path: "test-results/opaque-hyperplanes.png" });
  const initial = await hash(),
    box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  await page.mouse.move(box!.x + box!.width * 0.5, box!.y + box!.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(
    box!.x + box!.width * 0.65,
    box!.y + box!.height * 0.55,
    { steps: 8 },
  );
  await page.mouse.up();
  await expect.poll(hash).not.toBe(initial);
  const rotated = await hash();
  for (const checkbox of await controls.all()) await checkbox.uncheck();
  await expect.poll(hash).not.toBe(rotated);
  const pointsImage = await hash();
  await page
    .getByRole("checkbox", { name: "Mostra punti classe 1", exact: true })
    .uncheck();
  await expect.poll(hash).not.toBe(pointsImage);
  await page
    .getByRole("checkbox", { name: "Mostra tutti i punti", exact: true })
    .uncheck();
  const noPoints = await hash();
  await page
    .getByRole("checkbox", {
      name: "Evidenzia spazio classificato",
      exact: true,
    })
    .check();
  await expect(page.locator(".region-controls")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect.poll(hash).not.toBe(noPoints);
  const space = await page.evaluate(
    () =>
      (window as typeof window & { __surfacePlot?: PredictionGrid })
        .__surfacePlot,
  );
  expect(space?.regionsRequested).toBe(true);
  expect(space?.regions.length).toBe(3);
  expect(
    space?.regions.every(
      (region) => !region.approximate && region.faces.length >= 4,
    ),
  ).toBe(true);
  await canvas.screenshot({ path: "test-results/classification-space.png" });
  const allRegions = await hash();
  await page
    .getByRole("checkbox", { name: "Mostra spazio classe 2", exact: true })
    .uncheck();
  await expect.poll(hash).not.toBe(allRegions);
  await page.locator("#reset").click();
  await expect(controls).toHaveCount(0);
});
