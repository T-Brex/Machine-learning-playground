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
  await page.locator("#reset").click();
  await expect(controls).toHaveCount(0);
});
