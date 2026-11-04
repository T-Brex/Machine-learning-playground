import { expect, test } from "@playwright/test";

test("simulation controls remain visible below the sticky header while scrolling", async ({
  page,
}) => {
  await page.setViewportSize({ width: 900, height: 600 });
  await page.goto("/");
  await page.evaluate(() => window.scrollTo(0, 500));

  const toolbar = page.locator(".training-toolbar"),
    header = page.locator(".top");
  await expect(toolbar).toBeVisible();
  await expect
    .poll(async () => {
      const [toolbarBox, headerBox] = await Promise.all([
        toolbar.boundingBox(),
        header.boundingBox(),
      ]);
      return toolbarBox && headerBox
        ? Math.abs(toolbarBox.y - headerBox.y - headerBox.height)
        : Infinity;
    })
    .toBeLessThan(2);
});

test("large solver jobs leave navigation, chart gestures and animation on the UI thread responsive", async ({
  page,
}) => {
  const workerUrls: string[] = [];
  page.on("worker", (worker) => workerUrls.push(worker.url()));
  await page.goto("/");
  await page.locator("#sampleCount").fill("1000");
  await page.locator("#featureCount").selectOption("8");
  await page.locator("#algorithm").selectOption("forest");
  await page.locator("#methodParam_nEstimators").fill("300");
  await expect(page.locator("#train")).toBeEnabled();
  await page.locator("#train").click();
  await expect(page.locator("#status")).toContainText("Training in corso");

  // Fitting this ensemble is deliberately substantial. Navigation must not wait
  // for that epoch to finish, and cached chart pixels must still pan/zoom locally.
  await page
    .getByRole("button", { name: "Visualizzazioni", exact: true })
    .click({ timeout: 1500 });
  await page
    .getByRole("button", {
      name: "Aumenta zoom: Dati e previsione del modello",
      exact: true,
    })
    .click({ timeout: 1500 });
  const started = Date.now();
  const frames = await page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        const start = performance.now();
        let count = 0;
        function frame() {
          count++;
          if (performance.now() - start >= 600) resolve(count);
          else requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
      }),
  );
  expect(frames).toBeGreaterThanOrEqual(10);
  expect(Date.now() - started).toBeLessThan(1800);
  expect(workerUrls.some((url) => url.includes("solver.worker"))).toBe(true);

  // Reset must terminate an expensive atomic epoch, not merely set a flag that
  // the blocked numerical loop cannot see until it eventually finishes.
  await page
    .getByRole("button", { name: "Panoramica", exact: true })
    .click({ timeout: 1500 });
  await page.locator("#reset").click({ timeout: 1500 });
  await expect(page.locator("#train")).toBeEnabled({ timeout: 3000 });
  await expect(page.locator("#status")).toContainText("passo 0 /");
  await page.waitForTimeout(200);
  await expect(page.locator("#status")).toContainText("passo 0 /");
});
