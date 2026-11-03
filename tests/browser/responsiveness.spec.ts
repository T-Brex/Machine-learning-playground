import { expect, test } from "@playwright/test";

test("history and distribution charts keep axis titles fixed while navigating data", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const fillText = CanvasRenderingContext2D.prototype.fillText;
    CanvasRenderingContext2D.prototype.fillText = function (
      text,
      x,
      y,
      maxWidth,
    ) {
      if (text === "Epoca" || text === "Numero di epoche")
        this.canvas.dataset.axisTitle = JSON.stringify({ text, x, y });
      if (maxWidth === undefined) fillText.call(this, text, x, y);
      else fillText.call(this, text, x, y, maxWidth);
    };
  });
  await page.goto("/");
  await page.locator("#train").click();
  await expect(page.locator("#status")).toContainText("Simulazione completata");
  for (const label of [
    "Loss per epoca",
    "Accuracy train / validazione",
    "Distribuzione della loss",
  ]) {
    const chart = page.getByRole("img", { name: label, exact: true });
    await chart.scrollIntoViewIfNeeded();
    const title = await chart.getAttribute("data-axis-title");
    expect(title).not.toBeNull();
    const pixels = () =>
      chart.evaluate((canvas: HTMLCanvasElement) => canvas.toDataURL());
    const before = await pixels();
    const box = await chart.boundingBox();
    if (!box) throw new Error("Missing chart");
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(
      box.x + box.width / 2 + 80,
      box.y + box.height / 2 + 40,
      { steps: 4 },
    );
    await page.mouse.up();
    await expect.poll(pixels).not.toBe(before);
    await expect(chart).toHaveAttribute("data-axis-title", title!);
    await page
      .getByRole("button", { name: `Aumenta zoom: ${label}`, exact: true })
      .click();
    await expect(chart).toHaveAttribute("data-axis-title", title!);
  }
});

test("scrolling over a chart scrolls the page; Ctrl+scroll zooms and dragging pans", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#step")).toBeEnabled();
  const chart = page.getByRole("img", {
    name: "Dati e previsione del modello",
    exact: true,
  });
  await chart.scrollIntoViewIfNeeded();
  const pixels = () =>
    chart.evaluate((canvas: HTMLCanvasElement) => canvas.toDataURL());
  const original = await pixels();
  const initialScroll = await page.evaluate(() => scrollY);
  await chart.hover();
  await page.mouse.wheel(0, 160);
  await expect
    .poll(() => page.evaluate(() => scrollY))
    .toBeGreaterThan(initialScroll);
  expect(await pixels()).toBe(original);

  await chart.scrollIntoViewIfNeeded();
  await chart.hover();
  const beforeZoomScroll = await page.evaluate(() => scrollY);
  await page.keyboard.down("Control");
  await page.mouse.wheel(0, -240);
  await page.keyboard.up("Control");
  await expect.poll(pixels).not.toBe(original);
  expect(await page.evaluate(() => scrollY)).toBe(beforeZoomScroll);

  const zoomed = await pixels();
  const box = await chart.boundingBox();
  if (!box) throw new Error("Chart is missing");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width / 2 + 64,
    box.y + box.height / 2 + 32,
    { steps: 4 },
  );
  await page.mouse.up();
  await expect.poll(pixels).not.toBe(zoomed);
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
