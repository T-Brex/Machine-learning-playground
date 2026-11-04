import { expect, test } from "@playwright/test";

test("training, cancellation, task switching, tutorial, CSV and mobile navigation", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByText("ML Playground", { exact: true })).toBeVisible();
  // Worker training is much faster than the old frame-paced loop. Use a long
  // run so this test deliberately exercises pause/resume rather than completion.
  await page.locator("#methodParam_epochs").fill("2000");
  await page.locator("#step").click();
  await expect(page.locator("#status")).toContainText("passo 1 /");
  await page.locator("#train").click();
  await expect(page.locator("#trainingPause")).toBeEnabled();
  await page.locator("#trainingPause").click();
  await expect(page.locator("#trainingPause")).toBeDisabled();
  // Let more than one training interval elapse: a paused timer must publish no
  // additional iterations, even though the numerical model is retained to resume.
  const paused = await page.locator("#status").textContent();
  await page.waitForTimeout(150);
  await expect(page.locator("#status")).toHaveText(paused ?? "");
  await page.locator("#reset").click();
  await expect(page.locator("#status")).toContainText("passo 0 /");
  await page.locator("#dataset").selectOption("threeblobs");
  await expect(page.locator("#algorithm")).toHaveValue("softmax");
  await page.locator("#step").click();
  await page
    .getByRole("button", { name: "Guida didattica", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Regressione softmax", exact: true }),
  ).toBeVisible();
  await page.locator("#tutorialSampleIndex").fill("2");
  await page.getByRole("button", { name: "Avanti →", exact: true }).click();
  await page
    .getByRole("button", { name: "Formule e metriche", exact: true })
    .click();
  await expect(
    page.getByText("Metriche sul set di validazione", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Regressione", exact: true }).click();
  await page.getByRole("button", { name: "Panoramica", exact: true }).click();
  await page.locator("#csvFile").setInputFiles({
    name: "regression.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("x1,x2,y\n0,1,10\n1,2,20\n2,3,30\n3,4,40\n4,5,50"),
  });
  await expect(page.locator("#sampleCount")).toBeDisabled();
  await page.locator("#step").click();
  await page.getByRole("button", { name: "Clustering", exact: true }).click();
  await page.locator("#algorithm").selectOption("dbscan");
  await page.locator("#train").click();
  await expect(page.locator("#status")).toContainText("Simulazione completata");
  await page
    .getByRole("button", { name: "Ottimizzatori", exact: true })
    .click();
  await page.locator("#step").click();
  await expect(
    page.getByText("Pesi e bias · gradienti", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: "test-results/desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  // Mobile navigation must remain available, and wide tables must scroll inside
  // their panels rather than making the whole document wider than the viewport.
  await page.getByRole("button", { name: "Dataset", exact: true }).click();
  await expect(page.locator("#dataset")).toBeVisible();
  await expect(
    page.getByRole("navigation", { name: "Navigazione principale" }),
  ).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
  expect(errors).toEqual([]);
});
