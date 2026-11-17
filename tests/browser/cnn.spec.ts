import { gzipSync } from "node:zlib";
import { expect, test } from "@playwright/test";

function mnistFixtures() {
  const count = 800,
    images = Buffer.alloc(16 + count * 28 * 28),
    labels = Buffer.alloc(8 + count);
  images.writeUInt32BE(2051, 0);
  images.writeUInt32BE(count, 4);
  images.writeUInt32BE(28, 8);
  images.writeUInt32BE(28, 12);
  labels.writeUInt32BE(2049, 0);
  labels.writeUInt32BE(count, 4);
  for (let i = 0; i < count; i++) {
    const label = Math.floor(i / 80),
      start = 16 + i * 28 * 28;
    labels[8 + i] = label;
    for (let y = 4; y < 24; y++)
      for (let x = 4; x < 24; x++)
        if ((x + label) % 10 < 3 || (y + label) % 12 < 3)
          images[start + y * 28 + x] = 220;
  }
  return {
    images: gzipSync(images),
    labels: gzipSync(labels),
  };
}

test("CNN lab loads MNIST and exposes the complete forward pipeline", async ({
  page,
}) => {
  const errors: string[] = [],
    fixtures = mnistFixtures();
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route(
    "https://raw.githubusercontent.com/fgnt/mnist/master/**",
    async (route) => {
      const body = route.request().url().includes("images")
        ? fixtures.images
        : fixtures.labels;
      await route.fulfill({
        status: 200,
        headers: {
          "access-control-allow-origin": "*",
          "content-type": "application/gzip",
        },
        body,
      });
    },
  );
  await page.goto("/");
  await page.getByRole("button", { name: "CNN · MNIST", exact: true }).click();
  await page.getByRole("button", { name: "Carica MNIST", exact: true }).click();
  await expect(page.getByText(/800 immagini bilanciate/)).toBeVisible();
  for (const stage of [
    "1 · Immagine di input",
    "2 · Convolution layer 1",
    "3 · ReLU layer 1",
    "4 · Pooling layer 1",
    "5 · Repetition of feature extraction",
    "6 · Flattening layer",
    "7 · Fully connected (dense) layer",
    "8 · Output layer · Softmax",
  ])
    await expect(page.getByRole("heading", { name: stage })).toBeVisible();
  await expect(page.getByText("Dimensione vettore: 100")).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: "Calcolo dettagliato del pixel selezionato",
    }),
  ).toBeVisible();
  const convolutionMap = page.getByRole("img", {
    name: "Mappa di convoluzione 1",
  });
  const bounds = await convolutionMap.boundingBox();
  if (!bounds) throw new Error("Convolution map is not visible.");
  await convolutionMap.click({
    position: { x: bounds.width / 2, y: bounds.height / 2 },
  });
  await expect(page.locator("#cnn-pixel-row")).toHaveValue("13");
  await expect(page.locator("#cnn-pixel-column")).toHaveValue("13");
  await expect(
    page.getByRole("table", { name: "Patch immagine normalizzata (3×3)" }),
  ).toBeVisible();
  await expect(page.getByText("Convoluzione: somma + bias")).toBeVisible();
  const poolMap = page.getByRole("img", {
    name: "Mappa max-pooling 1 · 1",
  });
  const poolBounds = await poolMap.boundingBox();
  if (!poolBounds) throw new Error("Max-pooling map is not visible.");
  await poolMap.click({
    position: { x: poolBounds.width / 2, y: poolBounds.height / 2 },
  });
  await expect(page.locator("#cnn-pixel-stage")).toHaveValue("pool");
  await expect(page.locator("#cnn-pixel-row")).toHaveValue("6");
  await expect(page.locator("#cnn-pixel-column")).toHaveValue("6");
  await expect(page.getByText("massimo selezionato")).toBeVisible();
  await page.locator("#cnn-pixel-layer").selectOption("2");
  await expect(page.locator("#cnn-pixel-row")).toHaveValue("5");
  await expect(
    page.getByRole("table", { name: "Patch canale 1 (3×3)" }),
  ).toBeVisible();
  await expect(
    page.getByRole("table", { name: "Patch canale 4 (3×3)" }),
  ).toBeVisible();
  await page.locator("#cnn-pixel-layer").selectOption("1");
  await page.locator("#cnn-pixel-stage").selectOption("relu");
  await expect(page.locator("#cnn-pixel-row")).toHaveValue("12");
  await expect(page.locator("#cnn-pixel-column")).toHaveValue("12");
  await page.locator("#cnn-pooling").selectOption("average");
  await expect(
    page.getByRole("img", { name: "Mappa average pooling 1 · 1" }),
  ).toBeVisible();
  await expect(page.getByText("Valore pooling:")).toBeVisible();
  await page.locator("#cnn-dense-node").selectOption("1");
  await page
    .getByText("Mostra i 100 prodotti, la somma e il bias di questo neurone")
    .click();
  await expect(
    page.getByRole("columnheader", { name: "Attivazione precedente" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Un mini-batch", exact: true })
    .click();
  await expect(page.getByText(/mini-batch 1 \/ 200/)).toBeVisible();
  expect(errors).toEqual([]);
});
