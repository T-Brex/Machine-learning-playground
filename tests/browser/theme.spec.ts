import { expect, test } from "@playwright/test";

test("theme colors update controls and charts and persist in a cookie", async ({
  page,
  context,
}) => {
  await page.goto("/");
  await expect(page.locator("#step")).toBeEnabled();
  await page.locator("#step").click();
  await expect(page.locator("#status")).toContainText("passo 1 /");
  const chart = page.getByRole("img", { name: "Loss per epoca", exact: true });
  const original = await chart.evaluate((canvas: HTMLCanvasElement) =>
    canvas.toDataURL(),
  );
  await page.getByLabel("Personalizza colori").click();
  await page.getByLabel("Colore primario", { exact: true }).fill("#e8a344");
  await page
    .getByLabel("Colore evidenziazione", { exact: true })
    .fill("#eb83b5");
  await expect(page.locator("#train")).toHaveCSS(
    "background-color",
    "rgb(232, 163, 68)",
  );
  await expect(page.locator("nav button.active")).toHaveCSS(
    "color",
    "rgb(235, 131, 181)",
  );
  await expect
    .poll(() =>
      chart.evaluate((canvas: HTMLCanvasElement) => canvas.toDataURL()),
    )
    .not.toBe(original);
  const cookie = (await context.cookies()).find(
    (item) => item.name === "ml-playground-colors",
  );
  expect(JSON.parse(decodeURIComponent(cookie!.value))).toEqual({
    primary: "#e8a344",
    highlight: "#eb83b5",
  });
  await page.reload();
  await expect(page.locator("#train")).toHaveCSS(
    "background-color",
    "rgb(232, 163, 68)",
  );
  await page.getByLabel("Personalizza colori").click();
  await expect(
    page.getByLabel("Colore evidenziazione", { exact: true }),
  ).toHaveValue("#eb83b5");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByLabel("Colore primario", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await page
    .getByRole("button", { name: "Ripristina colori", exact: true })
    .click();
  await expect(page.getByLabel("Colore primario", { exact: true })).toHaveValue(
    "#54cbbb",
  );
});

test("malformed theme cookies fall back to defaults", async ({
  page,
  context,
}) => {
  await context.addCookies([
    {
      name: "ml-playground-colors",
      value: "%invalid",
      url: "http://127.0.0.1:5173",
    },
  ]);
  await page.goto("/");
  await page.getByLabel("Personalizza colori").click();
  await expect(page.getByLabel("Colore primario", { exact: true })).toHaveValue(
    "#54cbbb",
  );
});
