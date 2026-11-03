import { expect, test } from "@playwright/test";

test("all five languages update the interface without resetting the experiment", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator("#sampleCount").fill("140");
  await page.locator("#step").click();
  await expect(page.locator("#status")).toContainText("passo 1 /");
  const selector = page.locator(".language-picker select");
  for (const [language, heading, samples, colors, lang] of [
    ["en", "Model configuration", "140 samples", "Colors", "en"],
    ["ru", "Настройка модели", "140 объектов", "Цвета", "ru"],
    ["es", "Configuración del modelo", "140 muestras", "Colores", "es"],
    ["zh", "模型配置", "140 样本", "颜色", "zh-CN"],
    ["it", "Configurazione modello", "140 campioni", "Colori", "it"],
  ]) {
    await selector.selectOption(language);
    await expect(page.locator(".model-panel .ph")).toContainText(heading);
    await expect(page.locator(".dataset-summary span").first()).toHaveText(
      samples,
    );
    await expect(page.locator(".theme-picker summary")).toContainText(colors);
    await expect(page.locator("html")).toHaveAttribute("lang", lang);
    await expect(page.locator("#sampleCount")).toHaveValue("140");
    await expect(page.locator("#status")).toContainText("1 / 120");
  }
  await selector.selectOption("en");
  await page.reload();
  await expect(selector).toHaveValue("en");
  await expect(page.locator(".model-panel .ph")).toContainText(
    "Model configuration",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(selector).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await page.locator(".theme-picker summary").click();
  await expect(page.getByLabel("Primary color", { exact: true })).toBeVisible();
});
