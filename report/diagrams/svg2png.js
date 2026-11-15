// Render every diagram SVG in this folder to a PNG (used for the Word version). Run: node svg2png.js
const { chromium } = require('playwright'); const fs = require('fs'); const path = require('path');
(async () => {
  const dir = __dirname; const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: 2 });
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.svg'))) {
    await page.goto('file://' + path.join(dir, f)); const el = await page.$('svg');
    await el.screenshot({ path: path.join(dir, f.replace('.svg', '.png')) });
  }
  await browser.close();
})();
