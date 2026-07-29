// Self-check for the in-browser forecaster: node docs/forecast.test.mjs
import assert from "node:assert/strict";
import { parseSeries, fit, forecast, backtest, prepSweep } from "./forecast.js";

const day = i => new Date(Date.UTC(2024, 0, 1) + i * 864e5).toISOString().slice(0, 10);

// --- parsing: sorts, skips headers, rejects junk and too-short series ---
const parsed = parseSeries(["date,units", ...Array.from({ length: 25 }, (_, i) => `${day(24 - i)},${i}`)].join("\n"));
assert.equal(parsed.length, 25);
assert.ok(parsed[0].date < parsed[1].date, "rows are sorted by date");
assert.throws(() => parseSeries("2024-01-01,5"), /at least 21 days/);
assert.throws(() => parseSeries(Array.from({ length: 25 }, (_, i) => `${day(i)},x`).join("\n")), /bad units/);

// --- fit recovers a known linear relationship ---
const w = fit([[1, 0], [1, 1], [1, 2], [1, 3]], [1, 3, 5, 7]);
assert.ok(Math.abs(w[0] - 1) < 1e-3 && Math.abs(w[1] - 2) < 1e-3, `expected y=1+2x, got ${w}`);

// --- forecast learns a pure weekday pattern (weekends 2x) ---
const weekly = Array.from({ length: 70 }, (_, i) => ({
  date: new Date(Date.UTC(2024, 0, 1) + i * 864e5),
  units: [10, 10, 10, 10, 10, 20, 20][i % 7],   // Jan 1 2024 is a Monday
}));
const fc = forecast(weekly);
assert.equal(fc.length, 7);
// Day 69 is a Sunday, so the horizon runs Monday..Sunday: index 5 is Saturday.
assert.ok(fc[5].predicted > 15, `Saturday should forecast high, got ${fc[5].predicted}`);
assert.ok(fc[0].predicted < 15, `Monday should forecast low, got ${fc[0].predicted}`);

// --- backtest beats seasonal-naive on a trending series it can actually learn ---
const trend = Array.from({ length: 90 }, (_, i) => ({
  date: new Date(Date.UTC(2024, 0, 1) + i * 864e5),
  units: 20 + i * 0.5 + (i % 7 >= 5 ? 8 : 0),
}));
const bt = backtest(trend);
assert.equal(bt.days.length, 7);
// The naive baseline must only quote actuals from before the cutoff.
assert.equal(bt.days[6].actual, trend[89].units);
assert.ok(bt.mae < bt.naiveMae, `model MAE ${bt.mae} should beat naive ${bt.naiveMae}`);

// --- prep sweep: a perfect forecast is cheapest at 0% margin, and beats max-prep ---
const perfect = [10, 12, 8, 14, 9].map((u, i) => ({ date: day(i), actual: u, predicted: u }));
const sweep = prepSweep(perfect, 500, 175);
assert.equal(sweep.best.margin, 0);
assert.equal(sweep.best.cost, 0);
assert.ok(sweep.baselineCost > 0, "max-prep baseline wastes money");
// Under-prepping is priced at the lost profit margin, not the food cost.
const under = [{ date: day(0), actual: 10, predicted: 0 }];
assert.equal(prepSweep(under, 500, 175).best.cost, 10 * (500 - 175));

console.log("forecast.js: all checks passed");
