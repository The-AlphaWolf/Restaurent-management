// Browser-side demand forecasting for user-pasted history.
//
// The trained models in models/ are fit to *this* project's menu and item
// encodings, so they cannot score a stranger's series. Instead we refit the
// repo's own feature recipe (lag_1, lag_7, rolling_7, day-of-week, trend) on
// whatever history the visitor pastes — small enough to solve in the page, and
// honest about what it is. Waste/cost maths mirrors src/waste_optimizer.py.
//
// ponytail: ridge least squares, not the tuned tree ensemble. Swap in a real
// model only if someone needs per-item accuracy from a browser.

export const HORIZON = 7;
// Hold out exactly one week: any longer and the seasonal-naive baseline starts
// quoting actuals from inside the held-out window, which the model never sees.
export const BACKTEST_DAYS = 7;
const RIDGE = 1e-6;

/** Parse "date,units" lines (header optional). Throws on unusable input. */
export function parseSeries(text) {
  const rows = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /^(date|day)\b/i.test(line)) continue;
    const [d, u] = line.split(/[,;\t]/).map(s => (s || "").trim());
    const date = new Date(d + "T00:00:00Z");
    const units = Number(u);
    if (isNaN(date.getTime())) throw new Error(`bad date: "${d}"`);
    if (!isFinite(units) || units < 0) throw new Error(`bad units on ${d}: "${u}"`);
    rows.push({ date, units });
  }
  rows.sort((a, b) => a.date - b.date);
  if (rows.length < 21) throw new Error(`need at least 21 days, got ${rows.length}`);
  return rows;
}

const dow = date => (date.getUTCDay() + 6) % 7;              // 0 = Monday
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

/** Feature row for day index i of `units` (needs i >= 7). */
function features(units, dates, i) {
  const f = [1, units[i - 1], units[i - 7], mean(units.slice(i - 7, i)), i / 100];
  const d = dow(dates[i]);
  for (let k = 1; k < 7; k++) f.push(d === k ? 1 : 0);       // Monday is the reference day
  return f;
}

/** Ridge least squares via normal equations + Gaussian elimination. */
export function fit(X, y) {
  const p = X[0].length;
  const A = Array.from({ length: p }, () => new Float64Array(p + 1));
  for (let r = 0; r < p; r++) {
    for (let c = 0; c < p; c++) {
      let s = r === c ? RIDGE : 0;
      for (let i = 0; i < X.length; i++) s += X[i][r] * X[i][c];
      A[r][c] = s;
    }
    let s = 0;
    for (let i = 0; i < X.length; i++) s += X[i][r] * y[i];
    A[r][p] = s;
  }
  for (let col = 0; col < p; col++) {                         // partial pivoting
    let piv = col;
    for (let r = col + 1; r < p; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    if (Math.abs(A[col][col]) < 1e-12) continue;              // singular column -> zero weight
    for (let r = 0; r < p; r++) {
      if (r === col) continue;
      const factor = A[r][col] / A[col][col];
      for (let c = col; c <= p; c++) A[r][c] -= factor * A[col][c];
    }
  }
  return Array.from({ length: p }, (_, r) => (Math.abs(A[r][r]) < 1e-12 ? 0 : A[r][p] / A[r][r]));
}

/** Train on `rows`, then roll the forecast forward `days` days. */
export function forecast(rows, days = HORIZON) {
  const units = rows.map(r => r.units);
  const dates = rows.map(r => r.date);
  const X = [], y = [];
  for (let i = 7; i < rows.length; i++) { X.push(features(units, dates, i)); y.push(units[i]); }
  const w = fit(X, y);

  const out = [];
  for (let h = 1; h <= days; h++) {
    const next = new Date(dates[dates.length - 1].getTime() + 864e5);
    dates.push(next);
    const f = features(units, dates, units.length);
    const pred = Math.max(0, f.reduce((s, v, k) => s + v * w[k], 0));
    units.push(pred);                                          // recursive: feed prediction back
    out.push({ date: next, predicted: pred });
  }
  return out;
}

/** Fit on all but the last `days`, forecast them, and compare with seasonal-naive. */
export function backtest(rows, days = BACKTEST_DAYS) {
  const cut = rows.length - days;
  const held = rows.slice(cut);
  const preds = forecast(rows.slice(0, cut), days);
  const naive = held.map((r, i) => rows[cut + i - 7].units);   // same weekday last week
  const mae = arr => mean(arr.map(Math.abs));
  return {
    days: held.map((r, i) => ({ date: r.date, actual: r.units, predicted: preds[i].predicted })),
    mae: mae(held.map((r, i) => r.units - preds[i].predicted)),
    naiveMae: mae(held.map((r, i) => r.units - naive[i])),
  };
}

/**
 * Sweep safety margins over the backtest window and price both failure modes,
 * exactly as src/waste_optimizer.py does: waste costs the ingredients, a
 * stockout costs the profit margin.
 */
export function prepSweep(days, sellingPrice, foodCost) {
  const profit = sellingPrice - foodCost;
  const rows = [];
  for (let m = 0; m <= 0.5001; m += 0.05) {
    let waste = 0, stockout = 0;
    for (const d of days) {
      const prep = Math.ceil(Number((d.predicted * (1 + m)).toFixed(6)));
      waste += Math.max(0, prep - d.actual);
      stockout += Math.max(0, d.actual - prep);
    }
    rows.push({
      margin: Number(m.toFixed(2)), waste, stockout,
      cost: waste * foodCost + stockout * profit,
    });
  }
  // Baseline a cautious kitchen actually runs: prep the last 14 days' max.
  const maxSeen = Math.max(...days.map(d => d.actual));
  let bWaste = 0, bStock = 0;
  for (const d of days) { bWaste += Math.max(0, maxSeen - d.actual); bStock += 0; }
  const baselineCost = bWaste * foodCost + bStock * profit;
  const best = rows.reduce((a, b) => (b.cost < a.cost ? b : a));
  return { rows, best, baselineCost, baselineWaste: bWaste };
}
