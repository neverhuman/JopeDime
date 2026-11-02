"use strict";
// Fitted matrix inference. All coefficients and input dictionaries are immutable data.
const STATE = __STATE_JSON__;
function finite(x) { return Number.isFinite(x) ? x : 0.0; }
function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }
function sigmoid(x) {
  if (Number.isNaN(x)) x = -40.0;
  if (x >= 0.0) return 1.0 / (1.0 + Math.exp(-Math.min(x, 40.0)));
  const z = Math.exp(Math.max(x, -40.0)); return z / (1.0 + z);
}
function linear(model, row) {
  let value = model.weights[0];
  for (let j = 0; j < model.means.length; ++j) {
    const raw = model.finite_input ? finite(row[j]) : row[j];
    value += model.weights[j+1] * clamp((raw-model.means[j])/model.scales[j], -model.bound, model.bound);
  }
  return value;
}
function softmax(scores) {
  let maximum = -Infinity;
  for (const value of scores) maximum = Math.max(maximum, value);
  const values = scores.map(value => Math.exp(clamp(finite(value)-maximum, -40.0, 0.0)));
  let total = 0.0; for (const value of values) total += value;
  total = Math.max(total, 1e-300); return values.map(value => value / total);
}
function normalize(values) {
  let total = 0.0; for (const value of values) total += value;
  if (!Number.isFinite(total) || total <= 0.0) return values.map(() => 1.0 / values.length);
  return values.map(value => value / total);
}
function encode(row) {
  if (!Array.isArray(row) || row.length !== STATE.input.length || row.some(value => typeof value !== "string"))
    throw new Error("input shape differs");
  const out = [];
  for (let j = 0; j < row.length; ++j) {
    const raw = row[j], [kind, state] = Object.entries(STATE.input[j])[0];
    if (kind === "Numeric") {
      let absent = ["", "na", "nan", "null"].includes(raw.trim().toLowerCase());
      let value = absent ? state.fill : Number(raw);
      if (!absent && /^[+-]?inf(?:inity)?$/i.test(raw.trim())) value = Infinity;
      else if (!absent && !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw.trim()))
        throw new Error("invalid numeric input");
      absent = absent || !Number.isFinite(value);
      out.push(absent && state.restore_missing ? NaN : (absent ? state.fill : value));
      if (state.indicator) out.push(absent ? 1.0 : 0.0);
    } else {
      const key = ["", "na", "null"].includes(raw.trim().toLowerCase()) ? "\0MISSING" : raw;
      const found = state.vocabulary.indexOf(key);
      for (const index of state.indices) out.push(found === index ? 1.0 : 0.0);
    }
  }
  return out;
}
function distance(row, center, scales, squaredScale) {
  let total = 0.0;
  for (let j = 0; j < center.length; ++j) {
    let delta = finite(row[j])-center[j];
    if (squaredScale) total += delta*delta/scales[j];
    else { delta /= scales[j]; total += delta*delta; }
  }
  return total;
}
function predict(row) {
  const [kind, state] = Object.entries(STATE.head)[0];
  if (kind === "Regression") return [finite(clamp(linear(state,row), -60.0,60.0))];
  if (kind === "Binary") { const p=sigmoid(linear(state,row)); return [1.0-p,p]; }
  if (kind === "Softmax") return softmax(state.map(model => finite(linear(model,row))));
  if (kind === "Ovr") return normalize(state.map(model => Math.max(0.0,finite(sigmoid(linear(model,row))))));
  if (kind === "Lda") return softmax(state.centers.map((center,j) => state.priors[j]-0.5*distance(row,center,state.scales,true)));
  return softmax(state.centers.map(center => -distance(row,center,state.scales,false)));
}
function predictEncoded(rows) { return rows.map(row => predict(encode(row))); }
module.exports = { predictEncoded };
