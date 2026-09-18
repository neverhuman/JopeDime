// deploy/ci/emitters/driver.js — appended after an emitted model.js by deploy/ci/emitters.sh.
// Usage: node program.js train.csv test_features.csv predictions   (headerless CSV; the last train
// column is the target). Fits and predicts through the emitted ES2020 API and writes one
// prediction per line in JavaScript's shortest round-trip form.
const fs = require("fs");
function readCsv(path) {
  const cells = fs.readFileSync(path, "utf8").split("\n").filter((line) => line.trim().length > 0).map((line) => line.split(",").map(Number));
  const columns = cells[0].length;
  for (const row of cells) if (row.length !== columns) throw new Error(`ragged csv ${path}`);
  return { cells, rows: cells.length, columns };
}
const [trainPath, testPath, outPath] = process.argv.slice(2);
if (!outPath) { console.error("usage: node program.js train.csv test_features.csv predictions"); process.exit(2); }
const train = readCsv(trainPath);
const test = readCsv(testPath);
const columns = train.columns - 1;
if (columns < 1 || test.columns !== columns) process.exit(4);
const x = new Float64Array(train.rows * columns);
const y = new Float64Array(train.rows);
train.cells.forEach((row, r) => { for (let c = 0; c < columns; c++) x[r * columns + c] = row[c]; y[r] = row[columns]; });
const t = new Float64Array(test.rows * columns);
test.cells.forEach((row, r) => { for (let c = 0; c < columns; c++) t[r * columns + c] = row[c]; });
const fitted = modelFitNumeric(x, x.length, train.rows, columns, y, y.length);
if (fitted.status !== 0 || fitted.handle === null) process.exit(10);
const out = new Float64Array(test.rows);
if (modelPredictNumericBatch(fitted.handle, t, t.length, test.rows, columns, null, 0, out, out.length) !== 0) process.exit(11);
modelUnload(fitted.handle);
fs.writeFileSync(outPath, Array.from(out, (value) => String(value)).join("\n") + "\n");
