const RAW_CATEGORICAL = __KINDS__;
const RAW_CLASSES = __CLASSES__;
const RAW_REGRESSION = __REGRESSION__;
function rawTrim(value) {
  return value.replace(/^[\u0009-\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g, "");
}
function validUnicode(value) {
  for (let i=0; i<value.length; ++i) {
    const code=value.charCodeAt(i);
    if (code>=0xd800 && code<=0xdbff) {
      const next=value.charCodeAt(++i);
      if (!(next>=0xdc00 && next<=0xdfff)) return false;
    } else if (code>=0xdc00 && code<=0xdfff) return false;
  }
  return true;
}
function predictRaw(rows) {
  if (!Array.isArray(rows)) throw new Error("rows must be an array");
  const encoded=rows.map(row => {
    if (!Array.isArray(row) || row.length!==RAW_CATEGORICAL.length) throw new Error("input shape differs");
    return row.map((value,j) => {
      if (value===null) return "";
      if (typeof value!=="string") throw new Error("features must be strings or null");
      if (RAW_CATEGORICAL[j]) {
        if (!validUnicode(value)) throw new Error("invalid Unicode input");
        return "jope.category.v1:"+JSON.stringify(value);
      }
      const trimmed=rawTrim(value);
      if (!/^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/.test(trimmed) || !Number.isFinite(Number(trimmed)))
        throw new Error("numeric features must be finite or null");
      return trimmed;
    });
  });
  const matrix=predictEncoded(encoded);
  const width=RAW_REGRESSION ? 1 : RAW_CLASSES.length;
  if (matrix.length!==rows.length || matrix.some(row => row.length!==width || row.some(v => !Number.isFinite(v))))
    throw new Error("fitted output shape differs");
  if (RAW_REGRESSION) return {kind:"values",data:matrix.map(row=>row[0])};
  if (matrix.some(row => row.some(v=>v<0.0 || v>1.0) || Math.abs(row.reduce((a,v)=>a+v,0.0)-1.0)>1e-6))
    throw new Error("invalid fitted probability matrix");
  return {kind:"probabilities",data:{classes:RAW_CLASSES.slice(),values:matrix}};
}
module.exports = { predictRaw };
