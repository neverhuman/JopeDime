#!/usr/bin/env node
// Independent JavaScript (Node.js ESM) conformance verifier for jope.canonical-json/v1 (I02).
// Implements an independent recursive-descent strict JSON parser, canonical serializer,
// length-prefixed domain-separated SHA-256 hasher, and negative-vector refusal verifier.

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SAFE_INT_MIN = -9007199254740991n;
const SAFE_INT_MAX = 9007199254740991n;
const FORBIDDEN_CYCLE_KEYS = new Set([
  "own_id",
  "signature",
  "later_git_commit",
  "later_assessment_id",
]);

class WireError extends Error {
  constructor(kind, detail) {
    super(`${kind}: ${detail}`);
    this.kind = kind;
  }
}

function parseStrictJson(rawStr) {
  const buf = Buffer.from(rawStr, "utf8");
  if (buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf) {
    throw new WireError("ByteOrderMark", "UTF-8 BOM is forbidden");
  }
  let pos = 0;

  function skipWs() {
    while (pos < rawStr.length) {
      const c = rawStr.charCodeAt(pos);
      if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) {
        pos += 1;
      } else {
        break;
      }
    }
  }

  function parseString() {
    pos += 1; // '"'
    let out = "";
    while (pos < rawStr.length) {
      const ch = rawStr[pos];
      const code = rawStr.charCodeAt(pos);
      if (ch === '"') {
        pos += 1;
        return out;
      }
      if (code < 0x20) {
        throw new WireError("UnescapedControlChar", `control char at ${pos}`);
      }
      if (ch === "\\") {
        pos += 1;
        if (pos >= rawStr.length) {
          throw new WireError("UnexpectedEof", "truncated escape");
        }
        const esc = rawStr[pos];
        pos += 1;
        switch (esc) {
          case '"':
            out += '"';
            break;
          case "\\":
            out += "\\";
            break;
          case "/":
            out += "/";
            break;
          case "b":
            out += "\b";
            break;
          case "f":
            out += "\f";
            break;
          case "n":
            out += "\n";
            break;
          case "r":
            out += "\r";
            break;
          case "t":
            out += "\t";
            break;
          case "u": {
            const hex4 = rawStr.slice(pos, pos + 4);
            if (!/^[0-9a-fA-F]{4}$/.test(hex4)) {
              throw new WireError("InvalidStringEscape", `bad \\u${hex4}`);
            }
            pos += 4;
            const unit = Number.parseInt(hex4, 16);
            if (unit >= 0xd800 && unit <= 0xdbff) {
              if (rawStr.slice(pos, pos + 2) !== "\\u") {
                throw new WireError("UnpairedSurrogate", `unpaired high surrogate ${hex4}`);
              }
              const lowHex = rawStr.slice(pos + 2, pos + 6);
              if (!/^[0-9a-fA-F]{4}$/.test(lowHex)) {
                throw new WireError("InvalidStringEscape", `bad low \\u${lowHex}`);
              }
              const lowUnit = Number.parseInt(lowHex, 16);
              if (lowUnit < 0xdc00 || lowUnit > 0xdfff) {
                throw new WireError("UnpairedSurrogate", `invalid low surrogate ${lowHex}`);
              }
              pos += 6;
              const cp = 0x10000 + (((unit - 0xd800) << 10) | (lowUnit - 0xdc00));
              out += String.fromCodePoint(cp);
            } else if (unit >= 0xdc00 && unit <= 0xdfff) {
              throw new WireError("UnpairedSurrogate", `unpaired low surrogate ${hex4}`);
            } else {
              out += String.fromCodePoint(unit);
            }
            break;
          }
          default:
            throw new WireError("InvalidStringEscape", `unknown escape \\${esc}`);
        }
      } else {
        const cp = rawStr.codePointAt(pos);
        const s = String.fromCodePoint(cp);
        out += s;
        pos += s.length;
      }
    }
    throw new WireError("UnexpectedEof", "unterminated string");
  }

  function parseNumber(jsonPath, allowDisplayFloat) {
    const start = pos;
    if (rawStr[pos] === "-") {
      pos += 1;
    }
    const digitStart = pos;
    while (pos < rawStr.length && rawStr[pos] >= "0" && rawStr[pos] <= "9") {
      pos += 1;
    }
    if (pos === digitStart) {
      throw new WireError("SyntaxError", "missing digits");
    }
    let isFloat = false;
    if (rawStr[pos] === ".") {
      isFloat = true;
      pos += 1;
      const fracStart = pos;
      while (pos < rawStr.length && rawStr[pos] >= "0" && rawStr[pos] <= "9") {
        pos += 1;
      }
      if (pos === fracStart) {
        throw new WireError("SyntaxError", "missing fraction digits");
      }
    }
    if (rawStr[pos] === "e" || rawStr[pos] === "E") {
      isFloat = true;
      pos += 1;
      if (rawStr[pos] === "+" || rawStr[pos] === "-") {
        pos += 1;
      }
      const expStart = pos;
      while (pos < rawStr.length && rawStr[pos] >= "0" && rawStr[pos] <= "9") {
        pos += 1;
      }
      if (pos === expStart) {
        throw new WireError("SyntaxError", "missing exponent digits");
      }
    }
    const token = rawStr.slice(start, pos);
    if (allowDisplayFloat) {
      const f = Number(token);
      if (!Number.isFinite(f)) {
        throw new WireError("NonFiniteFloatBits", `non-finite display_f64 at ${jsonPath}`);
      }
      return { kind: "display_f64", raw: token };
    }
    if (token === "-0") {
      throw new WireError("NegativeZeroInteger", "-0 is forbidden");
    }
    const digits = rawStr.slice(digitStart, pos);
    if (!isFloat && digits.length > 1 && digits[0] === "0") {
      throw new WireError("LeadingZeroNumber", `leading zero in ${token}`);
    }
    if (isFloat) {
      throw new WireError("RawFloatInIdentityPayload", `raw float at ${jsonPath}`);
    }
    const bi = BigInt(token);
    if (bi < SAFE_INT_MIN || bi > SAFE_INT_MAX) {
      throw new WireError("IntegerOverflow", `${token} outside safe range at ${jsonPath}`);
    }
    return { kind: "int", value: bi };
  }

  function parseValue(jsonPath) {
    skipWs();
    if (pos >= rawStr.length) {
      throw new WireError("UnexpectedEof", "unexpected EOF");
    }
    const ch = rawStr[pos];
    if (ch === "{") {
      pos += 1;
      skipWs();
      const entries = new Map();
      if (rawStr[pos] === "}") {
        pos += 1;
        return { kind: "object", entries };
      }
      while (true) {
        skipWs();
        if (rawStr[pos] !== '"') {
          throw new WireError("SyntaxError", "expected string key");
        }
        const key = parseString();
        if (entries.has(key)) {
          throw new WireError("DuplicateObjectKey", `duplicate key ${key} at ${jsonPath || "/"}`);
        }
        skipWs();
        if (rawStr[pos] !== ":") {
          throw new WireError("SyntaxError", "expected ':'");
        }
        pos += 1;
        const childPath = `${jsonPath}/${key}`;
        const val =
          key === "display_f64"
            ? (skipWs(), parseNumber(childPath, true))
            : parseValue(childPath);
        entries.set(key, val);
        skipWs();
        if (rawStr[pos] === ",") {
          pos += 1;
          skipWs();
          if (rawStr[pos] === "}") {
            throw new WireError("SyntaxError", "trailing comma in object");
          }
        } else if (rawStr[pos] === "}") {
          pos += 1;
          break;
        } else {
          throw new WireError("SyntaxError", "expected ',' or '}'");
        }
      }
      return { kind: "object", entries };
    }
    if (ch === "[") {
      pos += 1;
      skipWs();
      const items = [];
      if (rawStr[pos] === "]") {
        pos += 1;
        return { kind: "array", items };
      }
      let idx = 0;
      while (true) {
        items.push(parseValue(`${jsonPath}/${idx}`));
        idx += 1;
        skipWs();
        if (rawStr[pos] === ",") {
          pos += 1;
          skipWs();
          if (rawStr[pos] === "]") {
            throw new WireError("SyntaxError", "trailing comma in array");
          }
        } else if (rawStr[pos] === "]") {
          pos += 1;
          break;
        } else {
          throw new WireError("SyntaxError", "expected ',' or ']'");
        }
      }
      return { kind: "array", items };
    }
    if (ch === '"') {
      return { kind: "string", value: parseString() };
    }
    if (rawStr.startsWith("true", pos)) {
      pos += 4;
      return { kind: "bool", value: true };
    }
    if (rawStr.startsWith("false", pos)) {
      pos += 5;
      return { kind: "bool", value: false };
    }
    if (rawStr.startsWith("null", pos)) {
      pos += 4;
      return { kind: "null" };
    }
    if (ch === "-" || (ch >= "0" && ch <= "9")) {
      return parseNumber(jsonPath, false);
    }
    throw new WireError("SyntaxError", `unexpected char ${ch}`);
  }

  const root = parseValue("");
  skipWs();
  if (pos !== rawStr.length) {
    throw new WireError("TrailingBytes", `trailing bytes at ${pos}`);
  }
  return root;
}

function checkLowerHex(jsonPath, val, expectedLen) {
  if (
    typeof val !== "string" ||
    val.length !== expectedLen ||
    !/^[0-9a-f]+$/.test(val)
  ) {
    throw new WireError("InvalidHexEncoding", `bad ${expectedLen}-hex at ${jsonPath}`);
  }
}

function validateAndProject(node, jsonPath = "") {
  const here = jsonPath || "/";
  if (node.kind === "array") {
    node.items.forEach((it, idx) => validateAndProject(it, `${jsonPath}/${idx}`));
    return;
  }
  if (node.kind !== "object") {
    return;
  }
  const map = node.entries;
  for (const forbidden of FORBIDDEN_CYCLE_KEYS) {
    if (map.has(forbidden)) {
      throw new WireError("ForbiddenCycleField", `forbidden ${forbidden} at ${here}`);
    }
  }
  if (map.has("f64_bits") || map.has("f32_bits")) {
    map.delete("display_f64");
  }
  for (const [k, v] of map.entries()) {
    validateAndProject(v, `${jsonPath}/${k}`);
  }
  if (map.has("f64_bits")) {
    if (map.size !== 1) {
      throw new WireError("UnknownField", `extra field in exact_f64 at ${here}`);
    }
    const hex = map.get("f64_bits").value;
    checkLowerHex(`${here}/f64_bits`, hex, 16);
    const bits = BigInt(`0x${hex}`);
    if ((bits & 0x7ff0000000000000n) === 0x7ff0000000000000n) {
      throw new WireError("NonFiniteFloatBits", `non-finite f64 ${hex} at ${here}`);
    }
  }
  if (map.has("f32_bits")) {
    if (map.size !== 1) {
      throw new WireError("UnknownField", `extra field in exact_f32 at ${here}`);
    }
    const hex = map.get("f32_bits").value;
    checkLowerHex(`${here}/f32_bits`, hex, 8);
    const bits = BigInt(`0x${hex}`);
    if ((bits & 0x7f800000n) === 0x7f800000n) {
      throw new WireError("NonFiniteFloatBits", `non-finite f32 ${hex} at ${here}`);
    }
  }
  if (map.has("seed_u64_hex")) {
    checkLowerHex(`${here}/seed_u64_hex`, map.get("seed_u64_hex").value, 16);
  }
  if (map.has("outcome")) {
    validateOutcome(map.get("outcome"), `${jsonPath}/outcome`);
  }
}

function validateOutcome(node, jsonPath) {
  const map = node.entries;
  const allowed = new Set([
    "authority",
    "metric_name",
    "metric",
    "evidence_receipt_sha256",
    "evidence_kind",
    "model_bundle_sha256",
    "synthetic_fixture_id",
    "missing_reason",
  ]);
  for (const k of map.keys()) {
    if (!allowed.has(k)) {
      throw new WireError("UnknownField", `unknown outcome field ${k} at ${jsonPath}`);
    }
  }
  const auth = map.get("authority")?.value;
  if (!["verified_measured", "predicted", "synthetic", "missing"].includes(auth)) {
    throw new WireError("UnknownEnumVariant", `unknown authority ${auth}`);
  }
  if (auth === "missing" && map.has("metric")) {
    throw new WireError("InvalidOutcomeAuthority", "missing outcome cannot contain metric");
  }
}

function canonString(str) {
  let out = '"';
  for (const ch of str) {
    const cp = ch.codePointAt(0);
    if (ch === '"') {
      out += '\\"';
    } else if (ch === "\\") {
      out += "\\\\";
    } else if (cp <= 0x1f) {
      out += `\\u${cp.toString(16).padStart(4, "0")}`;
    } else {
      out += ch;
    }
  }
  out += '"';
  return out;
}

function compareUnicodeScalarOrder(a, b) {
  return Buffer.compare(Buffer.from(a, "utf8"), Buffer.from(b, "utf8"));
}

function serializeCanonical(node) {
  switch (node.kind) {
    case "null":
      return "null";
    case "bool":
      return node.value ? "true" : "false";
    case "int":
      return node.value.toString(10);
    case "string":
      return canonString(node.value);
    case "array":
      return `[${node.items.map(serializeCanonical).join(",")}]`;
    case "object": {
      const sortedKeys = Array.from(node.entries.keys()).sort(compareUnicodeScalarOrder);
      const parts = sortedKeys.map(
        (k) => `${canonString(k)}:${serializeCanonical(node.entries.get(k))}`,
      );
      return `{${parts.join(",")}}`;
    }
    default:
      throw new Error(`unexpected node kind ${node.kind}`);
  }
}

function computeDomainSeparatedSha256(schemaId, canonicalPayloadStr) {
  const payloadBytes = Buffer.from(canonicalPayloadStr, "utf8");
  const lenBuf = Buffer.alloc(8);
  lenBuf.writeBigUInt64BE(BigInt(payloadBytes.length), 0);
  const hash = crypto.createHash("sha256");
  hash.update(Buffer.from(schemaId, "utf8"));
  hash.update(Buffer.from([0x00]));
  hash.update(lenBuf);
  hash.update(payloadBytes);
  return {
    byteLen: payloadBytes.length,
    sha256: hash.digest("hex"),
  };
}

const hereDir = path.dirname(fileURLToPath(import.meta.url));
const posDoc = JSON.parse(
  fs.readFileSync(path.join(hereDir, "positive_vectors.json"), "utf8"),
);
const negDoc = JSON.parse(
  fs.readFileSync(path.join(hereDir, "negative_vectors.json"), "utf8"),
);

let posPassed = 0;
for (const vec of posDoc.vectors) {
  const ast = parseStrictJson(vec.raw_input_json);
  validateAndProject(ast, "");
  const canonical = serializeCanonical(ast);
  if (canonical !== vec.expected_canonical_payload) {
    throw new Error(
      `${vec.id}: canonical payload mismatch:\n  expected: ${vec.expected_canonical_payload}\n  actual:   ${canonical}`,
    );
  }
  const { byteLen, sha256 } = computeDomainSeparatedSha256(vec.schema_id, canonical);
  if (byteLen !== vec.expected_payload_byte_len) {
    throw new Error(
      `${vec.id}: byte length mismatch: expected ${vec.expected_payload_byte_len}, got ${byteLen}`,
    );
  }
  if (sha256 !== vec.expected_domain_separated_sha256) {
    throw new Error(
      `${vec.id}: sha256 mismatch: expected ${vec.expected_domain_separated_sha256}, got ${sha256}`,
    );
  }
  posPassed += 1;
}

let negPassed = 0;
for (const vec of negDoc.vectors) {
  let caughtKind = null;
  try {
    const ast = parseStrictJson(vec.raw_input_json);
    validateAndProject(ast, "");
  } catch (err) {
    caughtKind = err.kind || "UnknownError";
  }
  if (caughtKind !== vec.expected_error_kind) {
    throw new Error(
      `${vec.id}: expected error ${vec.expected_error_kind}, got ${caughtKind}`,
    );
  }
  negPassed += 1;
}

console.log(
  JSON.stringify({
    verifier: "node-esm-independent-wire-contract-v1",
    positive_vectors_verified: posPassed,
    negative_vectors_verified: negPassed,
    status: "OK",
  }),
);
