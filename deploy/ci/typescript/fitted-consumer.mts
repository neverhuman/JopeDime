import {
  predict,
  predictBatch,
  type PredictorRow,
} from "./predictor.mjs";

const example: PredictorRow = {
  age: null,
  region: "north",
  'quote"field': 2,
};

const one: number = predict(example);
const many: number[] = predictBatch([example]);
void one;
void many;
