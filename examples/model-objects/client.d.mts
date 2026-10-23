export type ExportFormat = 'object' | 'fitted' | 'bound-json' | 'python' | 'rust' | 'javascript' | 'onnx';
export interface Inspection {
  schema: 'jope.model-object-inspection/v1'; id: string; integrity_verified: boolean; executable: boolean;
  unsupported_capabilities: string[]; format: string; primary: string;
  files: Array<{path: string; media_type: string; bytes: number; sha256: string}>;
  metadata: unknown;
}
export interface FitReceipt {
  outcome: string;
  outputs: Array<{role: string; path: string; bytes: number; sha256: string}>;
}
export class JopeModel {
  constructor(path: string, executable?: string);
  path: string;
  executable: string;
  static fit(options: {train: string; target: string; task?: 'auto' | 'regression' | 'binary' | 'multiclass'; method?: string; root?: string; executable?: string}): {model: JopeModel; receipt: FitReceipt};
  inspect(): Inspection;
  verify(): Inspection;
  predict(input: string, output: string): FitReceipt;
  export(format: ExportFormat, output: string): unknown;
}
export function saveState(options: {
  format: string; primary: string; files: Record<string, {bytes: Uint8Array; mediaType: string}>;
  metadata?: unknown; requiredCapabilities?: string[]; root?: string; executable?: string;
}): {path: string; inspection: Inspection};
