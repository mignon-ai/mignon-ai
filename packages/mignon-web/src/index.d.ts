export interface Token {
  text: string;
  /** Probability the model gave this token, 0 to 1. */
  prob: number;
}

export interface Reply {
  text: string;
  tokens: Token[];
  /** The weakest token probability: how sure the model was of the least certain piece. 1 for an empty reply. */
  minProb: number;
}

export interface ModelInfo {
  quantized: boolean;
  /** "chat" (answers prompts) or "text" (continues text). */
  task: string;
  params: number;
  vocab_size: number;
  context_size: number;
  blocks: number;
  state_size: number;
  hidden_size: number;
}

export type Source = string | URL | Response | ArrayBuffer | Uint8Array;

export interface LoadOptions {
  /** The model file (.mgn): a URL, a path in Node, a Response or the bytes. */
  model: Source;
  /** mignon-web.wasm; by default the one next to this package's script. */
  wasm?: Source;
}

export declare class Model {
  readonly info: ModelInfo;
  /** The JSON stored in the model file by whoever trained it, or null. */
  readonly extra: unknown;
  /** Synchronous: a reply takes 15 to 80 ms. */
  generate(prompt: string, options?: { maxTokens?: number }): Reply;
  /** Releases the model. Using it afterwards throws. */
  free(): void;
}

export declare function load(options: LoadOptions): Promise<Model>;
/** Version of the engine inside the wasm file. */
export declare function version(wasm?: Source): Promise<string>;
