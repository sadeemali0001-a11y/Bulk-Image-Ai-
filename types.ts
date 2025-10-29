export interface GeneratedImage {
  prompt: string;
  base64?: string;
  url?: string;
  error?: string;
  engine?: 'google' | 'stability';
}

export type AspectRatio = "1:1" | "16:9" | "9:16" | "4:3" | "3:4";

export enum AppState {
  IDLE,
  GENERATING_PROMPTS,
  GENERATING_IMAGES,
  DONE,
}