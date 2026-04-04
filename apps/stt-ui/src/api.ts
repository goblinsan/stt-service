export interface Segment {
  id: number;
  start: number;
  end: number;
  text: string;
  words?: { word: string; start: number; end: number; probability: number }[];
  speaker?: string;
}

export interface TranscribeResult {
  text: string;
  language: string;
  language_probability: number;
  duration: number;
  segments: Segment[];
  processing_time: number;
}

export interface ServiceInfo {
  model_size: string;
  device: string;
  compute_type: string;
  model_loaded: boolean;
  gpu: { name: string; vram_total_mb: number; vram_used_mb: number } | null;
  max_upload_mb: number;
}

export async function fetchInfo(): Promise<ServiceInfo> {
  const res = await fetch('/api/info');
  if (!res.ok) throw new Error(`Failed to fetch info: ${res.status}`);
  return res.json();
}

export async function transcribe(
  file: File,
  options: {
    language?: string;
    task?: string;
    word_timestamps?: boolean;
    initial_prompt?: string;
  },
  onProgress?: (stage: string) => void,
): Promise<TranscribeResult> {
  const form = new FormData();
  form.append('file', file);
  if (options.language) form.append('language', options.language);
  if (options.task) form.append('task', options.task);
  if (options.word_timestamps) form.append('word_timestamps', 'true');
  if (options.initial_prompt) form.append('initial_prompt', options.initial_prompt);

  onProgress?.('Uploading...');
  const res = await fetch('/api/transcribe', { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Transcription failed: ${res.status}`);
  }
  return res.json();
}
