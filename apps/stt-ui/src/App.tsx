import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchInfo, transcribe, type ServiceInfo, type TranscribeResult, type SpeakerSummary } from './api';

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
}

/** Generate a deterministic hue from a speaker label string. */
function speakerHue(label: string): number {
  let hash = 0;
  for (let i = 0; i < label.length; i++) hash = label.charCodeAt(i) + ((hash << 5) - hash);
  return Math.abs(hash) % 360;
}

function DropZone({ onFile, disabled }: { onFile: (f: File) => void; disabled: boolean }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile, disabled],
  );

  return (
    <div
      className={`drop-zone ${dragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.webm,.mp4,.aac,.wma"
        style={{ display: 'none' }}
        onChange={(e) => { if (e.target.files?.[0]) onFile(e.target.files[0]); }}
      />
      <div className="drop-icon">🎙</div>
      <div className="drop-label">Drop an audio file here or click to browse</div>
      <div className="drop-hint">WAV, MP3, FLAC, OGG, M4A, WebM, MP4</div>
    </div>
  );
}

function Options({
  language, setLanguage,
  task, setTask,
  wordTimestamps, setWordTimestamps,
  diarize, setDiarize,
  diarizationAvailable,
  minSpeakers, setMinSpeakers,
  maxSpeakers, setMaxSpeakers,
}: {
  language: string; setLanguage: (v: string) => void;
  task: string; setTask: (v: string) => void;
  wordTimestamps: boolean; setWordTimestamps: (v: boolean) => void;
  diarize: boolean; setDiarize: (v: boolean) => void;
  diarizationAvailable: boolean;
  minSpeakers: string; setMinSpeakers: (v: string) => void;
  maxSpeakers: string; setMaxSpeakers: (v: string) => void;
}) {
  return (
    <div className="options-row">
      <label>
        <span>Language</span>
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="">Auto-detect</option>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="it">Italian</option>
          <option value="pt">Portuguese</option>
          <option value="ja">Japanese</option>
          <option value="ko">Korean</option>
          <option value="zh">Chinese</option>
          <option value="ru">Russian</option>
          <option value="ar">Arabic</option>
        </select>
      </label>
      <label>
        <span>Task</span>
        <select value={task} onChange={(e) => setTask(e.target.value)}>
          <option value="transcribe">Transcribe</option>
          <option value="translate">Translate to English</option>
        </select>
      </label>
      <label className="check-label">
        <input type="checkbox" checked={wordTimestamps} onChange={(e) => setWordTimestamps(e.target.checked)} />
        <span>Word timestamps</span>
      </label>
      <label className={`check-label ${!diarizationAvailable ? 'disabled-option' : ''}`} title={!diarizationAvailable ? 'Set STT_HF_TOKEN to enable speaker diarization' : ''}>
        <input
          type="checkbox"
          checked={diarize}
          disabled={!diarizationAvailable}
          onChange={(e) => setDiarize(e.target.checked)}
        />
        <span>Speaker diarization</span>
      </label>
      {diarize && (
        <>
          <label>
            <span>Min speakers</span>
            <input
              type="number"
              min={1}
              value={minSpeakers}
              onChange={(e) => setMinSpeakers(e.target.value)}
              placeholder="auto"
              style={{ width: '5rem' }}
            />
          </label>
          <label>
            <span>Max speakers</span>
            <input
              type="number"
              min={1}
              value={maxSpeakers}
              onChange={(e) => setMaxSpeakers(e.target.value)}
              placeholder="auto"
              style={{ width: '5rem' }}
            />
          </label>
        </>
      )}
    </div>
  );
}

function SpeakerSummaryView({ speakers }: { speakers: SpeakerSummary[] }) {
  if (!speakers.length) return null;
  return (
    <div className="speaker-summary">
      <h3 className="speaker-summary-title">Speaker Summary</h3>
      <table className="speaker-table">
        <thead>
          <tr>
            <th>Speaker</th>
            <th>Segments</th>
            <th>Total Duration</th>
          </tr>
        </thead>
        <tbody>
          {speakers.map((s) => (
            <tr key={s.id}>
              <td>
                <span
                  className="seg-speaker"
                  style={{ '--speaker-hue': speakerHue(s.id) } as React.CSSProperties}
                >
                  {s.id}
                </span>
              </td>
              <td>{s.segment_count}</td>
              <td>{formatTime(s.total_duration)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultView({ result }: { result: TranscribeResult }) {
  const [view, setView] = useState<'text' | 'segments' | 'json'>('text');

  return (
    <div className="result-panel">
      <div className="result-meta">
        <span>Language: <strong>{result.language}</strong> ({(result.language_probability * 100).toFixed(0)}%)</span>
        <span>Duration: <strong>{formatTime(result.duration)}</strong></span>
        <span>Processed in: <strong>{result.processing_time.toFixed(1)}s</strong></span>
        <span>Speed: <strong>{(result.duration / result.processing_time).toFixed(1)}x</strong> realtime</span>
      </div>
      <div className="result-tabs">
        <button className={view === 'text' ? 'active' : ''} onClick={() => setView('text')}>Full Text</button>
        <button className={view === 'segments' ? 'active' : ''} onClick={() => setView('segments')}>Segments</button>
        <button className={view === 'json' ? 'active' : ''} onClick={() => setView('json')}>JSON</button>
      </div>
      {view === 'text' && (
        <div className="result-text">
          <p>{result.text}</p>
          <button className="copy-btn" onClick={() => navigator.clipboard.writeText(result.text)}>Copy</button>
        </div>
      )}
      {view === 'segments' && (
        <div className="result-segments">
          {result.segments.map((seg) => (
            <div key={seg.id} className="segment-row">
              <span className="seg-time">{formatTime(seg.start)} → {formatTime(seg.end)}</span>
              {seg.speaker && (
                <span
                  className="seg-speaker"
                  style={{ '--speaker-hue': speakerHue(seg.speaker) } as React.CSSProperties}
                >
                  {seg.speaker}
                </span>
              )}
              <span className="seg-text">{seg.text}</span>
            </div>
          ))}
        </div>
      )}
      {view === 'json' && (
        <pre className="result-json">{JSON.stringify(result, null, 2)}</pre>
      )}
      {result.speakers && result.speakers.length > 0 && (
        <SpeakerSummaryView speakers={result.speakers} />
      )}
    </div>
  );
}

export function App() {
  const [info, setInfo] = useState<ServiceInfo | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState('');
  const [task, setTask] = useState('transcribe');
  const [wordTimestamps, setWordTimestamps] = useState(false);
  const [diarize, setDiarize] = useState(false);
  const [minSpeakers, setMinSpeakers] = useState('');
  const [maxSpeakers, setMaxSpeakers] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TranscribeResult | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchInfo().then(setInfo).catch(() => {});
  }, []);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError('');
  }, []);

  const handleTranscribe = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError('');
    setResult(null);
    setStatus('Uploading...');
    const parsedMin = minSpeakers ? parseInt(minSpeakers, 10) : undefined;
    const parsedMax = maxSpeakers ? parseInt(maxSpeakers, 10) : undefined;
    if (parsedMin !== undefined && parsedMax !== undefined && parsedMin > parsedMax) {
      setError('Min speakers must be less than or equal to max speakers');
      setBusy(false);
      return;
    }
    try {
      const res = await transcribe(file, {
        language: language || undefined,
        task,
        word_timestamps: wordTimestamps,
        diarize,
        min_speakers: parsedMin,
        max_speakers: parsedMax,
      }, setStatus);
      setResult(res);
      setStatus('');
    } catch (e: any) {
      setError(e.message || 'Transcription failed');
      setStatus('');
    } finally {
      setBusy(false);
    }
  }, [file, language, task, wordTimestamps, diarize, minSpeakers, maxSpeakers]);

  const diarizationAvailable = info?.diarization?.available ?? false;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Speech to Text</h1>
        {info && (
          <div className="service-info">
            <span className={`status-dot ${info.model_loaded ? 'ready' : 'idle'}`} />
            <span>{info.model_size} on {info.device}</span>
            {info.gpu && <span>{info.gpu.name}</span>}
          </div>
        )}
      </header>

      <main className="app-main">
        <DropZone onFile={handleFile} disabled={busy} />

        {file && (
          <div className="file-info">
            <span className="file-name">{file.name}</span>
            <span className="file-size">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
            <button className="clear-btn" onClick={() => { setFile(null); setResult(null); }}>✕</button>
          </div>
        )}

        <Options
          language={language} setLanguage={setLanguage}
          task={task} setTask={setTask}
          wordTimestamps={wordTimestamps} setWordTimestamps={setWordTimestamps}
          diarize={diarize} setDiarize={setDiarize}
          diarizationAvailable={diarizationAvailable}
          minSpeakers={minSpeakers} setMinSpeakers={setMinSpeakers}
          maxSpeakers={maxSpeakers} setMaxSpeakers={setMaxSpeakers}
        />

        <div className="action-row">
          <button
            className="transcribe-btn"
            disabled={!file || busy}
            onClick={handleTranscribe}
          >
            {busy ? status || 'Processing...' : 'Transcribe'}
          </button>
        </div>

        {error && <div className="error-banner">{error}</div>}
        {result && <ResultView result={result} />}
      </main>
    </div>
  );
}
