import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchInfo, transcribe, type ServiceInfo, type TranscribeResult } from './api';

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
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
}: {
  language: string; setLanguage: (v: string) => void;
  task: string; setTask: (v: string) => void;
  wordTimestamps: boolean; setWordTimestamps: (v: boolean) => void;
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
              <span className="seg-text">{seg.text}</span>
            </div>
          ))}
        </div>
      )}
      {view === 'json' && (
        <pre className="result-json">{JSON.stringify(result, null, 2)}</pre>
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
    try {
      const res = await transcribe(file, {
        language: language || undefined,
        task,
        word_timestamps: wordTimestamps,
      }, setStatus);
      setResult(res);
      setStatus('');
    } catch (e: any) {
      setError(e.message || 'Transcription failed');
      setStatus('');
    } finally {
      setBusy(false);
    }
  }, [file, language, task, wordTimestamps]);

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
