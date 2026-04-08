import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchInfo, transcribe, type ServiceInfo, type TranscribeResult, type SpeakerSummary } from './api';

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${m}:${s.toString().padStart(2, '0')}.${ms}`;
}

function formatSeconds(value?: number | null): string {
  return value == null ? 'n/a' : `${value.toFixed(1)}s`;
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
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) onFile(droppedFile);
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
        accept="audio/*,video/*,.wav,.mp3,.flac,.ogg,.m4a,.webm,.mp4,.aac,.wma,.aif,.aiff"
        style={{ display: 'none' }}
        onChange={(e) => { if (e.target.files?.[0]) onFile(e.target.files[0]); }}
      />
      <div className="drop-icon">🎙</div>
      <div className="drop-label">Drop an audio file here or click to browse</div>
      <div className="drop-hint">WAV, MP3, FLAC, OGG, M4A, WebM, MP4, AIF, AIFF</div>
    </div>
  );
}

function CapabilityCard({
  title,
  value,
  detail,
  tone = 'neutral',
}: {
  title: string;
  value: string;
  detail?: string;
  tone?: 'neutral' | 'ready' | 'warn';
}) {
  return (
    <div className={`capability-card capability-card-${tone}`}>
      <span className="capability-title">{title}</span>
      <strong className="capability-value">{value}</strong>
      {detail && <span className="capability-detail">{detail}</span>}
    </div>
  );
}

function ServiceCapabilities({
  info,
  refreshing,
  onRefresh,
}: {
  info: ServiceInfo;
  refreshing: boolean;
  onRefresh: () => void;
}) {
  return (
    <section className="capability-panel">
      <div className="capability-header">
        <div>
          <h2>Service Capabilities</h2>
          <p>Live readiness and GPU usage for this STT + diarization node.</p>
        </div>
        <button type="button" className="refresh-btn" onClick={onRefresh} disabled={refreshing}>
          {refreshing ? 'Refreshing…' : 'Refresh status'}
        </button>
      </div>
      <div className="capability-grid">
        <CapabilityCard
          title="Engine"
          value={`${info.engine} · ${info.model_size}`}
          detail={`${info.device} / ${info.compute_type}`}
          tone={info.model_loaded ? 'ready' : 'warn'}
        />
        <CapabilityCard
          title="Whisper Model"
          value={info.model_loaded ? 'Ready' : 'Loading'}
          detail={info.model_loaded ? 'Transcription requests can run now.' : 'Background warmup is still in progress.'}
          tone={info.model_loaded ? 'ready' : 'warn'}
        />
        <CapabilityCard
          title="Diarization"
          value={!info.diarization.available ? 'Disabled' : info.diarization.ready ? 'Ready' : 'Lazy / warming'}
          detail={!info.diarization.available
            ? 'Set STT_HF_TOKEN to enable pyannote speaker attribution.'
            : `${info.diarization.model}${info.diarization.whisper_model_override ? ` · whisper override ${info.diarization.whisper_model_override}` : ''}`}
          tone={info.diarization.available && info.diarization.ready ? 'ready' : info.diarization.available ? 'warn' : 'neutral'}
        />
        <CapabilityCard
          title="GPU / Upload"
          value={info.gpu ? `${info.gpu.name}` : 'CPU / unknown GPU'}
          detail={`${info.gpu ? `${info.gpu.vram_used_mb} / ${info.gpu.vram_total_mb} MB used` : 'No GPU details'} · max upload ${info.max_upload_mb} MB`}
        />
      </div>
      {info.diarization.available && (
        <p className="capability-note">
          Pyannote idle timeout: <strong>{info.diarization.idle_timeout_sec ?? 'n/a'}s</strong>.
          {!info.diarization.ready && ' The first diarized request may take longer while the pipeline loads.'}
        </p>
      )}
    </section>
  );
}

function Options({
  language, setLanguage,
  task, setTask,
  wordTimestamps, setWordTimestamps,
  diarize, setDiarize,
  diarizationAvailable,
  diarizationReady,
  minSpeakers, setMinSpeakers,
  maxSpeakers, setMaxSpeakers,
  initialPrompt, setInitialPrompt,
}: {
  language: string; setLanguage: (v: string) => void;
  task: string; setTask: (v: string) => void;
  wordTimestamps: boolean; setWordTimestamps: (v: boolean) => void;
  diarize: boolean; setDiarize: (v: boolean) => void;
  diarizationAvailable: boolean;
  diarizationReady: boolean;
  minSpeakers: string; setMinSpeakers: (v: string) => void;
  maxSpeakers: string; setMaxSpeakers: (v: string) => void;
  initialPrompt: string; setInitialPrompt: (v: string) => void;
}) {
  return (
    <div className="options-stack">
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
              />
            </label>
          </>
        )}
      </div>
      <label className="prompt-field">
        <span>Initial Prompt</span>
        <textarea
          rows={3}
          value={initialPrompt}
          onChange={(e) => setInitialPrompt(e.target.value)}
          placeholder="Optional domain hint, glossary, or speaker context for Whisper."
        />
      </label>
      <p className="option-note">
        {!diarizationAvailable
          ? 'Speaker diarization is disabled until STT_HF_TOKEN is configured.'
          : diarizationReady
            ? 'Pyannote is ready for diarized requests.'
            : 'Pyannote will load on demand for the next diarized request.'}
      </p>
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
        <span>Whisper: <strong>{formatSeconds(result.whisper_time)}</strong></span>
        <span>Diarization: <strong>{formatSeconds(result.diarization_time)}</strong></span>
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
  const [initialPrompt, setInitialPrompt] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [infoBusy, setInfoBusy] = useState(false);
  const [result, setResult] = useState<TranscribeResult | null>(null);
  const [error, setError] = useState('');

  const loadInfo = useCallback(async () => {
    setInfoBusy(true);
    try {
      setInfo(await fetchInfo());
    } finally {
      setInfoBusy(false);
    }
  }, []);

  useEffect(() => {
    loadInfo().catch(() => {});
  }, [loadInfo]);

  const handleFile = useCallback((nextFile: File) => {
    setFile(nextFile);
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
        initial_prompt: initialPrompt || undefined,
        diarize,
        min_speakers: parsedMin,
        max_speakers: parsedMax,
      }, setStatus);
      setResult(res);
      setStatus('');
      await loadInfo();
    } catch (e: any) {
      setError(e.message || 'Transcription failed');
      setStatus('');
      await loadInfo();
    } finally {
      setBusy(false);
    }
  }, [file, language, task, wordTimestamps, initialPrompt, diarize, minSpeakers, maxSpeakers, loadInfo]);

  const diarizationAvailable = info?.diarization?.available ?? false;
  const diarizationReady = info?.diarization?.ready ?? false;

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Speech to Text</h1>
          <p className="header-copy">Test transcription, translation, and diarization directly against the live GPU node.</p>
        </div>
        {info && (
          <div className="service-info">
            <span className={`status-dot ${info.model_loaded ? 'ready' : 'idle'}`} />
            <span>{info.engine} · {info.model_size}</span>
            {info.gpu && <span>{info.gpu.name}</span>}
          </div>
        )}
      </header>

      <main className="app-main">
        {info && (
          <ServiceCapabilities info={info} refreshing={infoBusy} onRefresh={() => { void loadInfo(); }} />
        )}

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
          diarizationReady={diarizationReady}
          minSpeakers={minSpeakers} setMinSpeakers={setMinSpeakers}
          maxSpeakers={maxSpeakers} setMaxSpeakers={setMaxSpeakers}
          initialPrompt={initialPrompt} setInitialPrompt={setInitialPrompt}
        />

        <div className="action-row">
          <button
            className="transcribe-btn"
            disabled={!file || busy}
            onClick={handleTranscribe}
          >
            {busy ? status || 'Processing…' : diarize ? 'Transcribe + Diarize' : 'Transcribe'}
          </button>
          <p className="action-note">
            {diarize
              ? 'Diarized runs may take longer on the first request while pyannote loads.'
              : 'Use diarization when you need speaker labels and per-speaker summaries.'}
          </p>
        </div>

        {error && <div className="error-banner">{error}</div>}
        {result && <ResultView result={result} />}
      </main>
    </div>
  );
}
