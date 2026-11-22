import { useRef, useState } from "react";

const MAX_BYTES = 25 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = [".wav", ".flac", ".mp3"];
const LABELS = ["anger", "happiness", "neutral", "sadness"];
const MODELS = [
  { id: "svm", name: "MFCC + SVM", detail: "Compact acoustic-feature baseline" },
  { id: "wav2vec2", name: "wav2vec 2.0", detail: "Pretrained speech encoder · slower on CPU" },
];
const LABEL_COLORS = {
  anger: "#d86a55",
  happiness: "#d4a642",
  neutral: "#82938c",
  sadness: "#7886c7",
};

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  const total = Math.round(seconds || 0);
  const minutes = Math.floor(total / 60);
  const remaining = total % 60;
  return minutes ? `${minutes} min ${remaining} sec` : `${remaining} sec`;
}

function errorMessage(response, body) {
  if (response.status === 413) return "This recording is larger than 25 MB. Choose a smaller audio file.";
  if (response.status === 415) return "That file type is not supported. Choose a WAV, FLAC, or MP3 recording.";
  if (response.status === 422) return body?.detail || "The audio could not be analyzed. Try a clear recording with audible speech.";
  if (response.status === 503) return body?.detail || "The emotion model is not ready. Start the backend after setting up the model.";
  return body?.detail || "The analysis could not be completed. Check that the local API is running and try again.";
}

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("svm");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);

  function chooseFile(nextFile) {
    setResult(null);
    setError("");
    if (!nextFile) return;
    const extension = `.${nextFile.name.split(".").pop()}`.toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setFile(null);
      setError("Choose an audio file in WAV, FLAC, or MP3 format.");
      return;
    }
    if (nextFile.size === 0) {
      setFile(null);
      setError("That file is empty. Choose a recording that contains audio.");
      return;
    }
    if (nextFile.size > MAX_BYTES) {
      setFile(null);
      setError("That file is larger than 25 MB. Choose a smaller recording.");
      return;
    }
    setFile(nextFile);
  }

  async function analyze(event) {
    event.preventDefault();
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("model", model);
      const response = await fetch("/api/v1/analyze", { method: "POST", body: form });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(response, body));
      if (!body.scores || !LABELS.every((label) => Number.isFinite(body.scores[label]))) {
        throw new Error("The API returned an incomplete score report. Please try again.");
      }
      setResult(body);
    } catch (cause) {
      setError(cause instanceof TypeError
        ? "Could not reach the local analysis service. Start the backend and try again."
        : cause.message || "The analysis could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  function clearFile() {
    setFile(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  const prediction = result?.prediction;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="SER-CAT home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
          <span>SER<span className="brand-dash">–</span>CAT</span>
        </a>
        <div className="topbar-note"><span className="status-dot" /> Research prototype</div>
      </header>

      <main id="top" className="page-grid">
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow"><span className="eyebrow-line" /> SPEECH EMOTION ANALYSIS</p>
          <h1 id="page-title">Listen for the<br /><em>feeling</em> in a voice.</h1>
          <p className="intro-copy">Explore how vocal patterns in a recording map to four emotion categories. Add a clip to see the model’s estimate.</p>

          <div className="label-strip" aria-label="Supported emotion categories">
            {LABELS.map((label) => (
              <span className="label-chip" key={label}>
                <span className="label-dot" style={{ "--label-color": LABEL_COLORS[label] }} />
                {label}
              </span>
            ))}
          </div>

          <div className="method-card">
            <div className="method-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24"><path d="M4 14v-4m4 8V6m4 12V4m4 14V8m4 6v-4" /></svg>
            </div>
            <div>
              <p className="method-title">Compare two speech models</p>
              <p className="method-copy">Choose a compact MFCC + SVM baseline or a fine-tuned wav2vec 2.0 speech encoder.</p>
            </div>
          </div>

          <p className="privacy-note"><span aria-hidden="true">↗</span> Your clip is sent to the local API for analysis. No analysis history is saved.</p>
        </section>

        <section className="analysis-panel" aria-label="Upload and analysis">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">YOUR RECORDING</p>
              <h2>Start an analysis</h2>
            </div>
            <span className="step-badge">01 <span>/ 01</span></span>
          </div>

          <form onSubmit={analyze}>
            <fieldset className="model-selection" disabled={busy}>
              <legend>Choose a model</legend>
              <div className="model-options">
                {MODELS.map((option) => (
                  <label className={`model-option${model === option.id ? " is-selected" : ""}`} key={option.id}>
                    <input
                      type="radio"
                      name="model"
                      value={option.id}
                      checked={model === option.id}
                      onChange={() => {
                        setModel(option.id);
                        setResult(null);
                        setError("");
                      }}
                    />
                    <span className="model-option-copy">
                      <span className="model-option-name">{option.name}</span>
                      <span className="model-option-detail">{option.detail}</span>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept=".wav,.flac,.mp3,audio/wav,audio/flac,audio/mpeg"
              onChange={(event) => chooseFile(event.target.files?.[0])}
              tabIndex={-1}
            />
            {!file ? (
              <button
                className={`dropzone${dragging ? " is-dragging" : ""}`}
                type="button"
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  chooseFile(event.dataTransfer.files?.[0]);
                }}
                aria-describedby="upload-help"
              >
                <span className="upload-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24"><path d="M12 15V4m0 0L8 8m4-4 4 4M5 14v5h14v-5" /></svg>
                </span>
                <span className="drop-title">Drop your audio here</span>
                <span className="drop-subtitle">or <span className="browse-link">browse files</span></span>
                <span className="format-tags"><span>WAV</span><span>FLAC</span><span>MP3</span></span>
              </button>
            ) : (
              <div className="file-card">
                <div className="file-type" aria-hidden="true">AUD</div>
                <div className="file-info">
                  <p className="file-name" title={file.name}>{file.name}</p>
                  <p className="file-size">{formatBytes(file.size)} <span>·</span> ready to analyze</p>
                </div>
                <button className="remove-file" type="button" onClick={clearFile} disabled={busy} aria-label="Remove selected file">×</button>
              </div>
            )}
            <p id="upload-help" className="upload-guidance">WAV, FLAC, or MP3 <span>·</span> up to 25 MB <span>·</span> first 5 minutes analyzed</p>

            {error && <div className="message message-error" role="alert"><span className="message-symbol">!</span><span>{error}</span></div>}

            <button className="analyze-button" type="submit" disabled={!file || busy}>
              {busy ? <><span className="spinner" /> Analyzing recording…</> : <>Analyze recording <span aria-hidden="true">↗</span></>}
            </button>
            <p className="button-caption">wav2vec 2.0 may take longer, especially on CPU</p>
          </form>

          {busy && <div className="loading-note" role="status" aria-live="polite"><span className="loading-pulse" /> Analyzing with {MODELS.find((option) => option.id === model)?.name}</div>}

          {result && (
            <section className="result-card" aria-labelledby="result-title" aria-live="polite">
              <div className="result-heading">
                <div>
                  <p className="panel-kicker">ANALYSIS COMPLETE</p>
                  <h3 id="result-title">Estimated emotion</h3>
                </div>
                <div className="result-heading-side">
                  <span className="model-result-tag">{result.model_name}</span>
                  <span className="result-check" aria-hidden="true">✓</span>
                </div>
              </div>
              <div className="prediction-line">
                <span className="prediction-dot" style={{ "--label-color": LABEL_COLORS[prediction] || "#7886c7" }} />
                <span className="prediction-label">{prediction || "Unknown"}</span>
                <span className="duration-label">{formatDuration(result.duration_seconds)} analyzed</span>
              </div>
              {result.truncated && <div className="message message-truncated"><span className="message-symbol">i</span><span>This recording is longer than 5 minutes. Only the first 5 minutes were analyzed.</span></div>}
              <div className="scores-heading"><span>Relative model scores</span><span>Not probabilities</span></div>
              <div className="score-list">
                {LABELS.map((label) => {
                  const value = result.scores[label];
                  return (
                    <div className="score-row" key={label}>
                      <span className="score-label">{label}</span>
                      <div className="score-track" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(100, value * 100))}%`, "--label-color": LABEL_COLORS[label] }} /></div>
                      <span className="score-value">{(value * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
              <p className="result-footnote">Scores show relative model output. They are not calibrated confidence probabilities.</p>
            </section>
          )}
        </section>

        <footer className="disclaimer">
          <span className="disclaimer-mark" aria-hidden="true">i</span>
          <p><strong>A note on interpretation</strong> This model estimates vocal patterns associated with labels actors were instructed to portray. It cannot know a person’s internal feelings and is not a diagnosis.</p>
        </footer>
      </main>
      <div className="page-decoration decoration-one" aria-hidden="true" />
      <div className="page-decoration decoration-two" aria-hidden="true" />
    </div>
  );
}

export default App;
