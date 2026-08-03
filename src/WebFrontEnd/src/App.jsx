import React, { useEffect, useRef, useState } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import ProgressStepper from "./components/ProgressStepper.jsx";
import EventTimeline from "./components/EventTimeline.jsx";
import CommentaryPanel from "./components/CommentaryPanel.jsx";
import { uploadVideo, getJob, getEvents, getCommentary } from "./services/api.js";

export default function App() {
  const [job, setJob] = useState(null);
  const [events, setEvents] = useState([]);
  const [commentary, setCommentary] = useState(null);
  const [error, setError] = useState("");
  const pollRef = useRef(null);

  const busy = job && !["done", "failed"].includes(job.stage);

  async function handleUpload(file) {
    setError(""); setEvents([]); setCommentary(null);
    try {
      const j = await uploadVideo(file);
      setJob(j);
      startPolling(j.job_id);
    } catch (e) {
      setError(e.message);
    }
  }

  function startPolling(jobId) {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const j = await getJob(jobId);
        setJob(j);
        if (j.stage === "done") {
          clearInterval(pollRef.current);
          setEvents((await getEvents(jobId)).events);
          setCommentary(await getCommentary(jobId));
        } else if (j.stage === "failed") {
          clearInterval(pollRef.current);
          setEvents((await getEvents(jobId)).events);
        }
      } catch (e) {
        clearInterval(pollRef.current);
        setError(e.message);
      }
    }, 800);
  }

  useEffect(() => () => clearInterval(pollRef.current), []);

  function reset() {
    clearInterval(pollRef.current);
    setJob(null); setEvents([]); setCommentary(null); setError("");
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="hero__logo">⚽ TactiVision</div>
        <h1 className="hero__title">Maçını yükle, <span>AI spiker</span> anlatsın</h1>
        <p className="hero__sub">
          Video → oyuncu &amp; pas tespiti → coşkulu spiker anlatımı → gerçek insan sesi
        </p>
      </header>

      {!job && <UploadPanel onUpload={handleUpload} disabled={false} />}

      {error && <div className="alert">⚠️ {error}</div>}

      {job && (
        <section className="panel">
          <div className="panel__top">
            <span className="panel__file">📄 {job.filename}</span>
            <button className="btn-ghost" onClick={reset} disabled={busy}>
              {busy ? "İşleniyor…" : "Yeni video"}
            </button>
          </div>
          <ProgressStepper job={job} />
        </section>
      )}

      {commentary && (
        <section className="panel">
          <CommentaryPanel commentary={commentary} />
        </section>
      )}

      {events.length > 0 && (
        <section className="panel">
          <EventTimeline events={events} />
        </section>
      )}

      <footer className="foot">
        TactiVision · FastAPI + VisionEngine + Gemini + Edge TTS
      </footer>
    </div>
  );
}
