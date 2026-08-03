import React from "react";

const STEPS = [
  { key: "processing", label: "Pas Tespiti", icon: "🎯" },
  { key: "commentary", label: "AI Spiker", icon: "🎙️" },
  { key: "synthesizing", label: "Seslendirme", icon: "🔊" },
  { key: "done", label: "Hazır", icon: "✅" },
];

const ORDER = ["queued", "processing", "commentary", "synthesizing", "done"];

export default function ProgressStepper({ job }) {
  if (!job) return null;
  const current = ORDER.indexOf(job.stage);
  const failed = job.stage === "failed";

  return (
    <div className="stepper">
      <div className="stepper__bar">
        <div className="stepper__fill" style={{ width: `${job.progress}%` }} />
      </div>
      <div className="stepper__steps">
        {STEPS.map((s) => {
          const idx = ORDER.indexOf(s.key);
          const state = failed ? "fail" : idx < current ? "done" : idx === current ? "active" : "todo";
          return (
            <div key={s.key} className={`step step--${state}`}>
              <span className="step__icon">{s.icon}</span>
              <span className="step__label">{s.label}</span>
            </div>
          );
        })}
      </div>
      <div className={`stepper__msg ${failed ? "stepper__msg--fail" : ""}`}>
        {failed ? `Hata: ${job.error || job.message}` : job.message}
        {job.mock && !failed && <span className="badge">mock veri</span>}
      </div>
    </div>
  );
}
