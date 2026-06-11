import React from "react";

const META = {
  completed_pass: { label: "Başarılı Pas", cls: "ev--ok", icon: "✅" },
  intercepted_pass: { label: "Kapılan Pas", cls: "ev--int", icon: "🔁" },
  unknown_pass: { label: "Belirsiz", cls: "ev--unk", icon: "❔" },
};

function team(id) {
  if (id === 0) return "A";
  if (id === 1) return "B";
  return "?";
}

export default function EventTimeline({ events }) {
  if (!events?.length) return null;
  return (
    <div className="timeline">
      <h3 className="panel__h">Tespit Edilen Pas Olayları ({events.length})</h3>
      <ul className="events">
        {events.map((e) => {
          const m = META[e.event_type] || META.unknown_pass;
          return (
            <li key={e.event_id} className={`ev ${m.cls}`}>
              <span className="ev__time">{(e.start_time_sec ?? 0).toFixed(1)}″</span>
              <span className="ev__icon">{m.icon}</span>
              <span className="ev__body">
                <strong>#{e.from_player_id ?? "?"}</strong>
                <span className="ev__arrow">→</span>
                <strong>{e.to_player_id != null ? `#${e.to_player_id}` : "—"}</strong>
                <span className="ev__teams">[{team(e.from_team_id)}{e.to_team_id != null ? `→${team(e.to_team_id)}` : ""}]</span>
              </span>
              <span className="ev__type">{m.label}</span>
              {e.confidence != null && (
                <span className="ev__conf">{Math.round(e.confidence * 100)}%</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
