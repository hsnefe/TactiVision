import React, { useRef, useState } from "react";

export default function UploadPanel({ onUpload, disabled }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [name, setName] = useState("");

  function pick(file) {
    if (!file) return;
    setName(file.name);
    onUpload(file);
  }

  return (
    <div
      className={`upload ${dragOver ? "upload--over" : ""} ${disabled ? "upload--disabled" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (!disabled) pick(e.dataTransfer.files?.[0]);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        hidden
        onChange={(e) => pick(e.target.files?.[0])}
      />
      <div className="upload__icon">⚽</div>
      <div className="upload__title">Maç videosunu buraya sürükle</div>
      <div className="upload__hint">{name || "ya da tıklayıp seç (mp4, mov, mkv, webm)"}</div>
    </div>
  );
}
