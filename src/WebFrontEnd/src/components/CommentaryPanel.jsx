import React from "react";

export default function CommentaryPanel({ commentary }) {
  if (!commentary) return null;
  return (
    <div className="commentary">
      <div className="commentary__head">
        <h3 className="panel__h">🎙️ AI Spiker Anlatımı</h3>
        <span className="provider">{commentary.provider}</span>
      </div>

      {commentary.video_url ? (
        <>
          <video className="commentary__video" controls src={commentary.video_url} />
          <a className="btn-dl" href={commentary.video_url} download>
            ⬇️ Spikerli videoyu indir
          </a>
        </>
      ) : commentary.audio_url ? (
        <audio className="commentary__audio" controls src={commentary.audio_url} autoPlay>
          Tarayıcınız ses oynatmayı desteklemiyor.
        </audio>
      ) : (
        <div className="commentary__noaudio">
          Ses üretilemedi (TTS kapalı/erişilemez) — metin aşağıda.
        </div>
      )}

      <p className="commentary__text">{commentary.text}</p>
    </div>
  );
}
