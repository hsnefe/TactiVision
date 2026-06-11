"""AI spiker için sistem prompt'u ve event -> metin dönüştürme yardımcıları."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "Sen efsanevi bir Türk futbol spikerisin. Tarzın: son derece HEYECANLI, coşkulu, "
    "ateşli ve zaman zaman esprili. Tıpkı canlı yayında maç anlatan bir spiker gibi "
    "konuşursun; tempoyu yükseltir, kritik anlarda bağırırcasına vurgu yaparsın "
    "(GOOOL değil ama 'işte o pas!', 'müthiş bir kombinasyon!' gibi). "
    "Sana bir maçtaki pas olaylarının zaman sıralı listesi verilecek. Bunları akıcı, "
    "kesintisiz, tek parça bir CANLI ANLATIM metnine dönüştür.\n\n"
    "Kurallar:\n"
    "- Oyuncuları 'numara X' veya '#X' diye an (ör. '7 numara'). Takımları 'A takımı' ve "
    "'B takımı' olarak adlandır (team_id 0 = A, 1 = B).\n"
    "- completed_pass = başarılı pas, intercepted_pass = rakibin araya girip topu kaptığı "
    "kapılan pas (bunlara coşkuyla tepki ver!), unknown_pass = topun el değiştirdiği belirsiz "
    "an.\n"
    "- Olayları sırayla, akıcı cümlelerle bağla. Robotik 'sonra... sonra...' deme.\n"
    "- Sadece anlatım metnini döndür. Madde işareti, başlık, açıklama YOK. "
    "Düz, seslendirilmeye hazır paragraf(lar) yaz.\n"
    "- Tamamen Türkçe yaz."
)


def _team_name(team_id: Any) -> str:
    if team_id == 0:
        return "A takımı"
    if team_id == 1:
        return "B takımı"
    return "bir takım"


def events_to_lines(events: list[dict[str, Any]]) -> str:
    """Event listesini LLM'e verilecek kompakt, okunabilir bir özete çevir."""

    lines: list[str] = []
    for ev in events:
        t = ev.get("start_time_sec")
        ts = f"{t:0.1f}sn" if isinstance(t, (int, float)) else "?"
        etype = ev.get("event_type", "pass")
        frm = ev.get("from_player_id")
        to = ev.get("to_player_id")
        frm_team = _team_name(ev.get("from_team_id"))
        to_team = _team_name(ev.get("to_team_id"))
        conf = ev.get("confidence")
        conf_s = f", güven {conf:.0%}" if isinstance(conf, (int, float)) else ""

        if etype == "completed_pass":
            desc = f"{frm_team} {frm} numara -> {to} numaraya başarılı pas"
        elif etype == "intercepted_pass":
            desc = f"{frm_team} {frm} numaranın pasını {to_team} {to} numara KAPTI (kapılan pas)"
        else:
            desc = f"{frm_team} {frm} numarada belirsiz top hareketi"
        lines.append(f"[{ts}] {desc}{conf_s}")
    return "\n".join(lines)


def build_user_prompt(events: list[dict[str, Any]], target_seconds: float | None = None) -> str:
    summary = events_to_lines(events)

    budget = ""
    if target_seconds and target_seconds > 0:
        # Türkçe sesli anlatım ~2.3 kelime/sn. Sığması için biraz altında bütçele.
        max_words = max(12, int(target_seconds * 2.2))
        budget = (
            f"\n\nÇOK ÖNEMLİ — SÜRE SINIRI: Video sadece ~{int(round(target_seconds))} saniye. "
            f"Anlatım yüksek sesle okunduğunda bu süreye SIĞMALI. Bu yüzden EN FAZLA "
            f"{max_words} kelime kullan. Kısa, vurucu ve öz ol; gereksiz tekrar ve uzatma yapma. "
            f"Az olayı bile coşkuyla anlat ama süreyi AŞMA."
        )

    return (
        "Aşağıda bir futbol maçındaki pas olayları zaman sıralı olarak veriliyor. "
        "Bunları tek parça, coşkulu bir canlı maç anlatımına dönüştür:\n\n"
        f"{summary}"
        f"{budget}\n\n"
        "Şimdi spiker anlatımını yaz:"
    )
