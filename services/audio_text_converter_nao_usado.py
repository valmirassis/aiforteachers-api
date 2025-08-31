import os, base64, mimetypes, json, re, subprocess, shutil, math, wave, contextlib
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Optional
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Defina GOOGLE_API_KEY no .env")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # pode testar "gemini-1.5-flash" também
    google_api_key=GOOGLE_API_KEY,
    generation_config={"response_mime_type": "text/plain"},
)

def _guess_mime_type(filename: str, fallback: str = "audio/wav") -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("audio/"):
        return mime
    if mime in ("audio/mp4", "video/mp4"):  # m4a geralmente vira audio/mp4
        return "audio/mp4"
    return fallback

def _get_duration_seconds(path: str) -> float:
    """
    Tenta obter a duração real do áudio, nessa ordem:
    1) mutagen (mp3/m4a/ogg/etc.)
    2) wave (apenas .wav)
    3) ffprobe (se disponível no SO)
    """
    # 1) mutagen
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(path)
        if mf is not None and getattr(mf, "info", None) and hasattr(mf.info, "length"):
            return float(mf.info.length)
    except Exception:
        pass

    # 2) wave para .wav
    if path.lower().endswith(".wav"):
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)

    # 3) ffprobe (se existir)
    if shutil.which("ffprobe"):
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path
            ], stderr=subprocess.STDOUT, timeout=10)
            val = float(out.decode().strip())
            if val > 0:
                return val
        except Exception:
            pass

    raise ValueError("Não foi possível determinar a duração do áudio.")

def _to_seconds(ts: str) -> Optional[float]:
    """Aceita 'MM:SS', 'HH:MM:SS' ou número (str/float)."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    # número puro
    try:
        return float(s)
    except ValueError:
        pass
    # HH:MM:SS ou MM:SS
    parts = s.split(":")
    if not all(p.isdigit() for p in parts):
        return None
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    return None

def _to_mmss(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = int(round(seconds - m * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m:02d}:{s:02d}"

def _parse_json_loose(s: str):
    """Tenta parsear JSON; se vier texto ao redor, extrai o 1º bloco { ... }."""
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        candidate = m.group(0)
        for attempt in (candidate, candidate.replace("'", '"')):
            try:
                return json.loads(attempt)
            except Exception:
                continue
    return None

def _normalize_segments(segments: list, duration: float) -> list[dict]:
    """
    Normaliza segmentos para sempre terem start/end dentro do range do áudio.
    - start monotônico
    - end = próximo start (ou duração total se for o último)
    """
    parsed = []
    for seg in segments or []:
        if isinstance(seg, dict) and "text" in seg:
            start_sec = _to_seconds(seg.get("start"))
            if start_sec is None:
                start_sec = 0.0
            parsed.append((max(0.0, float(start_sec)), str(seg["text"]).strip()))
        elif isinstance(seg, str) and seg.strip():
            parsed.append((0.0, seg.strip()))

    if not parsed:
        return [{"start": "00:00", "end": _to_mmss(duration), "text": ""}]

    # ordena por start
    parsed.sort(key=lambda x: x[0])

    # força monotonicidade
    normalized = []
    prev = 0.0
    for s, t in parsed:
        s = max(s, prev)
        normalized.append((s, t))
        prev = s + 0.01

    # escala se último start > duração
    last_start = normalized[-1][0]
    if duration > 0 and last_start > duration:
        factor = max((duration - 0.1), 0.0) / last_start
        normalized = [(s * factor, t) for s, t in normalized]

    # agora calcula end = próximo start, ou duração total
    result = []
    for i, (s, t) in enumerate(normalized):
        if i < len(normalized) - 1:
            e = normalized[i + 1][0]
        else:
            e = duration
        result.append({
            "start": _to_mmss(min(max(0.0, s), duration)),
            "end": _to_mmss(min(max(0.0, e), duration)),
            "text": t
        })

    return result

def transcrever_audio_gemini(file_path: str, idioma: str):
    """
    Retorna:
    {
      "language": "...",
      "text": "...",
      "segments": [{"start":"MM:SS","text":"..."}],
      "audio_duration_seconds": 90.3,
      "audio_duration_mmss": "01:30"
    }
    """
    # lê bytes
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    if not audio_bytes or len(audio_bytes) < 10:
        raise ValueError("Arquivo de áudio vazio ou inválido.")

    # duração real do arquivo
    duration = _get_duration_seconds(file_path)
    duration_mmss = _to_mmss(duration)

    mime_type = _guess_mime_type(file_path)
    b64 = base64.b64encode(audio_bytes).decode("utf-8")

    idioma_label = "português do Brasil" if idioma.lower() in ("pt", "pt-br", "ptbr") else idioma
    prompt = (
        f"Transcreva integralmente o áudio em {idioma_label}. "
        f"A duração real do áudio é {duration_mmss}. "
        "Responda EXATAMENTE em JSON, sem texto extra, no formato:\n"
        "{\n"
        '  "language": "pt-BR",\n'
        '  "text": "transcrição completa",\n'
        '  "segments": [\n'
        '     {"start": "MM:SS", "text": "trecho 1"},\n'
        '     {"start": "MM:SS", "text": "trecho 2"}\n'
        "  ]\n"
        "}\n"
        f"Garante que todos os 'start' estejam entre 00:00 e {duration_mmss}, em ordem não-decrescente. "
        "Use timestamps aproximados (ex.: ~10–15s por segmento). Sem comentários fora do JSON."
    )

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "media", "mime_type": mime_type, "data": b64},
        ]
    )

    resp = model.invoke([msg])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)

    data = _parse_json_loose(raw)
    if not isinstance(data, dict):
        data = {"language": idioma_label, "text": raw.strip(), "segments": []}

    data.setdefault("language", idioma_label)
    data.setdefault("text", "")
    data.setdefault("segments", [])
    data["segments"] = _normalize_segments(data["segments"], duration)
    data["audio_duration_seconds"] = round(float(duration), 3)
    data["audio_duration_mmss"] = duration_mmss
    return data