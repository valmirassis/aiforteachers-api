import os, base64, mimetypes, json, re
from typing import Optional, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Defina GOOGLE_API_KEY no .env")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,

    # outros parâmetros diretos: temperature=..., max_output_tokens=...
)

def _guess_mime_type(filename: str, fallback: str = "audio/wav") -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("audio/"):
        return mime
    if mime in ("audio/mp4", "video/mp4"):
        return "audio/mp4"
    return fallback

def _parse_json_loose(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        candidate = m.group(0)
        for variant in (candidate, candidate.replace("'", '"')):
            try:
                return json.loads(variant)
            except Exception:
                continue
    return None

# ---- Split seguro em frases (pt-BR), caso o modelo não entregue segments ----
_ABBR = [
    "sr.", "sra.", "srta.", "dr.", "dra.", "prof.", "profa.", "eng.", "etc.", "e.g.", "i.e.",
    "gov.", "dep.", "sen.", "ex.", "art.", "fig.", "pág.", "cap."
]
def _split_sentences_pt(text: str) -> List[str]:
    if not text:
        return []
    # protege abreviações substituindo o ponto por marcador
    protected = text
    placeholders = {}
    for i, ab in enumerate(_ABBR):
        token = f"§§ABBR{i}§§"
        placeholders[token] = ab
        protected = re.sub(re.escape(ab), ab[:-1] + token, protected, flags=re.I)

    # split em fim de frase: ponto, interrogação, exclamação seguidos de espaço e letra maiúscula/aspas
    parts = re.split(r'(?<=[.!?])\s+(?=["“”\'(\[]?[A-ZÁÉÍÓÚÂÊÔÃÕÀ0-9])', protected)

    # restaura abreviações e limpa
    sentences = []
    for p in parts:
        for token, ab in placeholders.items():
            p = p.replace(token, ".")
        s = p.strip()
        if s:
            sentences.append(s)
    return sentences

def transcrever_audio_gemini(file_path: str, idioma: str):
    """
    Retorna dict:
    {
      "language": "pt-BR",
      "text": "transcrição completa",
      "segments": ["Frase 1.", "Frase 2.", ...]
    }
    """
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    if not audio_bytes or len(audio_bytes) < 10:
        raise ValueError("Arquivo de áudio vazio ou inválido.")

    mime_type = _guess_mime_type(file_path)
    b64 = base64.b64encode(audio_bytes).decode("utf-8")

    idioma_label = "português do Brasil" if idioma.lower() in ("pt", "pt-br", "ptbr") else idioma

    prompt = (
        f"Transcreva integralmente o áudio em {idioma_label}. "
        "Responda EXATAMENTE em JSON, sem texto extra, no formato:\n"
        "{\n"
        '  \"language\": \"pt-BR\",\n'
        '  \"text\": \"transcrição completa\",\n'
        '  \"segments\": [\"Frase 1.\", \"Frase 2.\", \"Frase 3.\"]\n'
        "}\n"
        "Divida os elementos de \"segments\" em cada fim de frase (ponto final, interrogação ou exclamação). "
        "Não inclua timestamps. Não inclua comentários fora do JSON."
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
        # fallback: cria JSON a partir do texto cru
        text_only = raw.strip()
        data = {
            "language": "pt-BR" if idioma.lower().startswith("pt") else idioma_label,
            "text": text_only,
            "segments": _split_sentences_pt(text_only),
        }
    else:
        # sane defaults e fallback de segments
        lang = data.get("language") or ("pt-BR" if idioma.lower().startswith("pt") else idioma_label)
        text = (data.get("text") or "").strip()
        segs = data.get("segments")
        if not isinstance(segs, list) or not segs:
            segs = _split_sentences_pt(text)
        else:
            # garante strings limpas
            segs = [str(s).strip() for s in segs if str(s).strip()]
        data = {"language": lang, "text": text, "segments": segs}

    return data