from io import BufferedWriter
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from services.question_generator import gerar_questoes_tema, gerar_questoes_pdf
from services.activity_generator import gerar_atividade_tema, gerar_atividade_pdf
from services.script_generator import gerar_roteiro_tema, gerar_roteiro_pdf
from services.audio_text_converter import transcrever_audio_gemini
from services.text_audio_converter import synthesize_speech
from services.image_describe import describe_image
import os
import uuid
import shutil



app = FastAPI()

API_TOKEN = os.getenv("CODIGO_ACESSO")
STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
os.makedirs(f"{STORAGE_DIR}/in", exist_ok=True)
os.makedirs(f"{STORAGE_DIR}/out", exist_ok=True)

# ------- GERADOR DE QUESTÕES MÚLTIPLA ESCOLHA -----------------------
@app.post("/gerar-questoes-tema")
def questoes_tema(
    token: str = Form(...),
    tema: str = Form(...),
    tipo: str = Form(...),
    qtd: int = Form(...),
    dificuldade: str = Form(...)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    questoes = gerar_questoes_tema(tema, tipo, qtd, dificuldade)
    return {"tema": tema, "questoes": questoes}

@app.post("/gerar-questoes-pdf")
def questoes_pdf(
    token: str = Form(...),
    tipo: str = Form(...),
    qtd: int = Form(...),
    dificuldade: str = Form(...),
    consulta: str = Form(""),
    arquivo: UploadFile = File(...)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    conteudo = arquivo.file.read()
    questoes = gerar_questoes_pdf(conteudo, tipo, qtd, consulta, dificuldade)
    return {"arquivo": arquivo.filename, "questoes": questoes}


# ------- GERADOR DE ATIVIDADES  -----------------------
@app.post("/gerar-atividade-tema")
def atividade_tema(
    token: str = Form(...),
    tema: str = Form(...),
    tipo: str = Form(...),
    quantidade: str = Form(...),
    infos_extras: str = Form("")
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    atividade = gerar_atividade_tema(tema, tipo, infos_extras, quantidade)
    return {"tipo": tipo, "tema": tema, "atividade": atividade}

@app.post("/gerar-atividade-pdf")
def atividade_pdf(
    token: str = Form(...),
    tipo: str = Form(...),
    quantidade: str = Form(...),
    infos_extras: str = Form(...),
    consulta: str = Form(""),
    arquivo: UploadFile = File(...)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    conteudo = arquivo.file.read()
    atividade = gerar_atividade_pdf(conteudo, tipo, quantidade, consulta, infos_extras)
    return {"tipo": tipo, "tema": "Baseado em arquivo", "atividade": atividade}

# ------- GERADOR DE ROTEIRO  -----------------------
@app.post("/gerar-roteiro-tema")
def roteiro_tema(
    token: str = Form(...),
    tema: str = Form(...),
    tipo: str = Form(...),
    tempo: str = Form(...),
    infos_extras: str = Form("")
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    roteiro = gerar_roteiro_tema(tema, tipo, infos_extras, tempo)
    return {"tipo": tipo, "tema": tema, "roteiro": roteiro}

@app.post("/gerar-roteiro-pdf")
def roteiro_pdf(
    token: str = Form(...),
    tipo: str = Form(...),
    tempo: str = Form(...),
    infos_extras: str = Form(...),
    consulta: str = Form(""),
    arquivo: UploadFile = File(...)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    conteudo = arquivo.file.read()
    roteiro = gerar_roteiro_pdf(conteudo, tipo, tempo, consulta, infos_extras)
    return {"tipo": tipo, "tema": "Baseado em arquivo", "roteiro": roteiro}



# ------- TRANSCRIÇÃO DE ÁUDIO -----------------------
@app.post("/transcrever-audio")
async def transcrever_audio(
    token: str = Form(...),
    arquivo: UploadFile = File(...),
    idioma: str = Form(...)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")

    # salva input (útil para auditoria/debug)
    fname = f"{uuid.uuid4()}_{arquivo.filename or 'audio'}"
    in_path = os.path.join(STORAGE_DIR, "in", fname)
    f: BufferedWriter
    with open(in_path, "wb") as f:
        shutil.copyfileobj(arquivo.file, f)

    try:
        result = transcrever_audio_gemini(in_path, idioma)
        # with open(os.path.join(STORAGE_DIR, "out", f"{fname}.json"), "w", encoding="utf-8") as out:
        #     json.dump(result, out, ensure_ascii=False, indent=2)

        return JSONResponse({
            "ok": True,
            "filename": arquivo.filename,
            "idioma": idioma,
            "result": result   # <- language, text, segments[]
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao transcrever: {e}")

# ------- TTS: TEXTO -> ÁUDIO (GOOGLE CLOUD) -----------------------
@app.post("/audio-sintetizar")
async def audio_sintetizar(
    token: str = Form(...),
    texto: str = Form(...),
    idioma: str = Form("pt-BR"),
    voz: str = Form("pt-BR-Wavenet-A"),
    formato: str = Form("MP3"),               # MP3 | OGG_OPUS | LINEAR16
    velocidade: float | None = Form(None),    # 0.25–4.0
    tom: float | None = Form(None),           # -20.0–20.0
    ganho_db: float | None = Form(None)       # -96.0–16.0
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")

    try:
        audio_bytes = synthesize_speech(
            text=texto,
            language_code=idioma,
            voice_name=voz,
            audio_format=formato,
            speaking_rate=velocidade,
            pitch=tom,
            volume_gain_db=ganho_db
        )

        # salva saída (auditoria)
        suffix = ".mp3" if formato.upper() == "MP3" else ".ogg" if formato.upper() == "OGG_OPUS" else ".wav"
        out_name = f"{uuid.uuid4()}_tts{suffix}"
        out_path = os.path.join(STORAGE_DIR, "out", out_name)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)

        ct_map = {
            "MP3": ("audio/mpeg", "audio.mp3"),
            "OGG_OPUS": ("audio/ogg", "audio.ogg"),
            "LINEAR16": ("audio/wav", "audio.wav"),
        }
        content_type, filename = ct_map.get(formato.upper(), ("audio/mpeg", "audio.mp3"))

        headers = {
            "Content-Disposition": f'inline; filename="{filename}"',
            "X-File-Saved": out_name
        }
        return Response(content=audio_bytes, media_type=content_type, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao sintetizar áudio: {e}")

# ------- TRANSCRIÇÃO DE IMAGEM -----------------------
@app.post("/descrever-imagem")
def descrever_imagem(
    token: str = Form(...),
    idioma_saida: str = Form("pt-BR"),
    tom: str = Form("neutro"),
    quantidade_caracteres: int = Form(...),
    arquivo: UploadFile = File(...)
):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")

    conteudo = arquivo.file.read()
    if not conteudo or len(conteudo) < 20:
        raise HTTPException(status_code=400, detail="Arquivo inválido ou vazio.")

    mime = arquivo.content_type or "image/jpeg"
    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem (png, jpg, webp, etc.).")

    #llm = build_chain()
    resultado = describe_image(
        image_bytes=conteudo,
        mime_type=mime,
        idioma_saida=idioma_saida,
        tom=tom,
        quantidade_caracteres=quantidade_caracteres
    )

    return {"descricao": resultado}