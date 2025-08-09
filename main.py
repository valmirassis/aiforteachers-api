from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from services.question_generator import gerar_questoes_tema, gerar_questoes_pdf
import os

app = FastAPI()

API_TOKEN = os.getenv("CODIGO_ACESSO")

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
