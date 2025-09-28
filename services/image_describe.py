import os
import base64
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
load_dotenv()

# Inicializa LLM
LLM_GEMINI = os.getenv("LLM_GEMINI")
llm = ChatGoogleGenerativeAI(model=LLM_GEMINI, temperature=0.4, max_output_tokens=8196)

def _to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def describe_image(
    image_bytes: bytes,
    mime_type: str,
    idioma_saida: str = "pt-BR",
    tom: str = "neutro",
    quantidade_caracteres: int = 160,
) -> Dict[str, Any]:
    """Gera descrição acessível para uma imagem usando o llm global."""

    if not image_bytes or len(image_bytes) < 20:
        raise ValueError("Imagem vazia ou inválida.")
    if not mime_type.startswith("image/"):
        raise ValueError("Mime-type inválido. Envie uma imagem.")

    sys_prompt = (
        "Você é um assistente que gera descrições acessíveis de imagens (alt-text e descrição longa). "
        "Não invente detalhes: quando algo não for visível, declare que não é possível inferir. "
        "Se existir texto na imagem, transcreva em 'texto_detectado'. "
        "Responda SEMPRE em JSON válido, seguindo exatamente este esquema:\n\n"
        "{\n"
        f'  "alt_text": "string curta (até 160 caracteres)",\n'
        f'  "descricao_longa": "parágrafo(s) objetivo(s), até {quantidade_caracteres}  caracteres (contando espaços).",\n'
        '  "texto_detectado": "texto OCR se houver",\n'
        '  "tags": ["3 a 10 palavras-chave"],\n'
        '  "seguranca": {\n'
        '      "nudity_suspeita": false,\n'
        '      "violencia_suspeita": false,\n'
        '      "info_pessoal_suspeita": false\n'
        "  }\n"
        "}\n\n"
        f"Idioma de saída: {idioma_saida}.\n"
        f"Tom: {tom}.\n"
        f"Imagem: ''.\n"
    )

    user_text = "Descreva a imagem para alguém que não pode ver.\nResponda somente com o JSON especificado."

    msg = HumanMessage(
        content=[
            {"type": "text", "text": sys_prompt},
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": _to_data_url(image_bytes, mime_type)},
        ]
    )

    resp = llm.invoke([msg])

    # Se vier string, usa direto; se vier objeto, pega .content
    if isinstance(resp, str):
        text = resp.strip()
    elif hasattr(resp, "content"):
        text = (resp.content or "").strip()
    else:
        text = str(resp).strip()

    # normaliza se vier com ```json
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "").replace("json\r\n", "").strip()

    try:
        data = json.loads(text)
    except Exception:
        data = {
            "alt_text": "",
            "descricao_longa": text,
            "texto_detectado": "",
            "tags": [],
            "seguranca": {
                "nudity_suspeita": False,
                "violencia_suspeita": False,
                "info_pessoal_suspeita": False,
            },
        }

    return data