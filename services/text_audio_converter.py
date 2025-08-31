import os
import json
from typing import Optional
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.oauth2 import service_account


load_dotenv()
# cria o client uma vez (singleton de módulo)
def _build_client() -> texttospeech.TextToSpeechClient:
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        credentials_info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        return texttospeech.TextToSpeechClient(credentials=credentials)
    # fallback: credenciais padrão se estiver dentro do GCP
    return texttospeech.TextToSpeechClient()

_client = _build_client()

def synthesize_speech(
    text: str,
    language_code: str,
    voice_name: str,
    audio_format: str = "MP3",
    speaking_rate: Optional[float] = None,  # 0.25–4.0
    pitch: Optional[float] = None,          # -20.0–20.0 (semitones)
    volume_gain_db: Optional[float] = None  # -96.0–16.0
) -> bytes:
    """
    Converte texto em áudio com Google Cloud TTS.
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
    )

    enc_map = {
        "MP3": texttospeech.AudioEncoding.MP3,
        "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS,
        "LINEAR16": texttospeech.AudioEncoding.LINEAR16,  # WAV PCM
    }
    encoding = enc_map.get(audio_format.upper(), texttospeech.AudioEncoding.MP3)

    audio_cfg = {
        "audio_encoding": encoding,
    }
    if speaking_rate is not None:
        audio_cfg["speaking_rate"] = speaking_rate
    if pitch is not None:
        audio_cfg["pitch"] = pitch
    if volume_gain_db is not None:
        audio_cfg["volume_gain_db"] = volume_gain_db

    audio_config = texttospeech.AudioConfig(**audio_cfg)

    response = _client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    return response.audio_content
 