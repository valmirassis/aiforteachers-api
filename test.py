from google.cloud import texttospeech
import os


def sintetizar_audio_google_cloud_mp3(
        texto: str,
        nome_arquivo_saida: str = "audio_gerado.mp3",  # Padrão agora é .mp3
        storage_dir: str = "storage/out"
):
    """
    Gera um áudio em formato MP3 usando a API Google Cloud Text-to-Speech.
    """
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=texto)
        voice = texttospeech.VoiceSelectionParams(
            language_code="pt-BR",
            name="pt-BR-Wavenet-A"  # Voz feminina de alta qualidade
        )

        # --- A ÚNICA MUDANÇA É AQUI ---
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3  # <-- Alterado para MP3
        )
        # -----------------------------

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        mp3_path = os.path.join(storage_dir, nome_arquivo_saida)
        os.makedirs(storage_dir, exist_ok=True)

        with open(mp3_path, "wb") as out:
            out.write(response.audio_content)
            print(f"Áudio MP3 salvo com sucesso em '{mp3_path}'")

        return mp3_path

    except Exception as e:
        print(f"Ocorreu um erro na API de Text-to-Speech: {e}")
        print("Verifique se sua autenticação do Google Cloud está configurada.")
        return None


# --- Exemplo de uso ---
if __name__ == "__main__":
    texto_exemplo = "Agora, este áudio será gerado diretamente no formato MP3, que é muito mais leve."
    sintetizar_audio_google_cloud_mp3(texto_exemplo)