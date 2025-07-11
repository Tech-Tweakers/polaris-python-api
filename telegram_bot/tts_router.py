import os
from tts_engines.eleven import tts_eleven
from tts_engines.coqui import tts_coqui


def gerar_audio(texto: str, path: str) -> str:
    engine = os.getenv("TTS_ENGINE", "coqui").lower()
    if engine == "eleven":
        return tts_eleven(texto, path)
    elif engine == "coqui":
        return tts_coqui(texto, path)
    else:
        raise ValueError(f"⚠️ Engine TTS não reconhecida: {engine}")
