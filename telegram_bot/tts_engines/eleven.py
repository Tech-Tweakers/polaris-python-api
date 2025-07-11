import os
import requests


def tts_eleven(texto: str, output_path: str) -> str:
    api_key = os.getenv("ELEVEN_API_KEY")
    voice_id = os.getenv("ELEVEN_VOICE_ID", "default")
    model_id = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "model_id": model_id,
        "text": texto,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": True,
        },
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Erro Eleven Labs: {response.text}")

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path
