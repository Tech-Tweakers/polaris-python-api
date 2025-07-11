import os
import subprocess
from moviepy import AudioFileClip
from TTS.api import TTS

from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.models.xtts import XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# 👇 esta linha é essencial:
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

COQUI_SPEAKER_WAV = os.getenv("COQUI_SPEAKER_WAV", "polaris-voice.wav")

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=False,
    gpu=False,
)


def limpar_texto(texto: str) -> str:
    return (
        texto.replace("...", ".")
        .replace("—", "")
        .replace("“", "")
        .replace("”", "")
        .strip()
    )


def tts_coqui(texto: str, output_path: str) -> str:
    wav_temp = output_path.replace(".mp3", ".wav")
    wav_clean = output_path.replace(".mp3", "_clean.wav")

    tts.tts_to_file(
        text=limpar_texto(texto),
        speaker_wav=COQUI_SPEAKER_WAV,
        language="pt",
        file_path=wav_temp,
        speed=1.5,
        split_sentences=False,
    )

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_temp,
                "-af",
                "silenceremove=stop_periods=-1:stop_duration=0.8:stop_threshold=-45dB",
                wav_clean,
            ],
            check=True,
        )
        wav_final = wav_clean
    except:
        wav_final = wav_temp

    audio = AudioFileClip(wav_final)
    audio.write_audiofile(output_path, codec="libmp3lame", bitrate="48k", fps=16000)

    for f in [wav_temp, wav_clean]:
        if os.path.exists(f):
            os.remove(f)

    return output_path
