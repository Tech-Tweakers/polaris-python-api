from TTS.api import TTS
from inspect import signature
import os

# 🚨 Troque pelo modelo que você estiver usando
MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"  # ou o seu modelo com \n nos speakers

# 🔁 Texto de teste
texto = "Olá, eu sou Polaris, sua assistente local e inteligente."

# 📁 Arquivo de saída
file_path = "voz.wav"

# 🚀 Carrega o modelo
print(f"🔄 Carregando modelo {MODEL_NAME}...")
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)

# 🎙️ Lista de speakers disponíveis
print("\n🎤 Vozes disponíveis:")
if hasattr(tts, "speakers") and tts.speakers:
    for spk in tts.speakers:
        print(f"- {repr(spk)}")
    speaker = tts.speakers[0]  # Usa a primeira como padrão
else:
    print("Nenhuma voz listada — o modelo pode ser mono-speaker.")
    speaker = None

# 🌍 Idiomas
if hasattr(tts, "languages") and tts.languages:
    print("\n🌐 Idiomas disponíveis:")
    for lang in tts.languages:
        print(f"- {lang}")

# 🔎 Verifica os parâmetros suportados
print("\n⚙️ Parâmetros aceitos por tts_to_file:")
sig = signature(tts.tts_to_file)
params_aceitos = sig.parameters.keys()
for param in params_aceitos:
    print(f"- {param}")

# 📦 Monta os kwargs dinamicamente
kwargs = {
    "text": texto,
    "file_path": file_path,
}

if "speaker" in params_aceitos and speaker:
    kwargs["speaker"] = speaker

if "language" in params_aceitos:
    kwargs["language"] = "pt-br"

# Parâmetros bônus: só adiciona se forem aceitos
extras = {
    "sample_rate": 48000,
    "speed": 1.3,
    "noise_scale": 0.33,
    "noise_scale_w": 0.7,
    "length_scale": 0.95,
}

for key, val in extras.items():
    if key in params_aceitos:
        kwargs[key] = val

print("\n🗣️ Gerando áudio...")
tts.tts_to_file(**kwargs)

# ✅ Toca o áudio (Linux/Mac)
if os.name == "posix":
    os.system(f"aplay {file_path} || afplay {file_path}")
elif os.name == "nt":
    os.system(f"start {file_path}")
