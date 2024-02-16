from audiocraft.models import MAGNeT
from audiocraft.data.audio import audio_write

model = MAGNeT.get_pretrained("facebook/audio-magnet-medium")

descriptions = ["heavy rain", "windy deep forest", "gun fight"]

wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
