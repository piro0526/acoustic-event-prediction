import os

os.environ["SDL_AUDIODRIVER"] = "dummy"

from maai import Maai, MaaiInput, MaaiOutput

wav = MaaiInput.Wav(wav_file_path='data/Linear Digressions_What makes a machine learning algorithm "superhuman"_30sec.wav')
zero = MaaiInput.Zero()

output = MaaiOutput.GuiPlot(frame_rate=10)

maai = Maai(
    mode="bc",
    lang="jp",
    frame_rate=20,
    audio_ch1=wav,
    audio_ch2=zero,
    device="cpu"
)

maai.start()

while True:
    result = maai.get_result()
    output.update(result)