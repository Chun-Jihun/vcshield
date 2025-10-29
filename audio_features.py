# audio_features.py
import torch
import torchaudio
from config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, WIN_LENGTH,
    N_MELS, FMIN, FMAX
)

# torchaudio의 MelSpectrogram 모듈을 사용해서 멜로 변환
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    f_min=FMIN,
    f_max=FMAX,
    n_mels=N_MELS,
    power=1.0  # magnitude spec
)

# 로그 멜 등 후처리도 여기에 포함해도 된다
def wav_to_mel(waveform: torch.Tensor):
    """
    waveform: (1, T)
    returns mel: (n_mels, time)
    """
    with torch.no_grad():
        mel = _mel_transform(waveform)  # (n_mels, time)
        mel_db = torch.log(mel + 1e-6)
    return mel_db

def mel_to_wav(mel_db: torch.Tensor):
    """
    placeholder:
    실제로는 HiFi-GAN / vocoder 모델 forward를 호출해 waveform을 복원해야 한다.
    여기서는 인터페이스만 정의.
    """
    raise NotImplementedError("Integrate neural vocoder (e.g., HiFi-GAN) here.")
