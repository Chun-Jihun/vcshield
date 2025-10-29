# config.py
import os

# 루트 경로들
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# LibriSpeech dev-clean 경로
LIBRISPEECH_ROOT = os.path.join(PROJECT_ROOT, "data", "train", "dev-clean")

# 오디오 파라미터
SAMPLE_RATE = 16000  # LibriSpeech 기본은 16kHz
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = 8000  # 16kHz의 Nyquist는 8k

# 배치 크기 등 (학습 시 사용)
BATCH_SIZE = 8
NUM_WORKERS = 4

# resemblizer 관련 설정 (placeholder)
RESEMBLIZER_MODEL_PATH = os.path.join(PROJECT_ROOT, "resemblizer")
# 예: 사전학습된 resemblizer 가중치/모델 파일 위치 등
