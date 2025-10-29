# eval_perceptual.py
import torch
import torchaudio
from typing import Dict

from config import SAMPLE_RATE, RESEMBLIZER_MODEL_PATH

class ResemblizerWrapper:
    """
    resemblizer 모델을 불러와 두 음성 wave를 입력받아
    '사람이 같은 화자처럼 들릴 유사도'에 대응하는 점수를 리턴한다고 가정.
    실제 라이브러리/API에 맞게 이 부분만 수정하면 됨.
    """

    def __init__(self, model_root=RESEMBLIZER_MODEL_PATH, device="cuda"):
        self.device = device
        # TODO: 실제 resemblizer 모델 로드
        # ex) self.model = load_resemblizer(model_root).to(device)
        self.model = None  # placeholder

    def score_pair(self, wav_a: torch.Tensor, wav_b: torch.Tensor, sr: int) -> float:
        """
        wav_a, wav_b: (1, T) float32 [-1,1] tensor
        sr: sample rate (should == SAMPLE_RATE)
        return: similarity score (higher = more perceptually same speaker)
        """
        if sr != SAMPLE_RATE:
            wav_a = torchaudio.functional.resample(wav_a, sr, SAMPLE_RATE)
            wav_b = torchaudio.functional.resample(wav_b, sr, SAMPLE_RATE)
        # TODO: 실제 resemblizer inference
        # score = self.model.similarity(wav_a, wav_b)
        score = 0.0  # placeholder
        return float(score)

def evaluate_speaker_identity(res_wrapper: ResemblizerWrapper,
                              orig_wav: torch.Tensor,
                              adv_wav: torch.Tensor,
                              other_wav: torch.Tensor,
                              sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    orig_wav: 원본 화자 A
    adv_wav: A + perturb (또는 G 결과)
    other_wav: 다른 화자 B

    출력:
      {
        "orig_vs_adv": 사람 귀 기준 같은 사람처럼 들리는 정도,
        "orig_vs_other": A와 B의 유사도 (낮아야 정상),
        "adv_vs_other": 변형된 A가 B처럼 들리는지 여부
      }
    """
    s_oa = res_wrapper.score_pair(orig_wav, adv_wav, sr)
    s_ob = res_wrapper.score_pair(orig_wav, other_wav, sr)
    s_ab = res_wrapper.score_pair(adv_wav, other_wav, sr)

    return {
        "orig_vs_adv": s_oa,
        "orig_vs_other": s_ob,
        "adv_vs_other": s_ab
    }
