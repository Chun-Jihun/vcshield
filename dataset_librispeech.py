# dataset_librispeech.py
import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from config import LIBRISPEECH_ROOT, SAMPLE_RATE

class LibriSpeechSpeakerDataset(Dataset):
    """
    dev-clean에서 특정 화자(또는 여러 화자)를 로드하는 Dataset.
    각 item은:
        {
          "waveform": (1, T) tensor (float32, -1~1),
          "speaker_id": str,
          "utt_path": str
        }
    """

    def __init__(self, root_dir=LIBRISPEECH_ROOT, speaker_list=None):
        """
        root_dir: dev-clean 경로
        speaker_list: [ '84', '174', ... ] 처럼 사용할 speaker ID 제한.
                      None이면 root_dir 내부의 모든 speaker 폴더 사용.
        """
        self.root_dir = root_dir

        # 1) 화자 폴더들 수집
        all_speakers = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()
        ]
        if speaker_list is not None:
            self.speakers = sorted([s for s in all_speakers if s in speaker_list])
        else:
            self.speakers = sorted(all_speakers)

        # 2) 각 화자 폴더 아래의 실제 음성 파일(.flac/.wav) 경로 전부 모으기
        self.items = []
        for spk in self.speakers:
            spk_dir = os.path.join(root_dir, spk)
            # LibriSpeech 구조: spk_id/chapter_id/*.flac
            chapter_dirs = [
                d for d in os.listdir(spk_dir)
                if os.path.isdir(os.path.join(spk_dir, d))
            ]
            for ch in chapter_dirs:
                chapter_dir = os.path.join(spk_dir, ch)
                wav_paths = glob.glob(os.path.join(chapter_dir, "*.wav"))
                flac_paths = glob.glob(os.path.join(chapter_dir, "*.flac"))
                audio_paths = sorted(wav_paths + flac_paths)

                for ap in audio_paths:
                    self.items.append({
                        "speaker_id": spk,
                        "utt_path": ap
                    })

        print(f"[LibriSpeechSpeakerDataset] total {len(self.items)} utterances from {len(self.speakers)} speakers.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_meta = self.items[idx]
        spk = item_meta["speaker_id"]
        path = item_meta["utt_path"]

        wav, sr = torchaudio.load(path)  # wav: (channels, T)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        # 모노로 강제
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        # float32 normalize (-1~1) torchaudio.load already float32 -1~1 usually fine.

        return {
            "waveform": wav,            # (1, T)
            "speaker_id": spk,
            "utt_path": path
        }
