import torch
import torchaudio
from onsets_and_frames.mel import melspectrogram

def load_audio(filepath, sr=16000):
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sr)
    return waveform

def audio_to_mel(filepath, sr=16000):
    waveform = load_audio(filepath, sr=sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # melspectrogramをCPUに移動
    mel_module = melspectrogram.cpu() if hasattr(melspectrogram, "cpu") else melspectrogram
    waveform = waveform.cpu()
    mel = mel_module(waveform)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    return mel

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python make_mel.py path/to/audio.wav")
        exit(1)
    mel = audio_to_mel(sys.argv[1])
    print("mel shape:", mel.shape)