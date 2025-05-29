import torch
from onsets_and_frames import OnsetsAndFrames, load_audio, audio_to_input
from onsets_and_frames.transcriber import load_weights
from onsets_and_frames.midi import save_midi

# 必要な定数を定義（train.pyと同じ値を使うこと）
N_MELS = 229
MIN_MIDI = 21
MAX_MIDI = 108
N_KEYS = MAX_MIDI - MIN_MIDI + 1
MODEL_COMPLEXITY = 64

# 推論に使うWAVファイル
audio_path = 'unaligned-supervision-master/datasets/a3JRV466Ci.wav'
# 学習済みモデルのパス
ckpt_path = 'runs/transcriber-250517-130455/transcriber_13.pt'

# モデル構築
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transcriber = OnsetsAndFrames(N_MELS, N_KEYS, MODEL_COMPLEXITY)
saved_transcriber = torch.load(ckpt_path, map_location=device)
load_weights(transcriber, saved_transcriber)
transcriber.eval()
transcriber.to(device)

# 音声ファイルをロードして前処理
audio = load_audio(audio_path)
inputs = audio_to_input(audio)  # (N_MELS, T)
inputs = inputs.unsqueeze(0).to(device)  # (1, N_MELS, T)

# 推論
with torch.no_grad():
    outputs = transcriber(inputs)
    onset_pred = outputs['onset'] > 0.5  # オンセット検出
    frame_pred = outputs['frame'] > 0.5  # ノート持続検出

# 必要に応じて後処理（例：MIDIファイル化など）
print("Onset shape:", onset_pred.shape)
print("Frame shape:", frame_pred.shape)

# MIDIファイルとして保存
# outputs, onset_pred, frame_pred からノート情報を抽出しMIDI化
# onsets_and_frames.midi.save_midi を利用
midi_path = 'output.mid'
# 必要に応じてvelocityやその他の情報も渡せます
save_midi(
    midi_path,
    onset_pred[0].cpu().numpy(),
    frame_pred[0].cpu().numpy(),
    velocity=outputs['velocity'][0].cpu().numpy() if 'velocity' in outputs else None,
    min_midi=MIN_MIDI
)
print(f"MIDIファイルを {midi_path} に保存しました")