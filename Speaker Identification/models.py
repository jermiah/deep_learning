import torch
import torchaudio
import torch.nn as nn

# === Constants (Optional: include if needed for preprocessing)
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION_SEC = 12
MAX_SAMPLES = TARGET_SAMPLE_RATE * MAX_AUDIO_DURATION_SEC

# === Audio Preprocessing ===
def preprocess_audio(filepath, use_repeat_padding=True):
    try:
        waveform, sample_rate = torchaudio.load(filepath)
        waveform = waveform.to(torch.float32)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        if waveform.numel() == 0 or not torch.isfinite(waveform).all():
            print(f"⚠️ Skipping invalid audio: {filepath}")
            return None

        if waveform.shape[0] < MAX_SAMPLES:
            if use_repeat_padding:
                repeat_factor = (MAX_SAMPLES + waveform.shape[0] - 1) // waveform.shape[0]
                waveform = waveform.repeat(repeat_factor)[:MAX_SAMPLES]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, MAX_SAMPLES - waveform.shape[0]))
        elif waveform.shape[0] > MAX_SAMPLES:
            waveform = waveform[:MAX_SAMPLES]

        return waveform.unsqueeze(0)

    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None




class EmbeddingClassifierBN(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.final = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        residual = x
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = x + residual
        x = self.drop(self.relu(self.bn3(self.fc3(x))))
        return self.final(x)




