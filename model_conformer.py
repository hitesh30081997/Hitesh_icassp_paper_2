"""
model_conformer.py
A from-scratch Conformer-encoder variant of the SLU model -- no pretrained
weights, no self-supervised backbone. Frontend, encoder, and both task
heads all train from random initialization directly on SLURP.

READ BEFORE USING -- this is a real departure from model.py, not a
drop-in swap:

  - Input: log-mel filterbank features computed INSIDE this model (so
    dataset.py / collate_fn stay 100% unchanged -- they still hand over
    raw waveform + attention_mask, exactly like the HuBERT/WavLM path).
  - No pretraining: expect this to underperform the SSL backbones you've
    been tuning, likely by a wide margin, especially at low SNR. That
    gap is precisely why SSL pretraining is standard practice for speech
    tasks with dataset sizes in SLURP's range. Consider this an ablation
    point unless you pretrain this encoder on a larger corpus first.
  - No "semantic layer" concept: without an SSL pretraining objective,
    there's no established reason an intermediate layer would beat the
    final one, so both heads read from the FINAL Conformer output.

Requires: pip install torchaudio (a reasonably recent version -- needs
torchaudio.models.Conformer, added in torchaudio 0.12+).
"""
import torch
import torch.nn as nn
import torchaudio

from model import AttentionPool  # reused as-is; it's backbone-agnostic


class LogMelFrontend(nn.Module):
    """Raw waveform -> log-mel filterbank features, computed on-device so
    dataset.py never needs to change."""

    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        super().__init__()
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0,
        )

    def forward(self, waveform, attention_mask):
        # waveform: (B, num_samples)
        mel = self.mel(waveform)                        # (B, n_mels, T)
        log_mel = torch.log(mel.clamp(min=1e-5)).transpose(1, 2)  # (B, T, n_mels)

        sample_lens = attention_mask.sum(-1)
        feat_lens = torch.div(sample_lens, self.hop_length, rounding_mode="floor") + 1
        feat_lens = feat_lens.clamp(max=log_mel.shape[1])
        return log_mel, feat_lens


class Conv2dSubsampling(nn.Module):
    """Standard 4x time-downsampling conv front-end (as used in ESPnet/NeMo
    Conformer ASR recipes): reduces sequence length before the relatively
    expensive self-attention layers, and gives a few frames of local
    context per output step up front."""

    def __init__(self, n_mels, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        freq_out = (((n_mels - 3) // 2 + 1) - 3) // 2 + 1
        self.out_proj = nn.Linear(d_model * freq_out, d_model)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)                               # (B, 1, T, n_mels)
        x = self.conv(x)                                 # (B, d_model, T', F')
        B, C, Tp, Fp = x.shape
        x = x.transpose(1, 2).contiguous().view(B, Tp, C * Fp)
        x = self.out_proj(x)                              # (B, T', d_model)

        out_lengths = ((lengths - 3) // 2 + 1 - 3) // 2 + 1
        out_lengths = out_lengths.clamp(min=1, max=Tp)
        return x, out_lengths


class ConformerSLUModel(nn.Module):
    def __init__(
        self,
        num_intents: int,
        ctc_vocab_size: int,
        pad_id: int,
        blank_id: int,
        n_mels: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_layers: int = 12,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        slot_lstm_hidden: int = 256,
        slot_lstm_layers: int = 2,
        blank_bias_init: float = -2.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self._ctc_blank_id = blank_id

        self.frontend = LogMelFrontend(n_mels=n_mels)
        self.subsample = Conv2dSubsampling(n_mels, d_model)
        self.conformer = torchaudio.models.Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

        # --- Intent branch ---
        self.intent_pool = AttentionPool(d_model)
        self.intent_dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(d_model, num_intents)

        # --- Slot filling branch (BiLSTM + CTC, same pattern as model.py) ---
        self.slot_lstm = nn.LSTM(
            input_size=d_model, hidden_size=slot_lstm_hidden,
            num_layers=slot_lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if slot_lstm_layers > 1 else 0.0,
        )
        self.slot_dropout = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(slot_lstm_hidden * 2, ctc_vocab_size)
        with torch.no_grad():
            self.ctc_head.bias[blank_id] = blank_bias_init

    def forward(self, input_values, attention_mask, ctc_target_lens=None):
        mel, mel_lens = self.frontend(input_values, attention_mask)
        sub, feat_lens = self.subsample(mel, mel_lens)
        enc_out, feat_lens = self.conformer(sub, feat_lens)   # (B, T, d_model)

        T = enc_out.shape[1]
        frame_mask = (torch.arange(T, device=enc_out.device)[None, :] < feat_lens[:, None]).long()

        # ---- Intent classification ----
        pooled = self.intent_pool(enc_out, frame_mask)
        intent_logits = self.intent_head(self.intent_dropout(pooled))

        # ---- Slot filling (CTC) ----
        slot_feats, _ = self.slot_lstm(enc_out)
        slot_feats = self.slot_dropout(slot_feats)
        ctc_logits = self.ctc_head(slot_feats)
        log_probs = torch.log_softmax(ctc_logits, dim=-1).transpose(0, 1)

        return {
            "intent_logits": intent_logits,
            "ctc_log_probs": log_probs,
            "feat_lens": feat_lens,
        }

    def compute_loss(self, batch, outputs, ctc_loss_weight=1.0):
        intent_loss = nn.functional.cross_entropy(
            outputs["intent_logits"], batch["intent_ids"]
        )
        ctc_loss = nn.functional.ctc_loss(
            outputs["ctc_log_probs"],
            batch["ctc_targets"],
            outputs["feat_lens"],
            batch["ctc_target_lens"],
            blank=self._ctc_blank_id,
            zero_infinity=True,
        )
        total = intent_loss + ctc_loss_weight * ctc_loss
        return total, intent_loss.detach(), ctc_loss.detach()
