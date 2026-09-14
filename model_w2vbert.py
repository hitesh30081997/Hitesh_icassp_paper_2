"""
model_w2vbert.py
wav2vec2-BERT 2.0 (facebook/w2v-bert-2.0) backbone variant.

IMPORTANT ARCHITECTURAL DIFFERENCE vs model.py's HubertSLUModel:
Wav2Vec2-BERT does NOT take raw waveform as input the way Wav2Vec2/HuBERT/
WavLM do. Per the official docs (Meta's "Seamless" paper), it follows the
Wav2Vec2-Conformer architecture but uses mel-spectrogram features as input
instead of raw audio -- specifically HF's SeamlessM4TFeatureExtractor,
producing 160-dim input_features (two stacked 80-bin mel frames) plus a
FEATURE-level attention_mask, not a raw-sample one. There is no raw-
waveform CNN feature_extractor submodule to freeze the way there is for
wav2vec2/HuBERT/WavLM, so most of HubertSLUModel's freeze/length logic
does not carry over as-is.

To keep dataset.py / collate_fn completely unchanged (still handing over
raw waveform + a raw-sample attention_mask, same as every other backbone
in this project), this model runs the OFFICIAL SeamlessM4TFeatureExtractor
itself, inside forward(), converting raw waveform -> input_features each
batch. Reusing a hand-rolled mel frontend (like we did for the from-
scratch Conformer) would NOT match this checkpoint's actual pretraining
preprocessing and would likely hurt performance -- so we intentionally use
their real feature extractor here instead, at some CPU<->GPU round-trip
cost per batch (optimizable later by precomputing features in dataset.py
if it becomes a bottleneck).

Two things below are NOT independently verified against a running
transformers install (no internet/disk budget for a full install in the
environment this was written in) -- please sanity-check them once you run
this for real:
  1. `self.w2v_bert.encoder.layers` as the path to per-block freezing.
     Wrapped in a try/except below with a clear message if the attribute
     doesn't exist in your installed transformers version.
  2. Exact default of `config.hidden_size` for this checkpoint (used
     generically via `.config.hidden_size`, which IS a name I confirmed
     is used consistently across this whole model family, so this part
     is low-risk even without a live install to check against).

Requires: pip install "transformers>=4.38" (first release with
Wav2Vec2-BERT support) and torchaudio (used internally by
SeamlessM4TFeatureExtractor for mel computation).
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

from model import AttentionPool  # reused as-is, backbone-agnostic


class Wav2Vec2BertSLUModel(nn.Module):
    def __init__(
        self,
        num_intents: int,
        ctc_vocab_size: int,
        pad_id: int,
        blank_id: int,
        model_name: str = "facebook/w2v-bert-2.0",
        freeze_encoder_layers: int = 0,
        slot_lstm_hidden: int = 256,
        slot_lstm_layers: int = 2,
        dropout: float = 0.1,
        blank_bias_init: float = -2.0,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.pad_id = pad_id
        self._ctc_blank_id = blank_id
        self.sample_rate = sample_rate

        # The exact preprocessing this checkpoint was pretrained with --
        # do not substitute a hand-rolled equivalent.
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)

        self.w2v_bert = Wav2Vec2BertModel.from_pretrained(model_name)
        hidden_size = self.w2v_bert.config.hidden_size

        if freeze_encoder_layers > 0:
            try:
                for i, layer in enumerate(self.w2v_bert.encoder.layers):
                    if i < freeze_encoder_layers:
                        for p in layer.parameters():
                            p.requires_grad = False
            except AttributeError as e:
                raise AttributeError(
                    "Could not find 'encoder.layers' on this Wav2Vec2BertModel "
                    "-- the internal module path may differ in your installed "
                    "transformers version. Run `print(self.w2v_bert)` to find "
                    "the correct attribute path and update this loop, or set "
                    "--freeze_encoder_layers 0 to skip freezing entirely."
                ) from e

        # --- Intent branch ---
        self.intent_pool = AttentionPool(hidden_size)
        self.intent_dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_intents)

        # --- Slot filling branch (BiLSTM + CTC, same pattern as model.py) ---
        self.slot_lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=slot_lstm_hidden,
            num_layers=slot_lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if slot_lstm_layers > 1 else 0.0,
        )
        self.slot_dropout = nn.Dropout(dropout)
        self.ctc_head = nn.Linear(slot_lstm_hidden * 2, ctc_vocab_size)
        with torch.no_grad():
            self.ctc_head.bias[blank_id] = blank_bias_init

    def _extract_features(self, input_values, attention_mask):
        """Convert raw waveform (as handed over by dataset.py/collate_fn,
        same as every other backbone in this project) into this model's
        expected input_features via the OFFICIAL feature extractor."""
        device = input_values.device
        waveforms = []
        for i in range(input_values.shape[0]):
            L = int(attention_mask[i].sum().item())
            waveforms.append(input_values[i, :L].detach().cpu().numpy())

        processed = self.feature_extractor(
            waveforms, sampling_rate=self.sample_rate,
            return_tensors="pt", padding=True,
        )
        input_features = processed["input_features"].to(device)
        feat_attention_mask = processed["attention_mask"].to(device)
        return input_features, feat_attention_mask

    def forward(self, input_values, attention_mask, ctc_target_lens=None):
        input_features, feat_attention_mask = self._extract_features(
            input_values, attention_mask
        )

        outputs = self.w2v_bert(
            input_features=input_features,
            attention_mask=feat_attention_mask,
        )
        enc_out = outputs.last_hidden_state              # (B, T, H)
        # No established "semantic layer" prior for this pretraining
        # objective the way there is for HuBERT -- using the final layer
        # by default. output_hidden_states=True + layer selection could be
        # added the same way as model.py's layer_fusion if you want to
        # explore that.
        feat_lens = feat_attention_mask.sum(-1)

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
