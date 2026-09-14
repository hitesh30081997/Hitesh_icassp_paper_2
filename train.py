"""
train.py
End-to-end training for the HuBERT SLURP SLU model.

Usage:
    python train.py \
        --slurp_jsonl_dir /path/to/slurp/dataset/slurp \
        --audio_root /path/to/slurp/audio \
        --output_dir ./ckpt \
        --hubert_name facebook/hubert-base-ls960 \
        --semantic_layer 8 \
        --epochs 20 --batch_size 8 --lr 3e-5
"""
import argparse
import functools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vocab import CTCVocab, build_label_maps, extract_slots_from_tagged_text
from dataset import SlurpDataset, collate_fn
from model import HubertSLUModel
from model_conformer import ConformerSLUModel
from model_w2vbert import Wav2Vec2BertSLUModel
from metrics import intent_accuracy, slu_f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slurp_jsonl_dir", required=True,
                    help="Directory containing train.jsonl / devel.jsonl / test.jsonl")
    p.add_argument("--audio_root", required=True,
                    help="Directory containing slurp_real/ and slurp_synth/")
    p.add_argument("--output_dir", default="./ckpt")
    p.add_argument("--backbone", choices=["ssl", "conformer", "w2vbert"], default="ssl",
                    help="'ssl': pretrained HuBERT/WavLM/etc via hubert_name. "
                         "'conformer': from-scratch Conformer, no pretrained weights. "
                         "'w2vbert': pretrained wav2vec2-BERT 2.0 (mel-input, different "
                         "pipeline) -- see model_w2vbert.py for details.")
    p.add_argument("--w2vbert_name", default="facebook/w2v-bert-2.0")
    p.add_argument("--conformer_layers", type=int, default=12)
    p.add_argument("--conformer_dmodel", type=int, default=256)
    p.add_argument("--hubert_name", default="facebook/hubert-base-ls960")
    p.add_argument("--semantic_layer", type=int, default=8,
                    help="Index into HuBERT hidden_states to read out (1..num_layers)")
    p.add_argument("--vocab_type", choices=["char", "bpe"], default="char",
                    help="CTC target granularity: char-level (vocab.CTCVocab) or "
                         "subword BPE (vocab_bpe.BPECTCVocab, trained from train.jsonl)")
    p.add_argument("--bpe_vocab_size", type=int, default=500,
                    help="Target BPE vocab size, only used when --vocab_type bpe")
    p.add_argument("--layer_fusion", choices=["single", "weighted_sum"], default="single",
                    help="'single': use hidden_states[semantic_layer] only. "
                         "'weighted_sum': learned SUPERB-style softmax-weighted sum "
                         "over --fusion_layers, with separate weights for the intent "
                         "and slot branches.")
    p.add_argument("--fusion_layers", type=str, default=None,
                    help="Comma-separated hidden_states indices to fuse over, e.g. "
                         "'4,6,8,10,12'. Only used when --layer_fusion weighted_sum. "
                         "Default: every Transformer block output (1..num_hidden_layers).")
    p.add_argument("--freeze_feature_extractor", action="store_true", default=True)
    p.add_argument("--freeze_encoder_layers", type=int, default=0,
                    help="Freeze the first N Transformer blocks (<= semantic_layer makes sense)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--ctc_loss_weight", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_audio_seconds", type=float, default=15.0)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_or_load_vocabs(args, out_dir):
    intent_map_path = out_dir / "intent_labels.json"
    train_jsonl = Path(args.slurp_jsonl_dir) / "train.jsonl"

    # Needed for exporting predictions in the OFFICIAL SLURP scorer's format
    # (scenario/action as separate fields) -- see export_official_predictions.py
    scenario_action_path = out_dir / "intent_scenario_action.json"
    if not scenario_action_path.exists():
        from vocab import build_intent_scenario_action_map
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping = build_intent_scenario_action_map([train_jsonl])
        scenario_action_path.write_text(json.dumps(mapping, indent=2))

    if args.vocab_type == "bpe":
        from vocab_bpe import BPECTCVocab, build_bpe_vocab_from_jsonl
        ctc_vocab_path = out_dir / "bpe_tokenizer.json"
        if intent_map_path.exists() and ctc_vocab_path.exists():
            intent_list = json.loads(intent_map_path.read_text())
            ctc_vocab = BPECTCVocab.load(ctc_vocab_path)
        else:
            intent_list, _ = build_label_maps([train_jsonl])
            out_dir.mkdir(parents=True, exist_ok=True)
            intent_map_path.write_text(json.dumps(intent_list, indent=2))
            ctc_vocab = build_bpe_vocab_from_jsonl(
                [train_jsonl], vocab_size=args.bpe_vocab_size, save_path=ctc_vocab_path
            )
    else:
        ctc_vocab_path = out_dir / "ctc_vocab.json"
        if intent_map_path.exists() and ctc_vocab_path.exists():
            intent_list = json.loads(intent_map_path.read_text())
            ctc_vocab = CTCVocab.load(ctc_vocab_path)
        else:
            intent_list, slot_types = build_label_maps([train_jsonl])
            ctc_vocab = CTCVocab(slot_types)
            out_dir.mkdir(parents=True, exist_ok=True)
            intent_map_path.write_text(json.dumps(intent_list, indent=2))
            ctc_vocab.save(ctc_vocab_path)

    intent2id = {label: i for i, label in enumerate(intent_list)}
    return intent_list, intent2id, ctc_vocab


@torch.no_grad()
def evaluate(model, loader, device, ctc_vocab, intent_list):
    model.eval()
    pred_intents, gold_intents = [], []
    pred_texts, gold_texts = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(batch["input_values"], batch["attention_mask"])

        pred_intents.extend(outputs["intent_logits"].argmax(-1).tolist())
        gold_intents.extend(batch["intent_ids"].tolist())

        greedy_ids = outputs["ctc_log_probs"].argmax(-1).transpose(0, 1).tolist()  # (B, T)
        feat_lens = outputs["feat_lens"].tolist()
        for ids, L in zip(greedy_ids, feat_lens):
            pred_texts.append(ctc_vocab.decode(ids[:L]))
        gold_texts.extend(batch["tagged_texts"])

    acc = intent_accuracy(pred_intents, gold_intents)
    f1 = slu_f1(pred_texts, gold_texts)
    model.train()
    return {"intent_accuracy": acc, **f1}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intent_list, intent2id, ctc_vocab = build_or_load_vocabs(args, out_dir)
    print(f"[vocab] {len(intent_list)} intents, {len(ctc_vocab)} CTC symbols")

    train_ds = SlurpDataset(
        Path(args.slurp_jsonl_dir) / "train.jsonl", args.audio_root,
        ctc_vocab, intent2id, args.max_audio_seconds,
    )
    dev_ds = SlurpDataset(
        Path(args.slurp_jsonl_dir) / "devel.jsonl", args.audio_root,
        ctc_vocab, intent2id, args.max_audio_seconds,
    )
    print(f"[data] train={len(train_ds)} dev={len(dev_ds)}")

    collate = functools.partial(collate_fn, pad_id=ctc_vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)

    if args.backbone == "ssl":
        model = HubertSLUModel(
            num_intents=len(intent_list),
            ctc_vocab_size=len(ctc_vocab),
            pad_id=ctc_vocab.pad_id,
            blank_id=ctc_vocab.blank_id,
            hubert_name=args.hubert_name,
            semantic_layer=args.semantic_layer,
            layer_fusion=args.layer_fusion,
            fusion_layers=([int(x) for x in args.fusion_layers.split(",")]
                           if args.fusion_layers else None),
            freeze_feature_extractor=args.freeze_feature_extractor,
            freeze_encoder_layers=args.freeze_encoder_layers,
        )
    elif args.backbone == "w2vbert":
        model = Wav2Vec2BertSLUModel(
            num_intents=len(intent_list),
            ctc_vocab_size=len(ctc_vocab),
            pad_id=ctc_vocab.pad_id,
            blank_id=ctc_vocab.blank_id,
            model_name=args.w2vbert_name,
            freeze_encoder_layers=args.freeze_encoder_layers,
        )
    else:  # conformer
        model = ConformerSLUModel(
            num_intents=len(intent_list),
            ctc_vocab_size=len(ctc_vocab),
            pad_id=ctc_vocab.pad_id,
            blank_id=ctc_vocab.blank_id,
            num_layers=args.conformer_layers,
            d_model=args.conformer_dmodel,
        )
    model.to(args.device)

    if args.backbone == "conformer":
        # No pretrained weights to protect here -- everything (frontend,
        # Conformer stack, both heads) is randomly initialized, so a single
        # LR for the whole model is appropriate (unlike the SSL path, where
        # the encoder needs a much smaller LR than the fresh heads).
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr
        )
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=max(1, total_steps)
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[optim] conformer backbone, single lr={args.lr}, "
              f"trainable params={total_params:,}")
        print("[note] --lr defaults to a value tuned for fine-tuning a pretrained "
              "encoder (e.g. 3e-5). From scratch, that's usually too low -- "
              "try something like 3e-4 to 1e-3 for --backbone conformer.")
    else:
        if args.backbone == "w2vbert":
            encoder_params = [p for p in model.w2v_bert.parameters() if p.requires_grad]
        else:
            encoder_params = [p for p in model.hubert.parameters() if p.requires_grad]
        head_modules = [model.intent_pool, model.intent_head, model.slot_lstm, model.ctc_head]
        if getattr(args, "layer_fusion", "single") == "weighted_sum":
            head_modules += [model.intent_layer_fusion, model.slot_layer_fusion]
        head_params = [p for m in head_modules for p in m.parameters() if p.requires_grad]

        head_lr = args.lr * 20  # heads (incl. layer-fusion weights) are randomly
                                 # initialized and need a much bigger LR than the
                                 # pretrained encoder to actually move within a
                                 # normal training budget.
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": args.lr},
            {"params": head_params, "lr": head_lr},
        ])
        total_steps = args.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[args.lr, head_lr], total_steps=max(1, total_steps)
        )
        print(f"[optim] encoder lr={args.lr}, head lr={head_lr}, "
              f"encoder params={sum(p.numel() for p in encoder_params):,}, "
              f"head params={sum(p.numel() for p in head_params):,}")

    best_f1 = -1.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(batch["input_values"], batch["attention_mask"])
            loss, intent_loss, ctc_loss = model.compute_loss(
                batch, outputs, ctc_loss_weight=args.ctc_loss_weight
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=float(loss), intent=float(intent_loss), ctc=float(ctc_loss))

        metrics = evaluate(model, dev_loader, args.device, ctc_vocab, intent_list)
        print(f"[epoch {epoch}] dev intent_acc={metrics['intent_accuracy']:.4f} "
              f"slu_f1={metrics['f1']:.4f} (P={metrics['precision']:.4f} R={metrics['recall']:.4f})")

        fusion_weights = model.log_layer_fusion_weights() if hasattr(model, "log_layer_fusion_weights") else None
        if fusion_weights is not None:
            print(f"[epoch {epoch}] layer fusion weights (layers={fusion_weights['layers']})")
            print(f"    intent: {[round(w, 3) for w in fusion_weights['intent']]}")
            print(f"    slot  : {[round(w, 3) for w in fusion_weights['slot']]}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"[epoch {epoch}] new best model saved (slu_f1={best_f1:.4f})")

    torch.save(model.state_dict(), out_dir / "last_model.pt")


if __name__ == "__main__":
    main()
