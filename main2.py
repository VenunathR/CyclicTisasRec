"""
main3.py — Multi-Scale Cyclic Bias + End-to-End Fine-Tuning
=============================================================

Takes 3 periodic scales as inputs. Each scale gets its own independent
PeriodicBiasComponent (separate w_sin_K, w_cos_K, w_sin_V, w_cos_V).
Their outputs are summed before being added to abs_pos_K / abs_pos_V,
exactly at the positional encoding injection point.

Unlike main2 (frozen backbone probe), here EVERYTHING is trainable:
  • All 3 bias modules
  • The full TiSASRec backbone (embeddings, attention, FFN, LayerNorms)

This answers: "does giving the model explicit multi-scale cyclic signals
and letting it re-learn around them improve over plain TiSASRec?"

Two separate learning rates are used:
  • --backbone_lr  (small, default 1e-4) — careful fine-tuning of pretrained weights
  • --bias_lr      (larger, default 1e-3) — bias modules start from scratch

Param groups
------------
  group 0  backbone weights with weight decay  →  backbone_lr
  group 1  backbone bias/norm/emb params       →  backbone_lr, wd=0
  group 2  all bias module params              →  bias_lr,     wd=1e-4

Usage
-----
    # start from a pretrained checkpoint (recommended)
    python main3.py \
        --dataset=ml-1m --train_dir=default \
        --state_dict_path=ml-1m_default/TiSASRec.epoch=600....pth \
        --scales 7.0 30.0 90.0 \
        --device=cuda

    # train from scratch (no checkpoint)
    python main3.py \
        --dataset=ml-1m --train_dir=default \
        --scales 7.0 30.0 90.0 \
        --device=cuda

    # custom scale magnitudes (amplitude of each bias, independent of period)
    python main3.py \
        --dataset=ml-1m --train_dir=default \
        --state_dict_path=... \
        --scales 7.0 30.0 90.0 \
        --scale_mags 0.5 1.0 2.0 \
        --device=cuda
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import copy
import random

from model import TiSASRec
from utils import (
    build_relation_matrix,
    data_partition,
    compute_repos,
)


# ---------------------------------------------------------------------------
# Dataset that returns time_seq (inlined from main2 to avoid import)
# ---------------------------------------------------------------------------

class TiSASRecDatasetWithTimeSeq(torch.utils.data.Dataset):
    def __init__(self, user_train, usernum, itemnum, relation_matrix, maxlen):
        self.valid_users     = [u for u in range(1, usernum + 1) if len(user_train[u]) > 1]
        self.user_train      = user_train
        self.itemnum         = itemnum
        self.relation_matrix = relation_matrix
        self.maxlen          = maxlen

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        user   = self.valid_users[idx]
        items  = self.user_train[user]
        maxlen = self.maxlen

        seq      = np.zeros(maxlen, dtype=np.int32)
        time_seq = np.zeros(maxlen, dtype=np.int32)
        pos      = np.zeros(maxlen, dtype=np.int32)
        neg      = np.zeros(maxlen, dtype=np.int32)

        nxt  = items[-1][0]
        ptr  = maxlen - 1
        seen = {x[0] for x in items}

        for item_id, ts in reversed(items[:-1]):
            seq[ptr]      = item_id
            time_seq[ptr] = ts
            pos[ptr]      = nxt
            if nxt != 0:
                t = np.random.randint(1, self.itemnum + 1)
                while t in seen:
                    t = np.random.randint(1, self.itemnum + 1)
                neg[ptr] = t
            nxt = item_id
            ptr -= 1
            if ptr < 0:
                break

        return (
            np.array([user], dtype=np.int32),
            seq,
            time_seq,
            self.relation_matrix[user],
            pos,
            neg,
        )


# ---------------------------------------------------------------------------
# Evaluation with time_seq (inlined from main2 to avoid import)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_with_bias(model, dataset, args, split="test"):
    train, valid, test, usernum, itemnum, _ = copy.deepcopy(dataset)
    eval_set = test if split == "test" else valid

    users = (
        random.sample(range(1, usernum + 1), 10_000)
        if usernum > 10_000
        else list(range(1, usernum + 1))
    )

    device = next(p for p in model.parameters()).device
    model.eval()
    NDCG = HT = valid_user = 0.0

    for u in users:
        if len(train[u]) < 1 or len(eval_set[u]) < 1:
            continue

        seq      = np.zeros(args.maxlen, dtype=np.int32)
        time_seq = np.zeros(args.maxlen, dtype=np.int32)
        ptr      = args.maxlen - 1

        if split == "test":
            seq[ptr]      = valid[u][0][0]
            time_seq[ptr] = valid[u][0][1]
            ptr -= 1

        for item_id, ts in reversed(train[u]):
            seq[ptr]      = item_id
            time_seq[ptr] = ts
            ptr -= 1
            if ptr < 0:
                break

        target = eval_set[u][0][0]
        rated  = {x[0] for x in train[u]}
        if split == "test":
            rated.add(valid[u][0][0])
        rated.update({target, 0})

        neg_items = []
        while len(neg_items) < 100:
            t = np.random.randint(1, itemnum + 1)
            if t not in rated:
                neg_items.append(t)

        item_idx    = np.array([target] + neg_items, dtype=np.int64)
        time_matrix = compute_repos(time_seq, args.time_span)

        seq_t  = torch.tensor(seq,         dtype=torch.long).unsqueeze(0).to(device)
        ts_t   = torch.tensor(time_seq,    dtype=torch.long).unsqueeze(0).to(device)
        tm_t   = torch.tensor(time_matrix, dtype=torch.long).unsqueeze(0).to(device)
        it_t   = torch.tensor(item_idx,    dtype=torch.long).to(device)

        logits = -model.predict(seq_t, tm_t, ts_t, it_t)
        rank   = logits[0].argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1.0 / np.log2(rank + 2)
            HT   += 1.0

    return NDCG / valid_user if valid_user > 0 else 0.0, HT / valid_user if valid_user > 0 else 0.0

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single periodic bias component (one scale/period pair, own weights)
# ---------------------------------------------------------------------------

class PeriodicBiasComponent(nn.Module):
    """
    One sinusoidal bias component for a single period.

    Computes:
        bias_K(t) = scale_mag * (w_sin_K * sin(2π·t/period) + w_cos_K * cos(2π·t/period))
        bias_V(t) = scale_mag * (w_sin_V * sin(2π·t/period) + w_cos_V * cos(2π·t/period))

    where w_* ∈ ℝ^H are independent learned vectors for this component.
    scale_mag is a fixed float amplitude set at construction time.
    """

    def __init__(self, hidden_units: int, period: float, scale_mag: float) -> None:
        super().__init__()
        self.period    = period
        self.scale_mag = scale_mag

        self.w_sin_K = nn.Parameter(torch.empty(hidden_units))
        self.w_cos_K = nn.Parameter(torch.empty(hidden_units))
        self.w_sin_V = nn.Parameter(torch.empty(hidden_units))
        self.w_cos_V = nn.Parameter(torch.empty(hidden_units))

        # small init — doesn't shock pretrained backbone at start
        std = 1.0 / math.sqrt(hidden_units)
        for p in [self.w_sin_K, self.w_cos_K, self.w_sin_V, self.w_cos_V]:
            nn.init.normal_(p, std=std)

    def forward(self, time_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        time_seq : (B, T) — integer timestamps per position

        Returns
        -------
        bias_K, bias_V : (B, T, H) each
        """
        t     = time_seq.float()
        phase = 2.0 * torch.pi * t / self.period   # (B, T)
        s     = torch.sin(phase)                    # (B, T)
        c     = torch.cos(phase)                    # (B, T)

        # (B, T, 1) * (H,) → (B, T, H)
        bias_K = self.scale_mag * (
            s.unsqueeze(-1) * self.w_sin_K +
            c.unsqueeze(-1) * self.w_cos_K
        )
        bias_V = self.scale_mag * (
            s.unsqueeze(-1) * self.w_sin_V +
            c.unsqueeze(-1) * self.w_cos_V
        )
        return bias_K, bias_V


# ---------------------------------------------------------------------------
# Multi-scale cyclic bias: 3 components summed
# ---------------------------------------------------------------------------

class MultiScaleCyclicBias(nn.Module):
    """
    Holds N independent PeriodicBiasComponents (one per scale/period).
    Forward sums all their K-side and V-side outputs respectively.

    Each component has its own w_sin / w_cos, so they can learn
    different response profiles for different cycle lengths.
    """

    def __init__(
        self,
        hidden_units: int,
        periods: list[float],
        scale_mags: list[float],
    ) -> None:
        super().__init__()
        assert len(periods) == len(scale_mags), \
            "periods and scale_mags must have the same length"

        self.components = nn.ModuleList([
            PeriodicBiasComponent(hidden_units, p, s)
            for p, s in zip(periods, scale_mags)
        ])

        log.info(
            "MultiScaleCyclicBias: %d components — periods=%s  scale_mags=%s",
            len(periods), periods, scale_mags,
        )

    def forward(self, time_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns summed bias_K and bias_V across all components.
        Shape: (B, T, H) each.
        """
        bias_K = bias_V = None
        for comp in self.components:
            bk, bv = comp(time_seq)
            bias_K = bk if bias_K is None else bias_K + bk
            bias_V = bv if bias_V is None else bias_V + bv
        return bias_K, bias_V   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Full model: TiSASRec backbone + MultiScaleCyclicBias (all trainable)
# ---------------------------------------------------------------------------

class TiSASRecMultiCyclic(nn.Module):
    """
    TiSASRec + 3-component cyclic positional bias, trained end-to-end.

    The bias is injected additively into abs_pos_K / abs_pos_V before
    the first (and every subsequent) attention block — same injection
    point as in main2, but now the backbone is also free to update.
    """

    def __init__(self, backbone: TiSASRec, cyclic_bias: MultiScaleCyclicBias) -> None:
        super().__init__()
        self.backbone     = backbone
        self.cyclic_bias  = cyclic_bias

    # ------------------------------------------------------------------
    def _seq2feats(
        self,
        log_seqs: torch.Tensor,       # (B, T)
        time_matrices: torch.Tensor,  # (B, T, T)
        time_seq: torch.Tensor,       # (B, T) raw timestamps
    ) -> torch.Tensor:
        bb = self.backbone
        B, T = log_seqs.shape

        seqs = bb.item_emb(log_seqs)
        seqs = seqs * (bb.item_emb.embedding_dim ** 0.5)
        seqs = bb.emb_dropout(seqs)

        positions = torch.arange(T, device=log_seqs.device).unsqueeze(0).expand(B, -1)
        abs_K = bb.pos_K_dropout(bb.abs_pos_K_emb(positions))   # (B, T, H)
        abs_V = bb.pos_V_dropout(bb.abs_pos_V_emb(positions))   # (B, T, H)

        # ── inject multi-scale cyclic bias ────────────────────────────
        cyc_K, cyc_V = self.cyclic_bias(time_seq)   # (B, T, H) each
        abs_K = abs_K + cyc_K
        abs_V = abs_V + cyc_V
        # ──────────────────────────────────────────────────────────────

        t_K = bb.time_K_dropout(bb.time_matrix_K_emb(time_matrices))
        t_V = bb.time_V_dropout(bb.time_matrix_V_emb(time_matrices))

        timeline_mask = (log_seqs == 0)
        seqs = seqs * (~timeline_mask.unsqueeze(-1))

        causal_mask = ~torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=log_seqs.device)
        )

        for attn_ln, attn_layer, fwd_ln, fwd_layer in zip(
            bb.attn_layernorms, bb.attn_layers,
            bb.fwd_layernorms,  bb.fwd_layers,
        ):
            Q       = attn_ln(seqs)
            mha_out = attn_layer(
                Q, seqs,
                timeline_mask, causal_mask,
                t_K, t_V, abs_K, abs_V,
            )
            seqs = Q + mha_out
            seqs = fwd_ln(seqs)
            seqs = fwd_layer(seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        return bb.last_layernorm(seqs)   # (B, T, H)

    # ------------------------------------------------------------------
    def forward(
        self,
        log_seqs: torch.Tensor,
        time_matrices: torch.Tensor,
        time_seq: torch.Tensor,
        pos_seqs: torch.Tensor,
        neg_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_feats  = self._seq2feats(log_seqs, time_matrices, time_seq)
        pos_embs   = self.backbone.item_emb(pos_seqs)
        neg_embs   = self.backbone.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        log_seqs: torch.Tensor,
        time_matrices: torch.Tensor,
        time_seq: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        log_feats  = self._seq2feats(log_seqs, time_matrices, time_seq)
        final_feat = log_feats[:, -1, :]
        item_embs  = self.backbone.item_emb(item_indices)
        if item_embs.dim() == 2:
            return final_feat @ item_embs.T
        return torch.bmm(item_embs, final_feat.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Param groups: two LRs — backbone (small) vs bias (larger)
# ---------------------------------------------------------------------------

def build_param_groups(
    model: TiSASRecMultiCyclic,
    backbone_lr: float,
    bias_lr: float,
    weight_decay: float,
) -> list[dict]:
    """
    Three param groups:
      0 — backbone decay params        → backbone_lr, wd=weight_decay
      1 — backbone no-decay params     → backbone_lr, wd=0
      2 — cyclic bias params           → bias_lr,     wd=1e-4

    Handles both compiled and non-compiled models via getattr.
    """
    _no_decay_keys = ("bias", "layernorm", "layer_norm", "emb")

    # Get the actual model (handles torch.compile wrapper)
    actual_model = getattr(model, "_orig_mod", model)

    bb_decay, bb_no_decay = [], []
    for name, param in actual_model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name.lower() for k in _no_decay_keys):
            bb_no_decay.append(param)
        else:
            bb_decay.append(param)

    bias_params = list(actual_model.cyclic_bias.parameters())

    log.info(
        "Param groups | bb_decay=%d  bb_no_decay=%d  bias=%d",
        len(bb_decay), len(bb_no_decay), len(bias_params),
    )

    return [
        {"params": bb_decay,    "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": bb_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": bias_params, "lr": bias_lr,     "weight_decay": 1e-4},
    ]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TiSASRec + 3-scale cyclic bias, end-to-end fine-tune"
    )

    # ── shared model/data args ───────────────────────────────────────
    p.add_argument("--dataset",      required=True)
    p.add_argument("--train_dir",    required=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--batch_size",   default=256,  type=int)
    p.add_argument("--maxlen",       default=50,   type=int)
    p.add_argument("--hidden_units", default=50,   type=int)
    p.add_argument("--num_blocks",   default=2,    type=int)
    p.add_argument("--num_heads",    default=1,    type=int)
    p.add_argument("--time_span",    default=256,  type=int)
    p.add_argument("--dropout_rate", default=0.2,  type=float)
    p.add_argument("--l2_emb",       default=5e-5, type=float)
    p.add_argument("--num_workers",  default=4,    type=int)
    p.add_argument("--bf16",         action="store_true", default=True)

    # ── checkpoint (optional — trains from scratch if omitted) ───────
    p.add_argument("--state_dict_path", default=None,
                   help="Pretrained TiSASRec .pth — highly recommended")

    # ── 3 periodic scales (period lengths) ───────────────────────────
    p.add_argument("--scales", nargs=3, type=float,
                   default=[7.0, 30.0, 90.0],
                   metavar=("SCALE1", "SCALE2", "SCALE3"),
                   help="3 period lengths for the cyclic bias components "
                        "(in preprocessed timestamp units, e.g. 7 14 30)")

    # ── amplitude of each bias component ─────────────────────────────
    p.add_argument("--scale_mags", nargs=3, type=float,
                   default=[1.0, 1.0, 1.0],
                   metavar=("MAG1", "MAG2", "MAG3"),
                   help="Fixed amplitude multiplier for each of the 3 bias "
                        "components (independent of learned weights)")

    # ── training ─────────────────────────────────────────────────────
    p.add_argument("--num_epochs",   default=200,  type=int)
    p.add_argument("--backbone_lr",  default=1e-4, type=float,
                   help="LR for the pretrained TiSASRec backbone (keep small)")
    p.add_argument("--bias_lr",      default=1e-3, type=float,
                   help="LR for the cyclic bias modules (can be larger)")
    p.add_argument("--eval_every",   default=20,   type=int)
    p.add_argument("--compile",      action="store_true",
                   help="torch.compile() — recommended for long runs on RTX 4090")

    return p.parse_args()


# ---------------------------------------------------------------------------
# main3
# ---------------------------------------------------------------------------

def main3() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        log.info("CUDA: %s", torch.cuda.get_device_name(0))

    # ── output dir ────────────────────────────────────────────────────
    scales_tag = "_".join(str(s) for s in args.scales)
    out_dir    = Path(f"{args.dataset}_{args.train_dir}_cyclic{scales_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.txt").open("w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k},{v}\n")

    # ── data ──────────────────────────────────────────────────────────
    (user_train, user_valid, user_test,
     usernum, itemnum, timenum) = data_partition(args.dataset)
    dataset = (user_train, user_valid, user_test, usernum, itemnum, timenum)
    log.info("Users=%d  Items=%d  TimeSpan=%d", usernum, itemnum, timenum)

    # ── relation matrix ───────────────────────────────────────────────
    cache = Path("data") / f"rel_{args.dataset}_{args.maxlen}_{args.time_span}.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            relation_matrix = pickle.load(f)
        log.info("Loaded relation matrix from cache")
    else:
        relation_matrix = build_relation_matrix(
            user_train, usernum, args.maxlen, args.time_span
        )
        with cache.open("wb") as f:
            pickle.dump(relation_matrix, f, protocol=4)

    # ── build model ───────────────────────────────────────────────────
    backbone = TiSASRec(usernum, itemnum, timenum, args).to(device)

    if args.state_dict_path:
        sd = torch.load(args.state_dict_path, map_location=device, weights_only=True)
        backbone.load_state_dict(sd)
        log.info("Loaded pretrained backbone from %s", args.state_dict_path)
    else:
        log.info("No checkpoint given — training backbone from scratch")

    cyclic_bias = MultiScaleCyclicBias(
        hidden_units=args.hidden_units,
        periods=args.scales,
        scale_mags=args.scale_mags,
    ).to(device)

    model = TiSASRecMultiCyclic(backbone, cyclic_bias).to(device)

    # log param counts
    total     = sum(p.numel() for p in model.parameters())
    bias_only = sum(p.numel() for p in cyclic_bias.parameters())
    log.info(
        "Total params: %d  |  Cyclic bias params: %d (%.2f%%)",
        total, bias_only, 100.0 * bias_only / total,
    )

    if args.compile:
        log.info("Compiling with torch.compile() ...")
        model = torch.compile(model)  # type: ignore[assignment]

    # ── optimiser + scheduler ─────────────────────────────────────────
    param_groups = build_param_groups(
        model,
        backbone_lr=args.backbone_lr,
        bias_lr=args.bias_lr,
        weight_decay=args.l2_emb,
    )
    optimiser = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.num_epochs, eta_min=1e-6)

    # ── AMP ───────────────────────────────────────────────────────────
    use_amp   = args.bf16 and args.device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    scaler    = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # ── dataloader ────────────────────────────────────────────────────
    ds = TiSASRecDatasetWithTimeSeq(
        user_train, usernum, itemnum, relation_matrix, args.maxlen
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # ── training loop ─────────────────────────────────────────────────
    bce      = nn.BCEWithLogitsLoss()
    best_ndcg = 0.0
    t0        = time.perf_counter()

    log.info(
        "Starting training — %d epochs | scales=%s | mags=%s | "
        "backbone_lr=%.1e | bias_lr=%.1e",
        args.num_epochs, args.scales, args.scale_mags,
        args.backbone_lr, args.bias_lr,
    )

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = epoch_auc = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}", ncols=80, leave=False):
            _, seq, time_seq_np, time_mat, pos, neg = batch

            seq      = seq.to(device, non_blocking=True)
            time_seq = time_seq_np.to(device, non_blocking=True)
            time_mat = time_mat.to(device, non_blocking=True)
            pos      = pos.to(device, non_blocking=True)
            neg      = neg.to(device, non_blocking=True)

            with torch.autocast(device_type=args.device, dtype=amp_dtype, enabled=use_amp):
                pos_logits, neg_logits = model(seq, time_mat, time_seq, pos, neg)

                is_target = (pos != 0).float()
                loss = (
                    bce(pos_logits, torch.ones_like(pos_logits))  * is_target +
                    bce(neg_logits, torch.zeros_like(neg_logits)) * is_target
                ).sum() / (is_target.sum() + 1e-8)

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            with torch.no_grad():
                auc = (
                    ((pos_logits - neg_logits).sign() + 1) / 2 * is_target
                ).sum() / (is_target.sum() + 1e-8)

            epoch_loss += loss.item()
            epoch_auc  += auc.item()

        scheduler.step()

        # ── per-component bias norm (shows how much each cycle learned) ──
        raw = getattr(model, "_orig_mod", model)
        bias_norms = [
            comp.w_sin_K.norm().item()
            for comp in raw.cyclic_bias.components
        ]

        if epoch % args.eval_every == 0:
            elapsed = time.perf_counter() - t0
            t0      = time.perf_counter()

            model.eval()
            v_ndcg, v_hr = evaluate_with_bias(model, dataset, args, split="valid")
            t_ndcg, t_hr = evaluate_with_bias(model, dataset, args, split="test")

            log.info(
                "Epoch %3d | %.1fs | loss=%.4f | auc=%.4f | "
                "valid NDCG@10=%.4f HR@10=%.4f | test NDCG@10=%.4f HR@10=%.4f | "
                "bias_norms(K) %s",
                epoch, elapsed,
                epoch_loss / len(loader),
                epoch_auc  / len(loader),
                v_ndcg, v_hr,
                t_ndcg, t_hr,
                [f"{n:.3f}" for n in bias_norms],
            )

            if t_ndcg > best_ndcg:
                best_ndcg = t_ndcg
                raw_model = getattr(model, "_orig_mod", model)
                ckpt = (
                    out_dir
                    / f"TiSASRec_cyclic"
                      f".epoch={epoch}"
                      f".scales={'_'.join(str(s) for s in args.scales)}"
                      f".ndcg={t_ndcg:.4f}.pth"
                )
                torch.save(
                    {
                        "backbone":    raw_model.backbone.state_dict(),
                        "cyclic_bias": raw_model.cyclic_bias.state_dict(),
                        "epoch":       epoch,
                        "scales":      args.scales,
                        "scale_mags":  args.scale_mags,
                        "test_ndcg":   t_ndcg,
                        "test_hr":     t_hr,
                    },
                    ckpt,
                )
                log.info("★ New best! Checkpoint → %s", ckpt)

    log.info("Done. Best test NDCG@10: %.4f", best_ndcg)


if __name__ == "__main__":
    main3()
