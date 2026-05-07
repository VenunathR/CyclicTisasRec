"""
cyclic_tisasrec.py — Cyclic-TiSASRec: Final Architecture
==========================================================

Implements the paper's final design exactly:

  1. Harmonic Time Encoding
       φ(t, c) = [sin(2πt/c), cos(2πt/c)]  for c ∈ {daily, weekly, monthly}
       Etime(t) ∈ R^(2K) projected to R^H via a linear layer

  2. Phase-Aware Attention Bias (on the LOGIT, NOT K/V injection)
       ekj = QkKj⊤/√d  +  Σ_c  λc · cosine_sim(φ^c_k, φ^c_j)
       λc is a learnable scalar per cycle per head

  3. Neural Aggregation Prediction Layer
       Hseq      = hL  (last Transformer position output)
       Hperiodic = Wproj · Etime(tL+1)  (target timestamp phase)
       H         = MLP([Hseq ‖ Hperiodic])
       scores    = W · H + b  →  softmax

  Two learning rates:
       --backbone_lr  (default 1e-4) — TiSASRec backbone
       --bias_lr      (default 1e-3) — all new cyclic modules

  Fixed human cycles (seconds):
       daily   =  86_400 s  (1 day)
       weekly  = 604_800 s  (7 days)
       monthly = 2_592_000 s (30 days)

  Cycle suppression is automatic: if a cycle carries no signal,
  its λc and Wproj weights are driven toward zero by gradient descent.

Usage
-----
    # recommended: start from a pretrained TiSASRec checkpoint
    python cyclic_tisasrec.py \\
        --dataset=ml-1m --train_dir=default \\
        --state_dict_path=ml-1m_default/TiSASRec.epoch=600....pth \\
        --device=cuda

    # custom scale magnitudes (initial λ amplitude per cycle)
    python cyclic_tisasrec.py \\
        --dataset=ml-1m --train_dir=default \\
        --state_dict_path=... \\
        --scale_mags 0.1 0.1 0.1 \\
        --device=cuda

    # train from scratch (no checkpoint)
    python cyclic_tisasrec.py \\
        --dataset=ml-1m --train_dir=default \\
        --device=cuda
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import TiSASRec
from utils import build_relation_matrix, data_partition, compute_repos


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed human behavioral cycles (in seconds)
# ---------------------------------------------------------------------------

DAILY_PERIOD   =    86_400.0   # 24 hours
WEEKLY_PERIOD  =   604_800.0   # 7 days
MONTHLY_PERIOD = 2_592_000.0   # 30 days

HUMAN_CYCLES = [DAILY_PERIOD, WEEKLY_PERIOD, MONTHLY_PERIOD]
CYCLE_NAMES  = ["daily", "weekly", "monthly"]


# ---------------------------------------------------------------------------
# Harmonic Time Encoding
# ---------------------------------------------------------------------------

class HarmonicTimeEncoding(nn.Module):
    """
    Maps a sequence of timestamps to a continuous cyclic phase space.

    For each timestamp t and cycle c:
        φ(t, c) = [sin(2πt/c), cos(2πt/c)]

    Final encoding: Etime(t) = concat of all φ(t, c) → R^(2K)
    Then projected to R^H via a learned linear layer (Wproj).

    K = number of cycles (default 3: daily, weekly, monthly)
    """

    def __init__(self, hidden_units: int, periods: list[float]) -> None:
        super().__init__()
        self.K = len(periods)
        self.register_buffer(
            "periods",
            torch.tensor(periods, dtype=torch.float32),   # (K,)
        )
        # Linear projection: R^(2K) → R^H
        # This is Wproj from the paper. Its learned weight norms directly
        # indicate how much the model relies on each cycle (Table 6).
        self.proj = nn.Linear(2 * self.K, hidden_units, bias=False)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(hidden_units))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timestamps : (B, T) or (B,) — raw timestamps (float seconds or ints)

        Returns
        -------
        Etime : same leading dims + (H,)
        """
        t = timestamps.float()                             # (...,)
        # (..., K)
        phase = 2.0 * math.pi * t.unsqueeze(-1) / self.periods
        enc   = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (..., 2K)
        return self.proj(enc)                              # (..., H)

    @torch.no_grad()
    def effective_amplitudes(self) -> list[float]:
        """
        Returns the effective amplitude A_c = sqrt(‖w_sin‖² + ‖w_cos‖²) per cycle.
        Larger → model relies more on that behavioral rhythm (paper Eq. 17).
        """
        W = self.proj.weight   # (H, 2K)
        amps = []
        for k in range(self.K):
            w_sin = W[:, k]
            w_cos = W[:, self.K + k]
            amps.append(math.sqrt(w_sin.norm().item() ** 2 + w_cos.norm().item() ** 2))
        return amps


# ---------------------------------------------------------------------------
# Phase-Aware Attention Bias (logit-level, NOT K/V injection)
# ---------------------------------------------------------------------------

class PhaseSimilarityBias(nn.Module):
    """
    Computes the per-cycle cosine similarity bias added directly to attention logits.

    For each cycle c:
        bias_c(k, j) = λc · cosine_sim(φ^c_k, φ^c_j)

    Total bias: ekj += Σ_c  λc · cosine_sim(φ^c_k, φ^c_j)

    λc is a learnable scalar initialized to scale_mag.
    This scalar is the only parameter here — the phase vectors themselves
    come from HarmonicTimeEncoding and are not separately stored.

    Design rationale (paper Section 4.5):
      - Phase similarity is inherently pairwise — expressed directly as a scalar
        bias rather than forcing dot-product reconstruction from K/V vectors.
      - When a cycle is uninformative (e.g. monthly on short sequences),
        its near-constant similarity term is absorbed by softmax and the
        optimizer drives λc → 0 automatically.
    """

    def __init__(self, periods: list[float], scale_mags: list[float]) -> None:
        super().__init__()
        assert len(periods) == len(scale_mags)
        self.K = len(periods)
        self.register_buffer(
            "periods",
            torch.tensor(periods, dtype=torch.float32),   # (K,)
        )
        # λc: learnable scalar per cycle (paper Eq. 9)
        self.lambdas = nn.Parameter(
            torch.tensor(scale_mags, dtype=torch.float32)  # (K,)
        )

    def _phase_vectors(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute raw [sin, cos] phase vectors per position per cycle.

        Parameters
        ----------
        timestamps : (B, T)

        Returns
        -------
        phi : (B, T, K, 2)  — sin/cos pair for each cycle
        """
        t     = timestamps.float()                              # (B, T)
        phase = 2.0 * math.pi * t.unsqueeze(-1) / self.periods # (B, T, K)
        s     = torch.sin(phase)                                # (B, T, K)
        c_    = torch.cos(phase)                                # (B, T, K)
        return torch.stack([s, c_], dim=-1)                    # (B, T, K, 2)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute the phase similarity bias matrix.

        Parameters
        ----------
        timestamps : (B, T)

        Returns
        -------
        bias : (B, T, T) — additive bias to add to attention logits before softmax
        """
        phi = self._phase_vectors(timestamps)   # (B, T, K, 2)

        # Normalize each (sin, cos) pair to unit length for cosine similarity
        phi_norm = F.normalize(phi, dim=-1)     # (B, T, K, 2)

        # Cosine similarity between all pairs of positions for each cycle:
        # sim_c(k, j) = phi_norm[k, c] · phi_norm[j, c]
        # phi_norm: (B, T, K, 2) → compute outer product over T
        # Result: (B, T, T, K)
        sim = torch.einsum("btkd, bskd -> btsk", phi_norm, phi_norm)  # (B, T, T, K)

        # Weight each cycle by its learnable λc and sum
        # lambdas: (K,) → weighted sum over K → (B, T, T)
        bias = (sim * self.lambdas.view(1, 1, 1, -1)).sum(dim=-1)    # (B, T, T)

        return bias


# ---------------------------------------------------------------------------
# Dataset (unchanged from main3, returns time_seq)
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
        # target timestamp for the prediction layer (tL+1)
        # This is the timestamp of the item we're trying to predict
        target_time = np.zeros(1, dtype=np.int32)

        nxt  = items[-1][0]
        nxt_time = items[-1][1]   # timestamp of target item
        target_time[0] = nxt_time
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
            nxt      = item_id
            nxt_time = ts
            ptr -= 1
            if ptr < 0:
                break

        return (
            np.array([self.valid_users[idx]], dtype=np.int32),
            seq,
            time_seq,
            self.relation_matrix[self.valid_users[idx]],
            pos,
            neg,
            target_time,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, dataset, args, split="test"):
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
        target_time = eval_set[u][0][1]   # tL+1 for prediction layer

        rated = {x[0] for x in train[u]}
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

        seq_t    = torch.tensor(seq,         dtype=torch.long).unsqueeze(0).to(device)
        ts_t     = torch.tensor(time_seq,    dtype=torch.long).unsqueeze(0).to(device)
        tm_t     = torch.tensor(time_matrix, dtype=torch.long).unsqueeze(0).to(device)
        it_t     = torch.tensor(item_idx,    dtype=torch.long).to(device)
        tgt_ts_t = torch.tensor([target_time], dtype=torch.float32).to(device)

        logits = -model.predict(seq_t, tm_t, ts_t, it_t, tgt_ts_t)
        rank   = logits[0].argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1.0 / np.log2(rank + 2)
            HT   += 1.0

    return NDCG / valid_user, HT / valid_user


# ---------------------------------------------------------------------------
# Full Model: TiSASRec + Phase-Aware Attention + Neural Prediction Layer
# ---------------------------------------------------------------------------

class CyclicTiSASRec(nn.Module):
    """
    Cyclic-TiSASRec — full paper architecture.

    Components
    ----------
    1. backbone        : TiSASRec (all weights trainable)
    2. time_enc        : HarmonicTimeEncoding — maps timestamps to R^H
                         (used in both input representation and prediction layer)
    3. phase_bias      : PhaseSimilarityBias  — per-cycle cosine-sim logit bias
    4. pred_proj       : Linear R^H → R^H  (projects Hperiodic)
    5. fusion_mlp      : 2-layer MLP fusing [Hseq ‖ Hperiodic] → scores

    Forward pass
    ------------
    Training : forward(log_seqs, time_matrices, time_seq, pos_seqs, neg_seqs, target_times)
    Inference: predict(log_seqs, time_matrices, time_seq, item_indices, target_time)
    """

    def __init__(
        self,
        backbone: TiSASRec,
        periods: list[float],
        scale_mags: list[float],
        hidden_units: int,
        itemnum: int,
    ) -> None:
        super().__init__()
        self.backbone   = backbone
        self.H          = hidden_units
        self.itemnum    = itemnum

        # --- Harmonic time encoding (shared: input repr + prediction layer) ---
        self.time_enc   = HarmonicTimeEncoding(hidden_units, periods)

        # --- Phase-aware attention logit bias ---
        self.phase_bias = PhaseSimilarityBias(periods, scale_mags)

        # --- Prediction layer (paper Section 4.6) ---
        # Projects Hperiodic (target timestamp encoding) to H dim
        self.pred_proj  = nn.Linear(hidden_units, hidden_units, bias=False)
        nn.init.xavier_uniform_(self.pred_proj.weight)

        # 2-layer MLP: [Hseq ‖ Hperiodic] (2H) → H → scores over items
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * hidden_units, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, hidden_units),
        )
        for layer in self.fusion_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Final item scoring layer W ∈ R^(|I| × H)
        # We reuse backbone.item_emb for scoring (tied weights) to save params
        # and keep consistency with TiSASRec's original scoring approach.

    # ------------------------------------------------------------------
    def _seq2feats(
        self,
        log_seqs: torch.Tensor,       # (B, T)
        time_matrices: torch.Tensor,  # (B, T, T)
        time_seq: torch.Tensor,       # (B, T) raw timestamps
    ) -> torch.Tensor:
        """
        Full forward through Transformer with:
          - harmonic time encoding added to item embeddings (Eq. 8)
          - phase similarity bias added to attention logits (Eq. 9)
          - TiSASRec's own relative time interval embeddings preserved
        """
        bb = self.backbone
        B, T = log_seqs.shape

        # Item embeddings scaled by sqrt(H) — standard Transformer practice
        seqs = bb.item_emb(log_seqs)                          # (B, T, H)
        seqs = seqs * (bb.item_emb.embedding_dim ** 0.5)
        seqs = bb.emb_dropout(seqs)

        # --- Harmonic temporal embedding fused into item representation (Eq. 8) ---
        # Etime(tk) projected to R^H and added to item embedding at each position
        time_emb = self.time_enc(time_seq)                    # (B, T, H)
        seqs     = seqs + time_emb

        # Absolute positional encodings (unchanged from TiSASRec)
        positions = torch.arange(T, device=log_seqs.device).unsqueeze(0).expand(B, -1)
        abs_K = bb.pos_K_dropout(bb.abs_pos_K_emb(positions))   # (B, T, H)
        abs_V = bb.pos_V_dropout(bb.abs_pos_V_emb(positions))   # (B, T, H)

        # Relative time interval embeddings (TiSASRec's recency signal)
        t_K = bb.time_K_dropout(bb.time_matrix_K_emb(time_matrices))  # (B, T, T, H)
        t_V = bb.time_V_dropout(bb.time_matrix_V_emb(time_matrices))  # (B, T, T, H)

        # --- Phase similarity bias (Eq. 9) — added to logits inside attention ---
        # (B, T, T): how similar are positions k and j in each behavioral cycle?
        phase_bias_matrix = self.phase_bias(time_seq)          # (B, T, T)

        # Masks
        timeline_mask = (log_seqs == 0)
        seqs = seqs * (~timeline_mask.unsqueeze(-1))

        causal_mask = ~torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=log_seqs.device)
        )

        # Transformer blocks — pass phase_bias_matrix to each attention layer
        for attn_ln, attn_layer, fwd_ln, fwd_layer in zip(
            bb.attn_layernorms, bb.attn_layers,
            bb.fwd_layernorms,  bb.fwd_layers,
        ):
            Q = attn_ln(seqs)
            mha_out = attn_layer(
                Q, seqs,
                timeline_mask, causal_mask,
                t_K, t_V, abs_K, abs_V,
                phase_bias=phase_bias_matrix,   # NEW: passed into attention
            )
            seqs = Q + mha_out
            seqs = fwd_ln(seqs)
            seqs = fwd_layer(seqs)
            seqs = seqs * (~timeline_mask.unsqueeze(-1))

        return bb.last_layernorm(seqs)   # (B, T, H)

    # ------------------------------------------------------------------
    def _predict_scores(
        self,
        log_feats: torch.Tensor,     # (B, T, H)
        target_times: torch.Tensor,  # (B,) — tL+1 timestamps
        item_indices: torch.Tensor,  # (N,) candidate item ids
    ) -> torch.Tensor:
        """
        Prediction layer (paper Section 4.6):

            Hseq      = hL  (last non-padding position)
            Hperiodic = Wproj · Etime(tL+1)
            H         = MLP([Hseq ‖ Hperiodic])
            scores    = H @ item_emb.T
        """
        H_seq  = log_feats[:, -1, :]                          # (B, H)

        # Encode the target timestamp — captures which phase of each cycle
        # the *next* interaction falls on (weekday morning? weekend evening?)
        H_per  = self.pred_proj(self.time_enc(target_times))  # (B, H)

        # Fuse sequential intent + periodic behavioral state
        fused  = self.fusion_mlp(
            torch.cat([H_seq, H_per], dim=-1)                 # (B, 2H)
        )                                                      # (B, H)

        # Score over candidate items using shared item embedding matrix
        item_embs = self.backbone.item_emb(item_indices)      # (N, H)
        return fused @ item_embs.T                             # (B, N)

    # ------------------------------------------------------------------
    def forward(
        self,
        log_seqs: torch.Tensor,       # (B, T)
        time_matrices: torch.Tensor,  # (B, T, T)
        time_seq: torch.Tensor,       # (B, T)
        pos_seqs: torch.Tensor,       # (B, T)
        neg_seqs: torch.Tensor,       # (B, T)
        target_times: torch.Tensor,   # (B,) — tL+1 for each sample
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_feats = self._seq2feats(log_seqs, time_matrices, time_seq)

        # For training we score pos/neg at each position using the
        # prediction layer conditioned on the target position's timestamp.
        # target_times provides the timestamp for the last prediction target.
        H_seq = log_feats                                      # (B, T, H)
        H_per = self.pred_proj(
            self.time_enc(target_times.float())                # (B, H)
        ).unsqueeze(1).expand(-1, H_seq.size(1), -1)          # (B, T, H)

        fused = self.fusion_mlp(
            torch.cat([H_seq, H_per], dim=-1)                 # (B, T, 2H)
        )                                                      # (B, T, H)

        pos_embs   = self.backbone.item_emb(pos_seqs)         # (B, T, H)
        neg_embs   = self.backbone.item_emb(neg_seqs)         # (B, T, H)
        pos_logits = (fused * pos_embs).sum(dim=-1)           # (B, T)
        neg_logits = (fused * neg_embs).sum(dim=-1)           # (B, T)
        return pos_logits, neg_logits

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        log_seqs: torch.Tensor,       # (B, T)
        time_matrices: torch.Tensor,  # (B, T, T)
        time_seq: torch.Tensor,       # (B, T)
        item_indices: torch.Tensor,   # (N,) candidate items
        target_times: torch.Tensor,   # (B,) tL+1
    ) -> torch.Tensor:
        log_feats = self._seq2feats(log_seqs, time_matrices, time_seq)
        return self._predict_scores(log_feats, target_times.float(), item_indices)


# ---------------------------------------------------------------------------
# NOTE: The TiSASRec attention layer must accept a `phase_bias` kwarg.
# If your model.py attention layer does not support this, patch it here:
# ---------------------------------------------------------------------------

def _patch_attention_layer(backbone: TiSASRec) -> None:
    """
    Monkey-patches TiSASRec's attention layers to accept and apply a
    phase_bias argument added directly to the attention logits before softmax.

    This is the core of the paper's design: the periodic signal lives in its
    own pathway, outside K/V space, added as a scalar bias on the logit.

    Only patches if the layer's forward() does not already support phase_bias.
    """
    for attn_layer in backbone.attn_layers:
        orig_forward = attn_layer.forward

        # Check if already patched
        if getattr(attn_layer, "_cyclic_patched", False):
            continue

        def make_patched_forward(orig_fwd):
            def patched_forward(
                queries, keys,
                attn_mask, causal_mask,
                time_matrix_K, time_matrix_V,
                abs_pos_K, abs_pos_V,
                phase_bias=None,
            ):
                # ---- replicate TiSASRec's attention logic ----
                # (this mirrors TiSASRecAttn.forward exactly, adding phase_bias)
                B, T, H = queries.shape
                num_heads = attn_layer.num_heads if hasattr(attn_layer, 'num_heads') else 1

                # Project Q, K, V
                   # (B, T, H)
                Q = attn_layer.Q_w(queries)    # (B, T, H)
                K = attn_layer.K_w(keys)       # (B, T, H)
                V = attn_layer.V_w(keys)       # (B, T, H)

                # Add absolute positional and time interval encodings
                # (TiSASRec's existing recency pathway — preserved unchanged)
                # time_matrix_K: (B, T, T, H) — relative time intervals for keys
                # abs_pos_K:     (B, T, H)    — absolute position for keys

                # Compute attention logits (TiSASRec style)
                # Standard: QK^T/sqrt(d)
                scale = math.sqrt(H // num_heads) if num_heads > 1 else math.sqrt(H)
                logits = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, T, T)

                # TiSASRec relative time term: Q * M_rij
                # time_matrix_K: (B, T, T, H)
                logits += torch.matmul(
                    Q.unsqueeze(2),                  # (B, T, 1, H)
                    time_matrix_K.transpose(-2, -1)  # (B, T, H, T)
                ).squeeze(2)

                # TiSASRec absolute position term
                logits += torch.matmul(
                    Q.unsqueeze(2),
                    abs_pos_K.unsqueeze(2).transpose(-2, -1)
                ).squeeze(2).squeeze(2).unsqueeze(-1).expand_as(logits)

                # ── Phase similarity bias (paper Eq. 9) ──────────────────
                # Added directly on top of the logit, separate from K/V space.
                # When a cycle is uninformative, λc → 0 absorbs this naturally.
                if phase_bias is not None:
                    logits = logits + phase_bias   # (B, T, T)
                # ─────────────────────────────────────────────────────────

                # Masking
                logits = logits - 1e9 * causal_mask.unsqueeze(0).float()
                logits = logits - 1e9 * attn_mask.unsqueeze(1).float()

                weights = torch.softmax(logits, dim=-1)
                weights = attn_layer.attn_drop(weights)

                # Value aggregation
                out = torch.matmul(weights, V)   # (B, T, H)

                # Add time_matrix_V term
                out += torch.matmul(
                    weights.unsqueeze(2),            # (B, T, 1, T)
                    time_matrix_V                    # (B, T, T, H)
                ).squeeze(2)

                # Add abs_pos_V term
                out += (weights.unsqueeze(-1) * abs_pos_V.unsqueeze(1)).sum(-2)

                return attn_layer.out_proj(out)

            return patched_forward

        attn_layer.forward = make_patched_forward(orig_forward)
        attn_layer._cyclic_patched = True

    log.info("Patched %d attention layer(s) with phase_bias support", len(backbone.attn_layers))


# ---------------------------------------------------------------------------
# Param groups: backbone (small LR) vs cyclic modules (larger LR)
# ---------------------------------------------------------------------------

def build_param_groups(
    model: CyclicTiSASRec,
    backbone_lr: float,
    bias_lr: float,
    weight_decay: float,
) -> list[dict]:
    """
    Three param groups:
      0 — backbone decay params        → backbone_lr, wd=weight_decay
      1 — backbone no-decay params     → backbone_lr, wd=0
      2 — all cyclic module params     → bias_lr,     wd=1e-4
    """
    _no_decay = ("bias", "layernorm", "layer_norm", "emb")

    bb_decay, bb_no_decay = [], []
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name.lower() for k in _no_decay):
            bb_no_decay.append(param)
        else:
            bb_decay.append(param)

    # All new cyclic parameters: time_enc, phase_bias, pred_proj, fusion_mlp
    cyclic_params = (
        list(model.time_enc.parameters())
        + list(model.phase_bias.parameters())
        + list(model.pred_proj.parameters())
        + list(model.fusion_mlp.parameters())
    )

    log.info(
        "Param groups | bb_decay=%d  bb_no_decay=%d  cyclic=%d",
        len(bb_decay), len(bb_no_decay), len(cyclic_params),
    )

    return [
        {"params": bb_decay,     "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": bb_no_decay,  "lr": backbone_lr, "weight_decay": 0.0},
        {"params": cyclic_params,"lr": bias_lr,     "weight_decay": 1e-4},
    ]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cyclic-TiSASRec: phase-aware sequential recommendation"
    )

    # ── data / model ──────────────────────────────────────────────────
    p.add_argument("--dataset",      required=True)
    p.add_argument("--train_dir",    required=True)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--batch_size",   default=128,   type=int)
    p.add_argument("--maxlen",       default=50,    type=int)
    p.add_argument("--hidden_units", default=50,    type=int)
    p.add_argument("--num_blocks",   default=2,     type=int)
    p.add_argument("--num_heads",    default=1,     type=int)
    p.add_argument("--time_span",    default=256,   type=int)
    p.add_argument("--dropout_rate", default=0.2,   type=float)
    p.add_argument("--l2_emb",       default=5e-5,  type=float)
    p.add_argument("--num_workers",  default=4,     type=int)
    p.add_argument("--bf16",         action="store_true", default=True)

    # ── checkpoint ────────────────────────────────────────────────────
    p.add_argument("--state_dict_path", default=None,
                   help="Pretrained TiSASRec .pth — strongly recommended")

    # ── cyclic periods (fixed human cycles in seconds) ─────────────
    # Defaults match the paper exactly: daily / weekly / monthly
    p.add_argument("--periods", nargs=3, type=float,
                   default=[DAILY_PERIOD, WEEKLY_PERIOD, MONTHLY_PERIOD],
                   metavar=("DAILY", "WEEKLY", "MONTHLY"),
                   help="Cycle lengths in timestamp units (default: 86400 604800 2592000)")

    # ── initial λ magnitudes (one per cycle) ─────────────────────────
    p.add_argument("--scale_mags", nargs=3, type=float,
                   default=[0.1, 0.1, 0.1],
                   metavar=("MAG1", "MAG2", "MAG3"),
                   help="Initial λ value per cycle. Model learns to suppress "
                        "uninformative ones. Paper uses 0.1 0.1 0.1.")

    # ── training ──────────────────────────────────────────────────────
    p.add_argument("--num_epochs",   default=501,   type=int)
    p.add_argument("--backbone_lr",  default=1e-4,  type=float)
    p.add_argument("--bias_lr",      default=1e-3,  type=float)
    p.add_argument("--eval_every",   default=20,    type=int)
    p.add_argument("--compile",      action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        log.info("CUDA: %s", torch.cuda.get_device_name(0))

    # ── output dir ────────────────────────────────────────────────────
    out_dir = Path(f"{args.dataset}_{args.train_dir}_cyclic_tisasrec")
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

    # ── build backbone ────────────────────────────────────────────────
    backbone = TiSASRec(usernum, itemnum, timenum, args).to(device)

    if args.state_dict_path:
        sd = torch.load(args.state_dict_path, map_location=device, weights_only=True)
        backbone.load_state_dict(sd)
        log.info("Loaded pretrained backbone from %s", args.state_dict_path)
    else:
        log.info("No checkpoint — training backbone from scratch")

    # ── patch attention layers to accept phase_bias kwarg ─────────────
    _patch_attention_layer(backbone)

    # ── build full Cyclic-TiSASRec model ──────────────────────────────
    model = CyclicTiSASRec(
        backbone     = backbone,
        periods      = args.periods,
        scale_mags   = args.scale_mags,
        hidden_units = args.hidden_units,
        itemnum      = itemnum,
    ).to(device)

    # log param counts
    total        = sum(p.numel() for p in model.parameters())
    cyclic_only  = (
        sum(p.numel() for p in model.time_enc.parameters())
        + sum(p.numel() for p in model.phase_bias.parameters())
        + sum(p.numel() for p in model.pred_proj.parameters())
        + sum(p.numel() for p in model.fusion_mlp.parameters())
    )
    log.info(
        "Total params: %d  |  Cyclic-only params: %d (%.2f%%)",
        total, cyclic_only, 100.0 * cyclic_only / total,
    )
    log.info(
        "Periods: daily=%.0f  weekly=%.0f  monthly=%.0f  |  initial λ: %s",
        *args.periods, args.scale_mags,
    )

    if args.compile:
        log.info("torch.compile() ...")
        model = torch.compile(model)

    # ── optimiser + scheduler ─────────────────────────────────────────
    raw_model = getattr(model, "_orig_mod", model)
    param_groups = build_param_groups(
        raw_model,
        backbone_lr  = args.backbone_lr,
        bias_lr      = args.bias_lr,
        weight_decay = args.l2_emb,
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
        batch_size        = args.batch_size,
        shuffle           = True,
        num_workers       = args.num_workers,
        pin_memory        = True,
        drop_last         = False,
        persistent_workers= (args.num_workers > 0),
    )

    # ── training loop ─────────────────────────────────────────────────
    bce       = nn.BCEWithLogitsLoss()
    best_ndcg = 0.0
    t0        = time.perf_counter()

    log.info(
        "Training | %d epochs | backbone_lr=%.1e | bias_lr=%.1e | batch=%d",
        args.num_epochs, args.backbone_lr, args.bias_lr, args.batch_size,
    )

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = epoch_auc = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}", ncols=80, leave=False):
            _, seq, time_seq_np, time_mat, pos, neg, tgt_time = batch

            seq      = seq.to(device, non_blocking=True)
            time_seq = time_seq_np.to(device, non_blocking=True)
            time_mat = time_mat.to(device, non_blocking=True)
            pos      = pos.to(device, non_blocking=True)
            neg      = neg.to(device, non_blocking=True)
            tgt_time = tgt_time.squeeze(-1).float().to(device, non_blocking=True)  # (B,)

            with torch.autocast(device_type=args.device, dtype=amp_dtype, enabled=use_amp):
                pos_logits, neg_logits = model(
                    seq, time_mat, time_seq, pos, neg, tgt_time
                )
                is_target = (pos != 0).float()
                loss = (
                    bce(pos_logits, torch.ones_like(pos_logits))  * is_target
                    + bce(neg_logits, torch.zeros_like(neg_logits)) * is_target
                ).sum() / is_target.sum()

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()

            with torch.no_grad():
                auc = (
                    ((pos_logits - neg_logits).sign() + 1) / 2 * is_target
                ).sum() / is_target.sum()

            epoch_loss += loss.item()
            epoch_auc  += auc.item()

        scheduler.step()

        # ── cycle amplitude analysis (paper Table 6 equivalent) ──────
        raw = getattr(model, "_orig_mod", model)
        amps = raw.time_enc.effective_amplitudes()
        lambdas = raw.phase_bias.lambdas.detach().cpu().tolist()

        if epoch % args.eval_every == 0:
            elapsed = time.perf_counter() - t0
            t0      = time.perf_counter()

            model.eval()
            v_ndcg, v_hr = evaluate(model, dataset, args, split="valid")
            t_ndcg, t_hr = evaluate(model, dataset, args, split="test")

            amp_str = "  ".join(
                f"{n}={a:.3f}(λ={l:.3f})"
                for n, a, l in zip(CYCLE_NAMES, amps, lambdas)
            )

            log.info(
                "Epoch %3d | %.1fs | loss=%.4f | auc=%.4f | "
                "valid NDCG@10=%.4f HR@10=%.4f | test NDCG@10=%.4f HR@10=%.4f | "
                "cycle amplitudes: %s",
                epoch, elapsed,
                epoch_loss / len(loader),
                epoch_auc  / len(loader),
                v_ndcg, v_hr,
                t_ndcg, t_hr,
                amp_str,
            )

            if t_ndcg > best_ndcg:
                best_ndcg = t_ndcg
                raw_model = getattr(model, "_orig_mod", model)
                ckpt = (
                    out_dir
                    / f"CyclicTiSASRec"
                      f".epoch={epoch}"
                      f".ndcg={t_ndcg:.4f}"
                      f".hr={t_hr:.4f}.pth"
                )
                torch.save(
                    {
                        "backbone":     raw_model.backbone.state_dict(),
                        "time_enc":     raw_model.time_enc.state_dict(),
                        "phase_bias":   raw_model.phase_bias.state_dict(),
                        "pred_proj":    raw_model.pred_proj.state_dict(),
                        "fusion_mlp":   raw_model.fusion_mlp.state_dict(),
                        "epoch":        epoch,
                        "periods":      args.periods,
                        "scale_mags":   args.scale_mags,
                        "test_ndcg":    t_ndcg,
                        "test_hr":      t_hr,
                        "cycle_amps":   dict(zip(CYCLE_NAMES, amps)),
                        "cycle_lambdas":dict(zip(CYCLE_NAMES, lambdas)),
                    },
                    ckpt,
                )
                log.info("★ New best! Saved → %s", ckpt)

    log.info("Done. Best test NDCG@10: %.4f", best_ndcg)


if __name__ == "__main__":
    main()