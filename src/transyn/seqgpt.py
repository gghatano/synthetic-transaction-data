"""小さな GPT 型の自己回帰 Transformer によるイベント系列の生成（issue #10）。

HALO・SynEHRgy・Banksformer と同じ系統（自己回帰 Transformer）の最小限の実装。各系列を次のトークン列にする:

    <bos> [属性トークン...] [コード] [間隔] [コード] [間隔] ... <eos>

- 属性トークン: spec.discrete_context の値と、spec.context の数値列を十分位で区切った区間
- 間隔トークン: 直前のイベントからの時間（spec.time_unit 単位）を log スケールの区間に離散化したもの。
  生成時は区間内で一様に値を戻す。先頭イベントの間隔は、系列の開始時刻（spec.origin からの経過）を表す
- 数値列（spec.value）には対応しない
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import SeqSpec, _seed

N_GAP_BINS = 24
N_NUM_BINS = 10
MAX_TOKENS = 512


class _GPT(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_layers: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        mask = torch.triu(torch.full((n, n), float("-inf"), device=x.device), diagonal=1)
        return self.head(self.blocks(h, mask=mask, is_causal=True))


class _Tokenizer:
    def __init__(self, train: pd.DataFrame, spec: SeqSpec):
        self.spec = spec
        unit = pd.Timedelta(1, unit=spec.time_unit)
        ordered = train.sort_values(["seq_id", "datetime"], kind="stable")
        first = ordered.groupby("seq_id")["datetime"].transform("min")
        gaps = ordered.groupby("seq_id")["datetime"].diff().fillna(first - spec.origin) / unit
        positive = gaps[gaps > 0]
        # 0 は専用の区間、正の間隔は log スケールの分位点で区切る
        self.gap_edges = np.unique(np.quantile(np.log(positive), np.linspace(0, 1, N_GAP_BINS)))
        self.num_edges = {
            col: np.unique(np.quantile(train.groupby("seq_id")[col].first(), np.linspace(0, 1, N_NUM_BINS + 1)))
            for col in spec.context
            if col not in spec.discrete_context
        }
        tokens = ["<pad>", "<bos>", "<eos>"]
        for col in spec.context:
            if col in spec.discrete_context:
                tokens += [f"{col}={v}" for v in sorted(train[col].astype(str).unique())]
            else:
                tokens += [f"{col}#{i}" for i in range(len(self.num_edges[col]) - 1)]
        tokens += [f"gap#{i}" for i in range(len(self.gap_edges) + 1)]
        tokens += [f"code={c}" for c in sorted(train["tcode"].unique())]
        self.itos = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.unit = unit

    def _gap_token(self, gap: float) -> str:
        if gap <= 0:
            return "gap#0"
        return f"gap#{1 + int(np.clip(np.searchsorted(self.gap_edges, math.log(gap)) - 1, 0, len(self.gap_edges) - 1))}"

    def encode(self, train: pd.DataFrame) -> list[list[int]]:
        seqs = []
        ordered = train.sort_values(["seq_id", "datetime"], kind="stable")
        for _, g in ordered.groupby("seq_id", sort=False):
            toks = ["<bos>"]
            for col in self.spec.context:
                v = g[col].iloc[0]
                if col in self.spec.discrete_context:
                    toks.append(f"{col}={v}")
                else:
                    edges = self.num_edges[col]
                    toks.append(f"{col}#{int(np.clip(np.searchsorted(edges, v, side='right') - 1, 0, len(edges) - 2))}")
            prev = self.spec.origin
            for t, code in zip(g["datetime"], g["tcode"]):
                toks += [f"code={code}", self._gap_token((t - prev) / self.unit)]
                prev = t
            toks.append("<eos>")
            seqs.append([self.stoi[x] for x in toks[:MAX_TOKENS]])
        return seqs

    def decode(self, ids: list[int], seq_id: int, rng: np.random.Generator) -> list[dict]:
        toks = [self.itos[i] for i in ids]
        attrs: dict = {}
        for col in self.spec.context:
            if col in self.spec.discrete_context:
                hit = next((t for t in toks if t.startswith(f"{col}=")), None)
                attrs[col] = type_cast(hit.split("=", 1)[1]) if hit else None
            else:
                hit = next((t for t in toks if t.startswith(f"{col}#")), None)
                edges = self.num_edges[col]
                i = int(hit.split("#")[1]) if hit else 0
                attrs[col] = int(round(rng.uniform(edges[i], edges[i + 1])))
        if any(v is None for v in attrs.values()):
            return []  # 属性トークンを生成しなかった系列は捨てる
        rows, now, code = [], self.spec.origin, None
        for t in toks:
            if t.startswith("code="):
                code = t.split("=", 1)[1]
            elif t.startswith("gap#") and code is not None:
                i = int(t.split("#")[1])
                if i == 0:
                    gap = 0.0
                else:
                    lo = self.gap_edges[i - 1]
                    hi = self.gap_edges[i] if i < len(self.gap_edges) else lo + 0.5
                    gap = math.exp(rng.uniform(lo, hi))
                now = now + gap * self.unit
                rows.append({"seq_id": seq_id, "datetime": now.round(self.spec.time_resolution), "tcode": code, **attrs})
                code = None
        return rows


def type_cast(v: str):
    try:
        return int(v)
    except ValueError:
        return v


def fit_generate_gpt(
    train: pd.DataFrame,
    n_seqs: int,
    epochs: int,
    seed: int,
    spec: SeqSpec,
    d_model: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.1,
    lr: float = 3e-4,
    batch_size: int = 32,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """GPT 型モデルを学習し、n_seqs 系列を生成する。属性トークンも含めて先頭から自己回帰で生成する。"""
    if spec.value:
        raise ValueError("seqgpt は数値列（spec.value）に対応していない")
    _seed(seed)
    tok = _Tokenizer(train, spec)
    data = tok.encode(train)
    max_len = max(len(s) for s in data)
    model = _GPT(len(tok.itos), d_model, n_layers, n_heads, max_len, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    started = time.time()
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(data))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = [data[i] for i in order[start : start + batch_size]]
            width = max(len(s) for s in batch)
            x = torch.zeros(len(batch), width, dtype=torch.long)
            for j, s in enumerate(batch):
                x[j, : len(s)] = torch.tensor(s)
            logits = model(x[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=tok.stoi["<pad>"]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch}/{epochs} loss {total / len(data):.3f} ({time.time() - started:.0f}s)", flush=True)

    model.eval()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    eos, bos = tok.stoi["<eos>"], tok.stoi["<bos>"]
    with torch.no_grad():
        for start in range(0, n_seqs, 64):
            n = min(64, n_seqs - start)
            x = torch.full((n, 1), bos, dtype=torch.long)
            done = torch.zeros(n, dtype=torch.bool)
            while x.shape[1] < max_len and not done.all():
                probs = torch.softmax(model(x)[:, -1] / temperature, dim=-1)
                probs[:, tok.stoi["<pad>"]] = 0
                nxt = torch.multinomial(probs, 1)
                nxt[done] = tok.stoi["<pad>"]
                done |= nxt[:, 0] == eos
                x = torch.cat([x, nxt], dim=1)
            for j in range(n):
                rows += tok.decode(x[j].tolist(), start + j, rng)
    gen = pd.DataFrame(rows)
    return gen.sort_values(["seq_id", "datetime"], kind="stable").reset_index(drop=True)
