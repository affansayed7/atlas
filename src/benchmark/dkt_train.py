"""DKT training loop — batching, padding, masked loss, Adam optimizer.

Key challenges (the real DL data-wrangling lesson):
- Sequences have different lengths -> pad to the batch's max length
- Padding must be MASKED in the loss, or the model learns garbage from
  fake padded positions
- We reuse the same predict-then-observe framing as BKT (input[t] predicts
  target[t], i.e. attempt t+1) for a fair, comparable evaluation
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score

from src.benchmark.dkt_data import build_sequences, encode_sequence, train_test_split
from src.benchmark.dkt_model import DKT
from src.benchmark.evaluate import load_and_clean


def make_batch(seqs, num_skills):
    """Turn a list of raw (skill,correct) sequences into padded tensors + a mask.

    Returns: inputs (B,T), target_skills (B,T), target_correct (B,T), mask (B,T)
    mask[b,t] = 1 if position t is REAL data, 0 if it's padding.
    """
    encoded = [encode_sequence(s, num_skills) for s in seqs]
    inputs = pad_sequence([e[0] for e in encoded], batch_first=True, padding_value=0)
    target_skills = pad_sequence([e[1] for e in encoded], batch_first=True, padding_value=0)
    target_correct = pad_sequence([e[2] for e in encoded], batch_first=True, padding_value=0)

    lengths = torch.tensor([len(e[0]) for e in encoded])
    max_len = inputs.shape[1]
    mask = torch.arange(max_len)[None, :] < lengths[:, None]   # (B,T) bool
    return inputs, target_skills, target_correct, mask


if __name__ == "__main__":
    df = load_and_clean()
    sequences, num_skills = build_sequences(df)
    train_seqs, test_seqs = train_test_split(sequences)

    # Smoke test: batch the first 4 training sequences and inspect shapes
    batch = make_batch(train_seqs[:4], num_skills)
    inputs, tsk, tcorr, mask = batch
    print(f"inputs:  {inputs.shape}")
    print(f"mask:    {mask.shape}  (sum={mask.sum().item()} real positions "
          f"out of {mask.numel()} total)")