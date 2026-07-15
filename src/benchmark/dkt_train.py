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


def masked_bce_loss(scores, target_skills, target_correct, mask):
    """BCE loss, but only on the skill actually asked, and only on real (non-pad) positions."""
    # Pick out the score for the specific skill at each position: (B,T,num_skills) -> (B,T)
    picked = torch.gather(scores, dim=2, index=target_skills.unsqueeze(-1)).squeeze(-1)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")   # per-element loss, we mask it ourselves
    per_position_loss = loss_fn(picked, target_correct)
    masked_loss = per_position_loss * mask.float()
    return masked_loss.sum() / mask.sum()               # average over REAL positions only


def train_dkt(train_seqs, num_skills, epochs=5, batch_size=32, lr=0.001):
    model = DKT(num_skills)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        # Shuffle sequence order each epoch (standard practice, reduces overfitting to order)
        perm = torch.randperm(len(train_seqs))
        shuffled = [train_seqs[i] for i in perm]

        for i in range(0, len(shuffled), batch_size):
            batch_seqs = shuffled[i:i + batch_size]
            inputs, tsk, tcorr, mask = make_batch(batch_seqs, num_skills)

            optimizer.zero_grad()
            scores = model(inputs)
            loss = masked_bce_loss(scores, tsk, tcorr, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch+1}/{epochs} — avg loss: {total_loss/n_batches:.4f}")

    return model


if __name__ == "__main__":
    df = load_and_clean()
    sequences, num_skills = build_sequences(df)
    train_seqs, test_seqs = train_test_split(sequences)

    print("\nTraining DKT...")
    model = train_dkt(train_seqs, num_skills, epochs=5)
    print("\nTraining complete.")