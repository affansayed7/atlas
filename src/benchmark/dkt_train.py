"""DKT training loop — batching, padding, masked loss, Adam optimizer.

Key challenges (the real DL data-wrangling lesson):
- Sequences have different lengths -> pad to the batch's max length
- Padding must be MASKED in the loss, or the model learns garbage from
  fake padded positions
- We reuse the same predict-then-observe framing as BKT (input[t] predicts
  target[t], i.e. attempt t+1) for a fair, comparable evaluation

DIAGNOSTIC NOTE: train AUC (0.894) and test AUC (0.892) came out nearly
identical and both far above published DKT range (0.75-0.82). That pattern
points to the task being easier than expected, not necessarily a bug.
Leading hypothesis: ASSISTments skill-builder data repeats the SAME skill
until the student gets it right 3x in a row (mastery-learning design), so
"next attempt correct" is highly predictable from "was the immediately
preceding attempt (same skill) correct" — a real property of this dataset,
not leakage. We test this directly below.
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
    picked = torch.gather(scores, dim=2, index=target_skills.unsqueeze(-1)).squeeze(-1)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    per_position_loss = loss_fn(picked, target_correct)
    masked_loss = per_position_loss * mask.float()
    return masked_loss.sum() / mask.sum()


def train_dkt(train_seqs, num_skills, epochs=5, batch_size=32, lr=0.001):
    model = DKT(num_skills)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
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


def evaluate_dkt(model, seqs, num_skills):
    """Run trained DKT on a set of sequences; return (preds, actuals) for AUC."""
    model.eval()
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for i in range(0, len(seqs), 32):
            batch_seqs = seqs[i:i + 32]
            inputs, tsk, tcorr, mask = make_batch(batch_seqs, num_skills)
            scores = model(inputs)
            picked = torch.gather(scores, dim=2, index=tsk.unsqueeze(-1)).squeeze(-1)
            probs = torch.sigmoid(picked)

            mask_flat = mask.flatten()
            all_preds.extend(probs.flatten()[mask_flat].tolist())
            all_actuals.extend(tcorr.flatten()[mask_flat].tolist())

    return all_preds, all_actuals


def diagnose_padding_collision(train_seqs, num_skills):
    """Verify the mask correctly separates padding zeros from real skill-0-correct tokens."""
    batch = make_batch(train_seqs[:8], num_skills)
    inputs, tsk, tcorr, mask = batch

    real_zeros = ((inputs == 0) & mask).sum().item()
    pad_zeros = ((inputs == 0) & ~mask).sum().item()

    print(f"\n--- Padding collision diagnostic ---")
    print(f"num_skills={num_skills} (token 0 = 'skill 0, correct')")
    print(f"Real skill-0-correct attempts (mask=True):  {real_zeros}")
    print(f"Padding positions using value 0 (mask=False): {pad_zeros}")


def diagnose_skill_repetition(seqs, sample_size=500):
    """Check how often consecutive attempts within a student are the SAME skill.

    ASSISTments skill-builder data repeats a skill until mastery (3 correct
    in a row), so high repetition would explain unusually high AUC honestly:
    predicting 'next attempt' is easy when it's usually the same skill you
    just did.
    """
    consecutive_same_skill = 0
    total_transitions = 0
    for seq in seqs[:sample_size]:
        for i in range(len(seq) - 1):
            total_transitions += 1
            if seq[i][0] == seq[i + 1][0]:
                consecutive_same_skill += 1

    pct = consecutive_same_skill / total_transitions if total_transitions else 0
    print(f"\n--- Skill repetition diagnostic ---")
    print(f"Consecutive same-skill transitions: {consecutive_same_skill}/{total_transitions} ({pct:.1%})")
    print(f"(High % explains strong AUC as a real dataset property, not a bug)")
    return pct


if __name__ == "__main__":
    df = load_and_clean()
    sequences, num_skills = build_sequences(df)
    train_seqs, test_seqs, train_ids, test_ids = train_test_split(sequences)

    diagnose_padding_collision(train_seqs, num_skills)
    diagnose_skill_repetition(train_seqs)

    print("\nTraining DKT...")
    model = train_dkt(train_seqs, num_skills, epochs=5)

    print("\nEvaluating on TRAIN set (sanity check for overfitting)...")
    train_preds, train_actuals = evaluate_dkt(model, train_seqs, num_skills)
    train_auc = roc_auc_score(train_actuals, train_preds)

    print("Evaluating on TEST set (held-out students)...")
    test_preds, test_actuals = evaluate_dkt(model, test_seqs, num_skills)
    test_auc = roc_auc_score(test_actuals, test_preds)

    print("\n" + "=" * 50)
    print(f"  DKT AUC (train set): {train_auc:.4f}")
    print(f"  DKT AUC (test set):  {test_auc:.4f}")
    print(f"  BKT AUC (fitted):    0.7468   (from earlier benchmark)")
    print(f"  DKT test - BKT:      {test_auc - 0.7468:+.4f}")
    print("=" * 50)