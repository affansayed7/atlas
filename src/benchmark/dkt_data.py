"""DKT data pipeline — turn ASSISTments attempts into padded tensor sequences.

Transformation:
  raw rows → per-student sequences → (skill,correct) integer tokens
  → padded equal-length tensors, split by student (no leakage).

Key encoding: skill s answered CORRECT -> token s
              skill s answered WRONG   -> token s + num_skills
  So with 123 skills, tokens range 0..245.
"""

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from src.benchmark.evaluate import load_and_clean


def build_sequences(df):
    """Return per-student lists of (skill_idx, correct) and skill vocab info."""
    # Map raw skill_ids -> compact 0..N-1 indices (skill_ids are large/sparse)
    unique_skills = sorted(df["skill_id"].unique())
    skill_to_idx = {sid: i for i, sid in enumerate(unique_skills)}
    num_skills = len(unique_skills)

    sequences = []
    # groupby preserves per-student order since df is already sorted by order_id
    for _uid, group in df.groupby("user_id", sort=False):
        seq = [
            (skill_to_idx[sid], int(correct))
            for sid, correct in zip(group["skill_id"], group["correct"])
        ]
        if len(seq) >= 2:            # need at least 2: one to learn from, one to predict
            sequences.append(seq)

    print(f"Built {len(sequences):,} student sequences over {num_skills} skills")
    return sequences, num_skills


def encode_sequence(seq, num_skills):
    """Encode (skill, correct) pairs into DKT input/target tensors.

    inputs[t]  = token for attempt t  (skill + correct*num_skills)
    target_skill[t] = the skill of attempt t+1 (what we predict)
    target_correct[t] = whether attempt t+1 was correct (the label)
    """
    inputs, target_skills, target_correct = [], [], []
    for t in range(len(seq) - 1):
        skill_t, correct_t = seq[t]
        token = skill_t + correct_t * num_skills          # 0..2*num_skills-1
        inputs.append(token)
        next_skill, next_correct = seq[t + 1]
        target_skills.append(next_skill)
        target_correct.append(next_correct)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(target_skills, dtype=torch.long),
        torch.tensor(target_correct, dtype=torch.float),
    )


def train_test_split(sequences, test_frac=0.2, seed=42):
    """Split by STUDENT (not attempt) so test students are fully unseen."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))
    n_test = int(len(sequences) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train = [sequences[i] for i in train_idx]
    test = [sequences[i] for i in test_idx]
    print(f"Split: {len(train):,} train / {len(test):,} test students")
    return train, test


if __name__ == "__main__":
    df = load_and_clean()
    sequences, num_skills = build_sequences(df)
    train, test = train_test_split(sequences)

    # Sanity check: encode one sequence and show shapes
    inp, tsk, tcorr = encode_sequence(train[0], num_skills)
    print(f"\nExample sequence length: {len(train[0])} attempts")
    print(f"  inputs shape:  {inp.shape}   (first 5: {inp[:5].tolist()})")
    print(f"  target_skills: {tsk.shape}   (first 5: {tsk[:5].tolist()})")
    print(f"  target_correct:{tcorr.shape} (first 5: {tcorr[:5].tolist()})")
    print(f"\nnum_skills = {num_skills}, so input tokens range 0..{2*num_skills-1}")