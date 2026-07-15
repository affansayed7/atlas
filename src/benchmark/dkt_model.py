"""DKT model — LSTM-based knowledge tracing.

Architecture:
  input tokens (skill+correct encoding)
    -> Embedding (learned vector per token)
    -> LSTM (reads sequence, maintains hidden knowledge state)
    -> Linear (projects hidden state -> one score per skill)
    -> sigmoid (applied at eval time -> P(correct) per skill)
"""

import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, num_skills: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.num_skills = num_skills
        # +1 because token range is 0..2*num_skills-1; we reserve nothing extra here
        # but embeddings need vocab_size = 2*num_skills (correct+incorrect per skill)
        self.embedding = nn.Embedding(num_embeddings=2 * num_skills, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_skills)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """input_tokens: (batch, seq_len) -> returns (batch, seq_len, num_skills) raw scores."""
        embedded = self.embedding(input_tokens)          # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)                 # (batch, seq_len, hidden_dim)
        scores = self.output(lstm_out)                     # (batch, seq_len, num_skills)
        return scores  # raw logits — sigmoid applied later (loss function does it internally)


if __name__ == "__main__":
    # Smoke test: does the model run on a tiny fake batch without crashing?
    model = DKT(num_skills=123)
    fake_batch = torch.randint(0, 246, (4, 10))   # 4 sequences, length 10, tokens 0..245
    out = model(fake_batch)
    print(f"Input shape:  {fake_batch.shape}")
    print(f"Output shape: {out.shape}")           # expect (4, 10, 123)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")