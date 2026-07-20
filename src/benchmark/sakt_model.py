"""SAKT — Self-Attentive Knowledge Tracing (transformer-based).

Replaces DKT's LSTM with self-attention: to predict the next exercise, the
model attends over ALL past interactions, learning which ones matter most.

Architecture (Pandey & Karypis 2019):
  interaction embedding (past (skill,correct) pairs)  -> keys & values
  exercise embedding (the skill being asked NOW)      -> queries
  + positional encoding (attention has no inherent order sense)
    -> masked multi-head self-attention (can't peek at the future)
    -> feed-forward -> sigmoid -> P(correct)
"""

import torch
import torch.nn as nn


class SAKT(nn.Module):
    def __init__(self, num_skills: int, max_len: int = 200,
                 embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_skills = num_skills
        self.embed_dim = embed_dim

        # Interaction embedding: (skill, correct) token -> vector  (keys/values source)
        self.interaction_embed = nn.Embedding(2 * num_skills, embed_dim)
        # Exercise embedding: which skill is being asked  (query source)
        self.exercise_embed = nn.Embedding(num_skills, embed_dim)
        # Positional encoding: attention is order-blind, so we add position info
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, 1)

    def forward(self, interaction_tokens, exercise_tokens):
        """interaction_tokens: (B,T) past (skill,correct); exercise_tokens: (B,T) skill asked next."""
        B, T = interaction_tokens.shape
        positions = torch.arange(T, device=interaction_tokens.device).unsqueeze(0)

        # Keys/Values from past interactions (+ position); Query from the exercise asked
        kv = self.interaction_embed(interaction_tokens) + self.pos_embed(positions)
        q = self.exercise_embed(exercise_tokens)

        # Causal mask: position t may only attend to positions <= t (no future peeking)
        causal_mask = torch.triu(torch.ones(T, T, device=interaction_tokens.device),
                                 diagonal=1).bool()

        attn_out, _ = self.attention(q, kv, kv, attn_mask=causal_mask)
        x = self.layer_norm1(attn_out + q)              # residual + norm
        ffn_out = self.ffn(x)
        x = self.layer_norm2(ffn_out + x)               # residual + norm
        return self.output(x).squeeze(-1)               # (B,T) raw logits


if __name__ == "__main__":
    model = SAKT(num_skills=112)
    B, T = 4, 10
    interactions = torch.randint(0, 224, (B, T))
    exercises = torch.randint(0, 112, (B, T))
    out = model(interactions, exercises)
    print(f"Interaction input: {interactions.shape}")
    print(f"Exercise input:    {exercises.shape}")
    print(f"Output:            {out.shape}   (expect (4, 10))")
    print(f"Model params:      {sum(p.numel() for p in model.parameters()):,}")