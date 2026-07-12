"""Bayesian Knowledge Tracing — estimates P(mastery) of a skill from attempt history.

Four parameters:
- p_L0:    prior P(known) before any practice
- p_T:     P(transit) — chance of learning on each attempt (unknown -> known)
- p_slip:  P(knows but answers wrong)
- p_guess: P(doesn't know but answers right)

Pure logic: no DB, no network — fully unit-testable.
"""

from dataclasses import dataclass


@dataclass
class BKTParams:
    p_L0: float = 0.2
    p_T: float = 0.3
    p_slip: float = 0.1
    p_guess: float = 0.2


class BKT:
    def __init__(self, params: BKTParams | None = None):
        self.params = params or BKTParams()
        self.p_known = self.params.p_L0  # current mastery estimate

    def update(self, correct: bool) -> float:
        """Apply one attempt (correct/incorrect). Returns the new P(known)."""
        p = self.params
        prior = self.p_known

        # Step 1 — condition on the observed evidence (Bayes' rule)
        if correct:
            numerator = prior * (1 - p.p_slip)
            denominator = prior * (1 - p.p_slip) + (1 - prior) * p.p_guess
        else:
            numerator = prior * p.p_slip
            denominator = prior * p.p_slip + (1 - prior) * (1 - p.p_guess)

        p_known_given_evidence = numerator / denominator if denominator > 0 else prior

        # Step 2 — account for learning during this attempt (transit)
        self.p_known = p_known_given_evidence + (1 - p_known_given_evidence) * p.p_T
        return self.p_known

    def fit_sequence(self, attempts: list[bool]) -> float:
        """Apply a whole sequence of attempts; return final P(known)."""
        for correct in attempts:
            self.update(correct)
        return self.p_known