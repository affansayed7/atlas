"""Unit tests for BKT — including the hand-calculated 0.818 case."""

from src.analytics.bkt import BKT, BKTParams


def test_single_correct_update_matches_hand_calc():
    """Hand-calculated: prior 0.5, correct, slip 0.1, guess 0.2, T=0.
    Bayes step alone → 0.8181... With T=0, transit adds nothing."""
    bkt = BKT(BKTParams(p_L0=0.5, p_T=0.0, p_slip=0.1, p_guess=0.2))
    result = bkt.update(correct=True)
    assert abs(result - 0.8181818) < 1e-4, f"Expected ~0.818, got {result}"
    print(f"PASS: single correct update → {result:.4f} (matches hand calc)")


def test_correct_raises_mastery():
    bkt = BKT(BKTParams(p_L0=0.5))
    before = bkt.p_known
    after = bkt.update(correct=True)
    assert after > before
    print(f"PASS: correct answer raises mastery ({before:.3f} → {after:.3f})")


def test_incorrect_lowers_evidence():
    """A wrong answer should lower the Bayes estimate (before transit)."""
    bkt = BKT(BKTParams(p_L0=0.5, p_T=0.0))  # T=0 to isolate the Bayes step
    after = bkt.update(correct=False)
    assert after < 0.5
    print(f"PASS: incorrect answer lowers mastery (0.5 → {after:.3f})")


def test_mastery_climbs_with_repeated_success():
    bkt = BKT()
    final = bkt.fit_sequence([True, True, True, True, True])
    assert final > 0.9
    print(f"PASS: 5 correct in a row → mastery {final:.3f}")


if __name__ == "__main__":
    test_single_correct_update_matches_hand_calc()
    test_correct_raises_mastery()
    test_incorrect_lowers_evidence()
    test_mastery_climbs_with_repeated_success()
    print("\nAll BKT tests passed ✅")