# Atlas — Client Requirements (v0.1, updated Day 2)

## Domains tracked
- Study: OOPs, DSA, DBMS, CN, OS, Aptitude, System Design (future)
- Fitness: gym sessions, exercise, consistency  ← added Day 2

## Log message styles (must support BOTH)
- Short: "solved 3 dsa mediums, 1 struggled" / "gym 1hr push day"
- Long multi-event: "today i solved 25 apti, 3 code, watched 3 lectures,
  did exercises" → parser must split ONE message into MANY events

## Personalization & multi-user
- Every feature must work per-user (schema: user_id on all rows — done)
- Full multi-user product = post-core phase (documented roadmap)

## Explicitly OUT of scope (senior-dev decision)
- Personalized health/medical recommendations. Atlas tracks fitness and
  gives generic consistency nudges only. Rationale: risk asymmetry —
  wrong study advice is harmless, wrong health advice is not.