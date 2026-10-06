# dltrain research lane (PLAN route DL-D1=A)

Research-only PyTorch lane for GP program generation; never serves. Plan, audit and gates:
`/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/{PLAN,AUDIT,handoff}.md`.

- Claim: AGENT_CHAT `2026-10-05T04:45:30.385967Z CLAIM dltrain-impl-v1` (Claude, screen 442374.JopeDime_DLTrain).
- Owner decisions: `claude/receipts/owner-decisions-20261005T044116Z.json` (CPU-only P0-P7 admitted; GPU-hours 0; deadline 2026-10-08T00:00Z).
- Coordination: `/home/ubuntu/JopeDime/DEEP_LEARNING_AGENT_CHAT.md` (flock `.lock`), plus locked AGENT_CHAT claims.
- Paths: SRC (this dir, git-ignored), RUN=/home/ubuntu/scratch/jopedime-dl-training-v1, VENV=/home/ubuntu/.venvs/jope-dltrain-v1,
  PREFIX=s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v1/.
