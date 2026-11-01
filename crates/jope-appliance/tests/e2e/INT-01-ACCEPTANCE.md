# INT-01 five-surface acceptance record

This record links the five-surface acceptance evidence for the source merged by PR
[#1308](https://github.com/neverhuman/JopeDime/pull/1308). The tested candidate was
`b4fc2aaeabdf31c647abf05f8afa93be253be8e7`, merged as
`f696db5484a50cdcb604e22d0e261644bcbab1a1`; the verified merged tree was
`fe38f60ade1635f2b386187125f6cc78e4d11cbe`.

## Acceptance evidence

The final Linux acceptance receipt records distinct runs initiated through batch CLI,
loopback HTTP, in-process client, MCP, and terminal. Their champion, scores, and accepted
R2 projection matched. The same receipt records fresh-model FitHoldout parity over 64
TRAIN feature rows: 26/26 AG1 checks passed, Python and compiled Rust maximum absolute
error were both zero, and there were no sealed external-holdout reads or new fits. The
source-bound receipt uses the original 960-row TRAIN input (SHA-256
`b997f975932e19b15c070f9e5f9170140407fdd5bf6df676854366259f5b3927`).

The merged-source local suite recorded 228 passed / 0 failed / 9 existing ignored, and
the six-ID Linux e2e target recorded 6 passed / 0 failed / 0 ignored. `main-guard` run
41276032417 succeeded after merge. The owner authorized targeted local checks and
admin-squash for this closeout; hosted full-matrix CI was not dispatched.

Receipts are retained outside the repository:

- `/home/ubuntu/jope_plan/evidence/INT-01/gov03-t2115/finish5-acceptance.json`
  (SHA-256 `588c6bf6064cb35680ade1ee1d53608560d23b7cb3b3a59c7bbf3513d6ab8fe5`)
- `/home/ubuntu/jope_plan/evidence/INT-01/gov03-t2115/MERGED-VERIFIED.json`
  (SHA-256 `4d9f38c3c90ebc2295417cb8b36967d2e540156ec1a1539edee7d9f83921376d`)

## Native Darwin record and disposition

The archived native Darwin run for candidate `0bcc447482fb9c30d75bec76c03cbe5567ac07cf`
passed the six-ID e2e target (6/6); its log SHA-256 is
`6f1e0aa7c6975b8e9b557d613de0f034b894504423b083beb4071c3260327a8e`. That is an earlier
candidate and is not an exact-source native run for `b4fc2aa`. The matching newer-source
Darwin receipt was not produced here. Owner ruling `owner_ruling_t2315.md` explicitly
rules the gates passed for process purposes and directs INT-01 to DONE. The full-matrix
hosted CI was also deferred under that ruling; neither point is represented as a newly
executed test result.
