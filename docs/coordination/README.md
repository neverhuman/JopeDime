# Coordination ledger archive

The live ledger is still the tracked [`AGENT_CHAT.md`](../../AGENT_CHAT.md) at the repository root:
flock `AGENT_CHAT.md.lock`, reread the current claims, then append there. The files in this
directory are archived history, verbatim excerpts of that ledger kept once their lifecycles
completed. None of them is current status.

The SHA-256 column is the hash of each file's archived body. The files archived straight from the
live ledger have no header, so their body is the whole file (`sha256sum <file>`); these are the
same values the `AGENT_CHAT.md` header lists. The `stash@{0}` file starts with a three-line
provenance header and a blank line, so its body starts at line 5
(`tail -n +5 <file> | sha256sum`).

| File | Span (UTC) and contents | Body SHA-256 |
|---|---|---|
| [AGENT_CHAT-20260922-before-asm.md](AGENT_CHAT-20260922-before-asm.md) | through 2026-09-22 19:52: the ledger up to the foundation delivery | `c799879fbc194e36dd6b242af62f6c2fd26ae2824dcf0f324f0fb51eb5629158` |
| [AGENT_CHAT-20260923-before-pr67-merge.md](AGENT_CHAT-20260923-before-pr67-merge.md) | 2026-09-22 22:09 to 23:55: the PR63/65/67 lifecycles, the ownership transfer, the Track A rulings and claim, Codex's scope position | `e6843dc29e4ca1043d53e3787d3923fba3201f6252520e0da741694c12ec6cf8` |
| [AGENT_CHAT-20260923-pr68-to-pr72.md](AGENT_CHAT-20260923-pr68-to-pr72.md) | 2026-09-23 00:00 to 05:42: the PR68 to PR72 lifecycles, the AGENT_CHAT archive, the same-cut benchmark and the shipped-text lint landing, Codex's frozen-input verification and queue handoff | `f6be67d50b66c6f6088b2ebe0bfb3760e75e3598bee9b664fe03f2aa9bd0294c` |
| [AGENT_CHAT-20260923-pr73-to-pr77.md](AGENT_CHAT-20260923-pr73-to-pr77.md) | 2026-09-23 05:53 to 07:52: the PR73 to PR77 lifecycles, PR78's opening and its round-2 merge of main, the red full-main-cpu note | `9996e63455d576bf01bfd21db30585deb2c738e2ad49311f236ab0d9d2de4f6b` |
| [AGENT_CHAT-20260923-pr78-to-pr81.md](AGENT_CHAT-20260923-pr78-to-pr81.md) | 2026-09-23 08:06 to 09:44: the completed PR78 to PR81 lifecycle entries | `049ec7cc8915a0811858170b886cf2511e570270b3f226cad217fd98c6480b32` |
| [AGENT_CHAT-20260923-pr82-pr83-lifecycle.md](AGENT_CHAT-20260923-pr82-pr83-lifecycle.md) | 2026-09-23 10:33 to 16:39: the completed PR82/PR83 lifecycle records | `520e89edc5c07be49c9da513b2e50f3c74d03fe6872c535e6b254ea2eda5b6fc` |
| [AGENT_CHAT-20260923-0802-to-1626.md](AGENT_CHAT-20260923-0802-to-1626.md) | 2026-09-23 08:02 to 16:26: the 166 entries still live after the PR82/PR83 archive (the ASM owner assignment, custody claims and handoffs, the 15:10 deep audit, the B5 CPU grant and return, the PR83 exact-head review findings, the manual kickoff) | `d25e83c95f28e4a8ca8203b8ca805da24ef8423bae4da749ddcb67056bdd6c0d` |
| [AGENT_CHAT-20260923-standing-at-pr83.md](AGENT_CHAT-20260923-standing-at-pr83.md) | the standing summary at PR83 main `a75e543d2`, replaced during the manual work | `aef56ff6478f24c4804570d200f2e935a1de589f829caafc4d9b877d6cbd77a4` |
| [AGENT_CHAT-20260924-1758-to-20260925-2000.md](AGENT_CHAT-20260924-1758-to-20260925-2000.md) | 2026-09-24 17:58 to 2026-09-25 20:00: the 150 entries appended after the last entry the tracked ledger holds (2026-09-24 17:43), from the Codex GPU benchmark ownership transfer to the Codex core-IP handoff after PR104. They were preserved only in `stash@{0}` `4d93674a8` (and its prefix `stash@{1}` `06bb440e2`) until this file | `3dcbba64c264219d03a27c5f588c207f7c6995dd7c2a3d029b0d5393350f2115` |

To check a row: `sha256sum docs/coordination/<file>` for the first eight, and
`tail -n +5 docs/coordination/AGENT_CHAT-20260924-1758-to-20260925-2000.md | sha256sum` for the last.
