# Delivery corrections after genuine independent review

The fresh actual Claude session 4b5e9a23-ce2f-494c-83da-fbfef0088f56 reviewed source 3f9b8200f5b79d0fcb3a5ff5ab25cc695b918322 and submitted COMMENT review 5560179600 with a NOT APPROVED verdict. It identified two blocking delivery defects despite the source's passing CI and static coverage checks.

B1: the validation matrix joined 66 rows with literal backslash-n text. GitHub rendered only I01. This revision replaces those delimiters with physical newlines, retains every command and acceptance cell, adds an independent static row check and verifies the actual GitHub GFM render contains I01 through I66. See RENDER_CHECK.json.

B2: the prior PR description incorrectly described git diff --check as clean. Reproduction at the reviewed v3 source exits 2 with 23 trailing-whitespace findings, all in original reference inputs S3, S4 and S8. Those bytes include Markdown hard-line-break spaces and must stay identical to the supplied specifications. WHITESPACE_EXCEPTION.json preserves command arguments, exit codes, diagnostic locations and output hashes. The separately scoped authored-doc check exits 0. The final source owner repeats both checks at the new exact head and records their actual results in the external publication receipt and PR body.

Commands from the registered source checkout, after confirming the base and ownership:

~~~sh
rtk git diff --check 0b7eb89075b8186f1aacf1643725432e75e8fe56 HEAD
# Expected exit 2: 23 unchanged original-input whitespace findings; not PASS.
rtk git diff --check 0b7eb89075b8186f1aacf1643725432e75e8fe56 HEAD -- docs/plans/algorithmic-selfplay-20261017/ ':(exclude)docs/plans/algorithmic-selfplay-20261017/reference-specs/**'
# Required exit 0 for authored documentation.
~~~

The concurrent APPROVED review 5560157745 on v3 is not credited as the genuine Claude review. The actual reviewer explicitly stated it did not post that approval. Its origin is unverified; its API record and all launch attempts are retained externally. The PR was returned to draft. A corrected source needs fresh full CI and a fresh actual independent reviewer; no old or unattributed approval qualifies the new head.

Other corrections: historical manifests are explicitly labelled in-tree; the baseline source/test anchor count is 134; owner-local macOS intake paths are omitted from current public planning metadata while original private intake and prior Git history remain intact; the Grok blocker records its bytes/hash and labels external evidence as external; progress declares its snapshot semantics. Source input bytes, scientific holds, native grant restrictions, custody holds, 66 proposed packets and 30 scientific/design decisions remain unchanged.
