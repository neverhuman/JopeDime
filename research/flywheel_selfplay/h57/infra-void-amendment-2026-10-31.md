# H57 infra-void accounting amendment — 2026-10-31

This supplemental note records the post-training disk-guard void and its recovery policy before the next H57 outcome attempt. It does not alter the preregistered arms, seeds, cell order, population, eight-generation horizon, admission rule, or 5,400-second cap for accepted outcome/fold-B cells.

## Attempt 1 receipt

`sigma001/seed15` completed its eight deterministic generations from the pinned r882 copy and then failed the unchanged H55 writer guard at 95.08% used, before saving the candidate adapter or summary. The generation-wall sum was 508.971 seconds; the estimated cell elapsed wall was 565.914 seconds. The estimate uses the coordinator START timestamp through the child's final log write because the original coordinator receipt omitted failed-child wall. See `~/jope-data/h57/h57-wave/outcomes/infra-void-attempt1.json`. DOPE remained pinned. The training fitness rows are in-sample diagnostics only and are excluded from all candidate, arm, admission, and terminal decisions. No fold-B, terminal DL-Val, holdout, or outer rows were read.

## Disk gates and retry behavior

A new cell may start only when `statvfs('/')` reports used space below 94.5%. This is a launch margin against disk variation. The existing writer guard remains strictly below 95%; it is not raised or weakened. A child failure specifically identified as `unsafe filesystem / before H55 write` is an infra-void, never an outcome. The coordinator waits for the launch margin and reruns the same arm/seed from the r882 copy, deterministic initial state, and same fixed horizon. At most two retries follow the initial attempt. Each attempt has a separate log; existing partial output is preserved and prevents automatic retry. Void wall is recorded separately from accepted outcome-cell wall.

## Cap interpretation

The owner ruling in `luna_ns_hbaseline_h57_infravoid_t2130.md` says void-cell GPU wall is excluded from the 5,400-second outcome cap but logged. This differs from a literal reading of the earlier cap as covering every GPU cell attempt, including failed infrastructure attempts. Accordingly, this record keeps the accepted-cell projection at 5,220.746/5,400 seconds and adds the first void's estimated 565.914 seconds as separately reported infrastructure expenditure. If all GPU attempts were included, the original projection plus this void would be approximately 5,786.660 seconds, 386.660 seconds above 5,400. No arm or horizon is changed to conceal this distinction.

## Custody

The infra-void receipt and the source evidence are retained locally. An S3 manifest is prepared separately; upload and hash verification remain pending because neither this host nor xbabe3 currently has AWS credentials. No S3 success or deletion of local evidence is claimed.
