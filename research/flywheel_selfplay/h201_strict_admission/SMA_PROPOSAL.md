# H201 SMA proposal — no live change

The descriptive replay rejects all 2,418 candidates, including four ties and four
improvements, satisfying H2. Propose a separate owner-reviewed admission PR that
requires strictly positive paired-family lower95% gain and rejects ties, with
the chosen frozen model's parameter hash bound to its evaluated utility vector.

Use the approved native DL-Val identities and an explicit execution contract
(device, batch shape, candidate order and first-argmin semantics). Require baseline
regression and checkpoint/evaluation parity before considering admission. Persist
the paired context/family vectors, interval, exact model hash and decision for
every candidate; checkpoint and utility publication must refer to the same model.
Keep DOPE frozen and require rollback on failed admission or identity validation.

Before rollout, calibrate power and sequential false-positive behavior for repeated
adaptive candidates. Nineteen families may give a highly conservative rule: this
counterfactual retained its r774 initial state and accepted nothing. The replay
does not justify rolling the live model back or claim superior future utility.
The H201 native/archive r3175 gate failed, so neither that checkpoint nor this
policy is qualified by this report. Deployment/adoption remains a separate owner
decision after this closeout; this proposal performs no live write.
