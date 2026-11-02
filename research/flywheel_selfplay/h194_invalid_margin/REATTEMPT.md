# H194 infrastructure repeats: exercised and closed

A1 stopped on the shared xbabe2 disk93.8% guard, after1080updates and no B selection or
candidate A/Val. Owner t1200 (`hwave_t1200_reattempt_h194.md`,SHA256
`8e415dff378fa81d882d63d7b57c615a50d9b952a6dff68dbb90ac8e9930cb83`) authorized one fresh A2
after A1closeout#1627 merged97ce7c95e9d573eb9b7c5518953b64bc9937e41c.
A2 stopped on load131.8555>108.8 across3independent5s windows after1246optimizer calls,
with no B lock or candidate A/Val. Its closeout#1674 merged55dda298f6d2f9ac536f34c4260ca66fea13ca68
at reviewed12d7ea4c3266a473718abfefce924113ed1cb1e3;22controls/Ruff/5records passed.
A2's zero-model prebinding abort and31-state zero-forward checksum closure remain preserved.

Owner t1423 (`hwave_t1423_reattempt_h194_a3.md`,SHA256
`c822c821f6177a5a4171385d5f012f66235860e1e4f3a14d19eda1830754bdd7`) authorized ONE FINAL A3
only after A2closeout. No A4 was authorized. A3 completed all45trajectories/1800updates,
all45candidate A/45Val panels in169.22minutes, ending NO_ADVANCE. No A1/A2 trajectory or
B-selection reuse occurred; the earlier infrastructure stops supplied no candidate A/Val
result to choose among. All4126optimizer calls across the three attempts are reported.

The scientific source remains #1601/9686429f7e89fb34da3e797afbb2855c93e02fb1, unchanged active
kernel/frozen DOPE c51a56d0/47adapter tensors29211parameters. Seeds194101–194103,independent
fold seeds194141–194143,bootstrap194150,and40updates per arm/cell remained fixed.

Actual preferred host xbabe1 qualified with1336SHA-identical inputs; only4files/14386305bytes
were copied. Byte hashing staged opaque TRAIN files; native91 exact utility/cost replay preceded
numeric TRAIN decoding. External runner wrapper changed exactly the in-memory actual-host
admission predicate from xbabe2-only to xbabe1/xbabe2, without changing the original file,
math, function bodies, or hostname. Original runnerSHA1af8e2b1cb2c375566feca1248a291b2f593ec01191a8f9e276770a0c77d00bd,
in-memory host-amendedSHA6b2f73d8a43b7369528a1111f9bd1b04f67171a674b0007cce3dc410b92ecef6,
wrapperSHA1d787715965e41d2565c8d119c301716b7df7ff7f77a209955014637acd7c563;
external monitorSHAf33f1229d8b7c012e9b69005e4dc3d566aed521bd7fa2e400355785bb285f627.
Diffs, launch binding, and boundary controls are archived.

A3 guards: CPU-only,nice19,math1,aggregate<=3threads,RSS<6GB,MemAvailable>=12GiB,
free disk>=40GB,output<500MB,CPU<=85%,load108.8/3independent5s windows,and4h wall.
The4h A3 cap is the owner budget-only amendment. xbabe1 has no93.8% disk-fraction stop;
that threshold applies to fallback xbabe2. No guard triggered. Claims are H194_attempt3.claim
and H194_claim.txt, with approximately10-minute progress under flock. Protected flywheelPID2297929
and H199/H200/H201 files/processes were untouched. Never GPU,xbabe0,or xbabe3.

The numerical audit's sorted-index representation correction and archive live-log exclusion
are zero-update validation/packaging repairs; original science and saved arrays were unchanged.
Both failed auxiliary records are retained. Final native/numerical audits and actual-version
S3all-member checks pass. No scientific retry follows these audits. Root README census remains
outside H194; only an identical clean-main census failure may use the owner's explicit waiver.
