# Major-release PR inventory

Local 1.0.0 preparation on main `a3aeb34fae3c7ff7723bdabf0b771104dfbf8bb0`, captured 2026-10-08 07:30 UTC.
The last published release is [`v0.2.0`](https://github.com/neverhuman/JopeDime/releases/tag/v0.2.0),
peeled commit `629e69d865a674222e00224cebf45c1ffddc8dea`. This list includes every GitHub-merged PR whose merge commit lies in that tag-to-main range.
It contains 127 merged PRs. The owner-authorized #249 squash is recorded separately because GitHub reports that PR as closed without a native merge.

The [changelog](../../CHANGELOG.md) gives the release behavior and limits. This inventory records source inclusion;
it does not establish receiving-main, live-training or serving qualification. Final notes must be refreshed after the queue drains.

## Merged since v0.2.0

| PR | Change | Merge commit |
| --- | --- | --- |
| [#126](https://github.com/neverhuman/JopeDime/pull/126) | [kernel2] PR 01: compatibility freeze before any kernel change | `ab48e389c9951f87e158b5f52930f4d2c9b6d2f4` |
| [#127](https://github.com/neverhuman/JopeDime/pull/127) | [kernel2] PR 02: equal-budget search ablation harness and a bit-exact parity test | `f65026dd54ba29ef3e7e990b51b15eb42652c089` |
| [#128](https://github.com/neverhuman/JopeDime/pull/128) | [kernel2] PR 03: fault classes for the scalar evaluator, values unchanged | `3a32d780814d5536489e9378335fb81e2808f9b2` |
| [#129](https://github.com/neverhuman/JopeDime/pull/129) | lambda: the frozen teacher v2 search engine and Lambda worker, their Python custody manifest and the import records | `6e0ae2491cd4e7918c4d8b07aa211c83376c6764` |
| [#130](https://github.com/neverhuman/JopeDime/pull/130) | [kernel2] PR 04: per-stage fault counts beside unchanged stage outputs | `9146cbcc26b609fb8f59788088f6c87cc60bade2` |
| [#131](https://github.com/neverhuman/JopeDime/pull/131) | [kernel2] PR 05: a static scope and width check for programs, library only | `dd5ccf3f19c3314f5ef9bf34ccf362422650cce7` |
| [#132](https://github.com/neverhuman/JopeDime/pull/132) | [kernel2] PR 06: the fitted-width contract, with checked prediction | `d1a1d21f820d5b1867bb55cb90c42e18cd604a2d` |
| [#133](https://github.com/neverhuman/JopeDime/pull/133) | [kernel2] PR 08: opt-in fault policies for the library search | `e1383b378e8e611835c6c9e0da7c8022a1b1e46e` |
| [#134](https://github.com/neverhuman/JopeDime/pull/134) | [kernel2] X3 — fitted C++17 runtime revision 3 | `0616a93ef13838625d54f43a315e675738932a74` |
| [#135](https://github.com/neverhuman/JopeDime/pull/135) | [kernel2] WP-2.1, WP-2.2, E1 — declare the capability manifest and the search profiles | `ad0488a073c50320fcd1e6aed842dd7f650656b3` |
| [#136](https://github.com/neverhuman/JopeDime/pull/136) | [kernel2] WP-2.3 to WP-2.5, X5 — settle the manifest drift, report export support and gate closure | `e709ed90e0890a619b91fdc2fe395439b20ffb65` |
| [#138](https://github.com/neverhuman/JopeDime/pull/138) | [kernel2] B00 — extend the historical compatibility corpus and golden gate | `3e27963ee79ffaa8904c902bfbb6e9a7d6c514d4` |
| [#139](https://github.com/neverhuman/JopeDime/pull/139) | [kernel2] B01 — define ordered-axis admission | `47c3eb8934813e68fcf0a36bf59024e25b984f79` |
| [#140](https://github.com/neverhuman/JopeDime/pull/140) | [kernel2] B03 — complete expression recipe export | `f05eca57e53cc2834c21152cca3db40dfc26fb97` |
| [#141](https://github.com/neverhuman/JopeDime/pull/141) | [kernel2] B04 — reserve version-2 operator identities | `a3bdc15eaf3db35642e95eb145d713daa45d290f` |
| [#142](https://github.com/neverhuman/JopeDime/pull/142) | [kernel2] B04 follow-up - scope active wire_v2 audit exception | `d9c5dca9bc5941f6c06b560ca746278fe5d8a482` |
| [#143](https://github.com/neverhuman/JopeDime/pull/143) | [kernel2] Prove GP tree round-trips before emitted parity | `e7bd5ef9f5801cd8a552aa4fa2dd96304aa9947d` |
| [#144](https://github.com/neverhuman/JopeDime/pull/144) | Prove GP wire recipe and genome form conversions | `2b972b044604d2a52b786cfe684546da7bd40f83` |
| [#145](https://github.com/neverhuman/JopeDime/pull/145) | [kernel2] Prove generated C++17 numeric ABI boundaries | `e2d3d6e69e9a6b83699db246d69870193ec38e67` |
| [#146](https://github.com/neverhuman/JopeDime/pull/146) | [kernel2] B02 — guard zero-width Roll input; mint evaluator identity | `5f1cd6714225e5b3cd520f4e8de681daaef88c6f` |
| [#149](https://github.com/neverhuman/JopeDime/pull/149) | [kernel2] B03 — opt-in C++17 expression training recipes | `0b3a0aa0957f986b92071da513c90f3cf5b192cb` |
| [#150](https://github.com/neverhuman/JopeDime/pull/150) | [kernel2] Propagate B02 width errors in neural GPR example | `d954a557caa2dd83b1bd1e2ebd91ec1750ca90e4` |
| [#151](https://github.com/neverhuman/JopeDime/pull/151) | Prove expression recipe wire and seeded conversion parity | `643a4ed2fe25313039b00cb7d73dc0f63af100ba` |
| [#152](https://github.com/neverhuman/JopeDime/pull/152) | [kernel2] Prove versioned program and model conversion boundaries | `78bdf65922385ed62bbf95cbce1f98bfb9b49064` |
| [#153](https://github.com/neverhuman/JopeDime/pull/153) | [kernel2] E0 — Admit profile children at exact canonical fixed points | `2b14841573e63f819f562b0bfbe2981a9372f65c` |
| [#154](https://github.com/neverhuman/JopeDime/pull/154) | [kernel2] Add version-two expression search operators | `73d3fd77077ecdc2b265e6b42352a32f52b54960` |
| [#155](https://github.com/neverhuman/JopeDime/pull/155) | [kernel2] Add profile-aware pipeline wrappers; mint evaluator identity | `bf43614226d644f9bb426cfc64cb4afff4eb9e81` |
| [#156](https://github.com/neverhuman/JopeDime/pull/156) | [kernel2] Measure faults and refine expression constants on training rows | `add33f04f2960b85dc58145456ffa0fe85c9a7b0` |
| [#157](https://github.com/neverhuman/JopeDime/pull/157) | [kernel2] Bound constant-refinement trials | `5043dd94f4491e0dd0d7127e42a20dbffc1ca893` |
| [#158](https://github.com/neverhuman/JopeDime/pull/158) | [kernel2] Refuse invalid refinement objectives before admission | `bb4a83f5684cf820befdef1c7903395b0093c354` |
| [#161](https://github.com/neverhuman/JopeDime/pull/161) | [kernel2] Admit pipeline refinement work under caller control | `601a0aacd7acbcfee100eabc201119125ee87a41` |
| [#159](https://github.com/neverhuman/JopeDime/pull/159) | build(deps): bump astral-sh/setup-uv from 10.1.0 to 10.2.0 | `d6ee3a26bd289339303939f41beb9e9d881c97a3` |
| [#160](https://github.com/neverhuman/JopeDime/pull/160) | build(deps): bump taiki-e/install-action from 2.87.18 to 2.87.22 | `334fb9cb43834b1b278286526e13392bccfefaf2` |
| [#162](https://github.com/neverhuman/JopeDime/pull/162) | [kernel2] Export fitted C++17 from pipeline runs | `fec8963068b32d641096007ad37d36afb64be612` |
| [#163](https://github.com/neverhuman/JopeDime/pull/163) | [kernel2] Coordinate bounded profile search | `c40141b53ecb71a7d48b912e3a1f8ce2baf180ea` |
| [#164](https://github.com/neverhuman/JopeDime/pull/164) | [kernel2] Expose profile search and v4 CLI evidence | `ec5894c683089aecd8723d9fc1310c321263e563` |
| [#165](https://github.com/neverhuman/JopeDime/pull/165) | [kernel2] Bind v4 recipe artifacts to selected manifest | `43a6095c263ba719cb8c28d3718a57f0f4669f12` |
| [#166](https://github.com/neverhuman/JopeDime/pull/166) | [kernel2] Witness shipped-profile element generation closure | `148de22124226c176d92a34817294b45b47874c0` |
| [#167](https://github.com/neverhuman/JopeDime/pull/167) | [kernel2] Freeze versioned inverse-link numerical kernels | `772a797955a6f54394dad6d4b23e3bebaed9a8a7` |
| [#168](https://github.com/neverhuman/JopeDime/pull/168) | [kernel2] Read bounded v2 inverse-link expressions | `aa9e575476d339ed6a9b4112e36002687e56eb6d` |
| [#169](https://github.com/neverhuman/JopeDime/pull/169) | [kernel2] Read stable exponential scalar expressions | `0d030295024b4c7b83c001c993b00da82a35a937` |
| [#170](https://github.com/neverhuman/JopeDime/pull/170) | [kernel2] Certify opt-in v2 numerical kernels | `1251d4ed69d1f5f0d5180ee789de018bbca29d08` |
| [#171](https://github.com/neverhuman/JopeDime/pull/171) | [kernel2] Carry opt-in numerical policy in scalar programs | `f36fc2be83138d2d132335267ee528fb7dff91c0` |
| [#172](https://github.com/neverhuman/JopeDime/pull/172) | [kernel2] Execute explicit scalar numerical policies | `4eb555be7f619456f54276920e40cfe9ec31d171` |
| [#173](https://github.com/neverhuman/JopeDime/pull/173) | [kernel2] Qualify guarded portable exponential helpers | `24a31b9bdda62b7b76dae42f3acb7cb6fad9fd51` |
| [#174](https://github.com/neverhuman/JopeDime/pull/174) | [kernel2] Execute qualified versioned exponential calls | `bb756bcbd1224ffa1e5ca289611a2d377faadd1a` |
| [#175](https://github.com/neverhuman/JopeDime/pull/175) | [test] Separate synthetic executor preparation from authorization | `b036848460f3a2887fcdc6120db97152a3c01490` |
| [#176](https://github.com/neverhuman/JopeDime/pull/176) | [kernel2] Add causal integer difference kernels | `2e75f00cc8abe2323fa6e3c5e2d0965c35b68bc3` |
| [#179](https://github.com/neverhuman/JopeDime/pull/179) | [ci] Keep CI evidence artifacts for 7 days instead of 30 | `87b81928deeed097099ed1e0d4d718d0bc26ceaf` |
| [#181](https://github.com/neverhuman/JopeDime/pull/181) | [test] Bound daemon state polling across request timeouts | `6df43201750b50ffcee908333e2f90dee8672829` |
| [#183](https://github.com/neverhuman/JopeDime/pull/183) | [kernel2] Add Savitzky–Golay row filter; mint evaluator identity | `e043aebfd9280902a24b67e0c881b4cf10e1ebd3` |
| [#182](https://github.com/neverhuman/JopeDime/pull/182) | [kernel2] Route causal difference programs and ordered models | `829c83aa95fd5e041eb9f4d096bdb268b24d271b` |
| [#184](https://github.com/neverhuman/JopeDime/pull/184) | [kernel2] Add sparse Goertzel bank; mint evaluator identity | `8773054f7a495800c889b7de10607276199a085d` |
| [#187](https://github.com/neverhuman/JopeDime/pull/187) | [test] Synchronize the lost-ACK deadline fixture | `4183d24ae5832c114d0194d4a6a7128647027782` |
| [#186](https://github.com/neverhuman/JopeDime/pull/186) | [kernel2] Fit and preserve v2 scalar ridge models | `b1d91ac0af6e0c24e7081ddca33b30fde3258891` |
| [#188](https://github.com/neverhuman/JopeDime/pull/188) | [kernel2] Add opt-in epsilon-lexicase parent selection; mint evaluator identity | `bdcdb0aace7cb0321598064258189183a973d453` |
| [#185](https://github.com/neverhuman/JopeDime/pull/185) | [kernel2] Export ordered difference recipes and fitted C++ | `b80a8120d50c5e6023306025a585d252eaba26c4` |
| [#191](https://github.com/neverhuman/JopeDime/pull/191) | [kernel2] Admit bounded causal difference Experimental search | `14a7ce05b56be89f759373f5371bc06ee1402b44` |
| [#189](https://github.com/neverhuman/JopeDime/pull/189) | [kernel2] Add certified fast ridge scoring; mint evaluator identity | `46b7a48c9c2de707c9aeaa3c01079fe40c052126` |
| [#192](https://github.com/neverhuman/JopeDime/pull/192) | [kernel2] Add causal cumulative integration and ordered exports | `79a31f7650b5199e636524ac85ec58af7e09ae43` |
| [#194](https://github.com/neverhuman/JopeDime/pull/194) | [ci] Preserve full-scratch golden test deadline | `0b44783bf3cd52d2cc7528aaa1227b80257eb328` |
| [#190](https://github.com/neverhuman/JopeDime/pull/190) | [kernel2] Emit scalar-v2 train/predict recipes and fitted C++ | `2df9e0c2d50e3f80c7e1d643e6c0705f5b505f51` |
| [#177](https://github.com/neverhuman/JopeDime/pull/177) | [kernel2] Prove parameterized recipe conversion bits | `5daf6cd79725c011593d012be1a98a400d6145d8` |
| [#197](https://github.com/neverhuman/JopeDime/pull/197) | [kernel2] Stage a versioned loss x complexity Pareto archive (B18 core, test-only) | `e5caf0071384ad163d4654e8608307f2bc862c31` |
| [#198](https://github.com/neverhuman/JopeDime/pull/198) | [kernel2] Route scalar fits and models through versioned ingress | `ce9a65eac882bf9757e081c402edf78d23638fd0` |
| [#195](https://github.com/neverhuman/JopeDime/pull/195) | [kernel2] Stage bounded LM constant fitting with a dual-number Jacobian (B17 core, test-only) | `62deb9cc795b3fa13351c6f60879450ba3919d89` |
| [#201](https://github.com/neverhuman/JopeDime/pull/201) | [kernel2] Route scalar v2 recipes and fitted C++ through versioned models | `10b73a3ca582e39d6d2a8ac99fa6174fe17627ae` |
| [#178](https://github.com/neverhuman/JopeDime/pull/178) | [kernel2] Persist opt-in scalar calibration winners | `5e950f998646201d88761dd480e96d691cec8477` |
| [#196](https://github.com/neverhuman/JopeDime/pull/196) | [kernel2] Add causal irregular-grid backward difference | `bf0cb46c593962795155d2bd61cbef26b120ecd9` |
| [#199](https://github.com/neverhuman/JopeDime/pull/199) | Add train-only GPU budget, output evidence, and encoder pins | `8085e1be52b3fb711b3f041e5d87bea6e36ed32f` |
| [#208](https://github.com/neverhuman/JopeDime/pull/208) | [kernel2] Publish each refined child under its own observation | `52d0f6432899f655b8d905ddb2059858c20ce033` |
| [#207](https://github.com/neverhuman/JopeDime/pull/207) | [kernel2] Prove persisted calibration winner export bits | `0d90435019cae913a335e0fcfe5f01473cc1ba18` |
| [#206](https://github.com/neverhuman/JopeDime/pull/206) | fix(runner): bind PR provenance to base ref (unblocks runner-change PRs) | `df48c3206bbc19cc560a58a4ec40923ec3fd52fd` |
| [#209](https://github.com/neverhuman/JopeDime/pull/209) | chore(clippy): burn down 29 behavior-preserving lint locations | `d8c6aa033e475d176bc296abe0bfcf5fc7e9ec54` |
| [#210](https://github.com/neverhuman/JopeDime/pull/210) | fix(runner): validate base SHA and bound base fetch | `d720906b4b86ce44ba5b3547a450531510d19fbf` |
| [#204](https://github.com/neverhuman/JopeDime/pull/204) | Make S1 teacher selection independent of test scores and cap every attempt | `da67d5bbfb4415f8bfacbbbb7a412f70078f8c02` |
| [#205](https://github.com/neverhuman/JopeDime/pull/205) | fix(runner): record swap=0 in systemd so a daemon-reload cannot unlimit live listeners | `8124c1be8363524412290ee89552ac0e5b654a06` |
| [#211](https://github.com/neverhuman/JopeDime/pull/211) | fix(runner): stale shallow.lock, digest REV validation, fetch-timeout cap | `219d384d20c667a85797f24c6d4e256fc84b6f86` |
| [#212](https://github.com/neverhuman/JopeDime/pull/212) | Fix manifest hash comparison; announce identity-only lineage transition | `a0c2f0163ca027cbc45b7c499e7219ef5bce0cce` |
| [#213](https://github.com/neverhuman/JopeDime/pull/213) | docs(runner): swap pin deploys only after merge, in a post-merge freeze | `d2938302c6cacc176e24beb2f386a96d6f1840a5` |
| [#214](https://github.com/neverhuman/JopeDime/pull/214) | fix(s1): restore two-fold teacher search and materialize fresh G0 | `6bf65b12e9f0e5ec2697720a6119f13ea27d4835` |
| [#215](https://github.com/neverhuman/JopeDime/pull/215) | [test] Synchronize fit fixture identity with served readiness | `de51d99620ee294e0c93d793428ad8fe7cd9b962` |
| [#203](https://github.com/neverhuman/JopeDime/pull/203) | [kernel2] Add fixed-five causal realized path statistics (B16) | `8db19e1ce94ccf1875c911f1e6633813624e172f` |
| [#200](https://github.com/neverhuman/JopeDime/pull/200) | [kernel2] Stage bounded Remez compression for one-variable subtrees (B19 core, test-only) | `3db153f4ece55ac5d34a28949c0b774776463844` |
| [#216](https://github.com/neverhuman/JopeDime/pull/216) | chore(dl): preserve pinned train-only builder source custody | `aac714fd56e28a5d0f1523c3959940ffc6f48a0d` |
| [#217](https://github.com/neverhuman/JopeDime/pull/217) | Make synthetic authority race deterministic under neural load | `988b3f32eb53d72271d8216e438e659f3a7db502` |
| [#202](https://github.com/neverhuman/JopeDime/pull/202) | [kernel2] Stage protected-semantics e-graph simplification (B20 core, test-only) | `1f352bb0b740bffad7f88fc36583659981da59e5` |
| [#220](https://github.com/neverhuman/JopeDime/pull/220) | docs(datasets): record owner ruling ALL-DATA-RESEARCH (all datasets usable for DL research regardless of licence; leak rule unchanged) | `6d752ac06b7d9df9a9aed024b9fe077b2592ff6b` |
| [#219](https://github.com/neverhuman/JopeDime/pull/219) | Fix inherited Clippy findings in receiving libraries | `2ba761e413f3cbe7761d42cad49d3005d61ca241` |
| [#222](https://github.com/neverhuman/JopeDime/pull/222) | fix: reject out-of-range CPU affinity indices | `710a0c1f0ef19c9b69958c1489a62bb3c658e4e4` |
| [#225](https://github.com/neverhuman/JopeDime/pull/225) | test(ci): preserve work limits while stabilizing durable fixture execution | `4a09caa87c2c25605b494a76b55c11a7237af61c` |
| [#218](https://github.com/neverhuman/JopeDime/pull/218) | chore(dl): source custody follow-up — non-raw re-split, group split, publisher and publish tools (365ec3ce) | `fdbdf6827e6a2f9ca097416f734f801074c8d491` |
| [#221](https://github.com/neverhuman/JopeDime/pull/221) | [kernel2] Run kernel2 searches in the equal-budget ablation harness (PR-0) | `54fa12839a6e2744224695e9a0e53500b9a041d9` |
| [#226](https://github.com/neverhuman/JopeDime/pull/226) | fix(gpu): enforce device copy bounds and scalar representations | `ab5ca51cdc5b26b8808550cd00783c7980efcb73` |
| [#228](https://github.com/neverhuman/JopeDime/pull/228) | test(smartcluster): isolate control descriptors from inherited locks | `c0e3f2b78d5a693ae97ec570b96a4d2708b04f43` |
| [#223](https://github.com/neverhuman/JopeDime/pull/223) | [kernel2] Fix GP emitter warnings and signed-zero ties; gate deterministic compilation | `0ac90543237356a4c297a51debb502699785e3c0` |
| [#227](https://github.com/neverhuman/JopeDime/pull/227) | [kernel2] Wire damped least-squares constant refinement into kernel2 (opt-in) | `e0e7f819c6b1644869d19b329cfbf16f40ed4e15` |
| [#230](https://github.com/neverhuman/JopeDime/pull/230) | [kernel2] Publish an opt-in loss x complexity front of the measured programs | `f4357b6bde6f76aa7a288c6cd9566a74694ecda4` |
| [#229](https://github.com/neverhuman/JopeDime/pull/229) | fix(engine): check corpus mapping dimensions and record indices | `b15f803ac29b4d49e3d41e6eb903aacc82dfe770` |
| [#224](https://github.com/neverhuman/JopeDime/pull/224) | chore(dl): strip outer-test targets from published JTT1 payloads (owner ruling 06:52 MT; handoff 4419ff43) | `2e0b7723b56a4edfcde87a875d24555724fc6801` |
| [#231](https://github.com/neverhuman/JopeDime/pull/231) | test(kernel2): prove GP math trees, recipe conversion and protected Tanh folds | `1d0a27acb4bb9357f277ef1b8e6518fe2f6d17fb` |
| [#232](https://github.com/neverhuman/JopeDime/pull/232) | [kernel2] Offer exact-rewrite simplifications of the ranked parents (opt-in) | `9662441e1207e702b18f513d10d96346dd5a12bd` |
| [#233](https://github.com/neverhuman/JopeDime/pull/233) | refactor(campaigns): remove disabled duplicate final CV loop | `c7b810acd3d697c97f15e365120d40ea0011367c` |
| [#234](https://github.com/neverhuman/JopeDime/pull/234) | docs(release): record all Jankurai findings and link the agent contract | `e8ccd65ac15a2832a3eec21ea50f76d74ad027cb` |
| [#236](https://github.com/neverhuman/JopeDime/pull/236) | dl: retain S1 GP HWM evidence through verified create-only publication | `5ef2e10c824c852714eff3859bd5dfefbc063304` |
| [#238](https://github.com/neverhuman/JopeDime/pull/238) | test(gp): certify tiny-tail compositions and emitted fault bits | `ca1f7570acd0ed058365883d8bf45228c2c78d26` |
| [#237](https://github.com/neverhuman/JopeDime/pull/237) | fix(gpu): validate ridge dimensions before backend routing | `fb161588a3e1f9eb982b16bdaf59cb2d80f02062` |
| [#239](https://github.com/neverhuman/JopeDime/pull/239) | dl: publish raw GPU encoding manifests with source-bound cut identity | `4fc1492e4755b0777134a78705feedb0a2f73d30` |
| [#240](https://github.com/neverhuman/JopeDime/pull/240) | Fix mechanical workspace Clippy findings without changing behavior | `38c9ac47b014c0ab80342003b46150f9f950b0f7` |
| [#241](https://github.com/neverhuman/JopeDime/pull/241) | Close remaining workspace Clippy findings within owner-approved scope | `15394471157e6ee36f656e905966098d83a06794` |
| [#242](https://github.com/neverhuman/JopeDime/pull/242) | Start full-main golden tail first without changing coverage | `9585b364f2a8a36d2b06e285ef5a28673b48193e` |
| [#243](https://github.com/neverhuman/JopeDime/pull/243) | Bound the projected GP census by completed observations | `c4d018591cc313030f8003be3663972659609764` |
| [#244](https://github.com/neverhuman/JopeDime/pull/244) | Use eight build workers for full-main CPU families | `cad789d9375dcff4841f568f7d73f3e880b9aadb` |
| [#246](https://github.com/neverhuman/JopeDime/pull/246) | dl: land the raw-route group hook (RAW-GROUP-PR) and the LINEAR-DETERMINED gate | `b486da4518a80d395629a492b96cf3f2c75863e8` |
| [#248](https://github.com/neverhuman/JopeDime/pull/248) | [ci] Cache trusted full-main library compilations | `c40eba2d57d921a5c3ac8c0692ac77c01e2e31e6` |
| [#247](https://github.com/neverhuman/JopeDime/pull/247) | [kernel2] Offer bounded minimax linearizations of one-variable subtrees (opt-in) | `a95a72875d045c773da7f3f6bbc83ec52d1db51f` |
| [#252](https://github.com/neverhuman/JopeDime/pull/252) | [kernel2] Close the B17 and B18 review notes: named refusals and read-back-bounded seeds | `e1d6be9d4c86629612412195b35c3ab519c05ba1` |
| [#253](https://github.com/neverhuman/JopeDime/pull/253) | fix: close rejected sealed-IPC descriptor transfers | `258ac8d4c55b4b5edaabbaadbd0ec5eff206d330` |
| [#250](https://github.com/neverhuman/JopeDime/pull/250) | dl: bind published contexts' outer split and GPU inputs/outputs to their build | `8b7476fdfbf2c4631e5426bfdf62991020b55e0a` |
| [#254](https://github.com/neverhuman/JopeDime/pull/254) | test: Collect teacher handoff security cases independently | `df760a3ed315df45f3ece18b83d94a54833047cf` |
| [#256](https://github.com/neverhuman/JopeDime/pull/256) | dl: publish S1 HWM evidence for G3 and catalog contexts from sha-bound P3 decisions (GP-2) | `3d1d0819ac2026d1e3b9c78cb6fbddf1b754bb77` |
| [#258](https://github.com/neverhuman/JopeDime/pull/258) | test: isolate authority executors and bound PR fixture concurrency | `024f4f040c34f7b20e0f9980c45fb216d4439f21` |
| [#261](https://github.com/neverhuman/JopeDime/pull/261) | docs: justify feasibility probe allocator safety | `9099e98488827740db0d3c2a40b06fed9f05b7d2` |
| [#260](https://github.com/neverhuman/JopeDime/pull/260) | fix: bind score-corpus parsing to a streamed ledger digest | `7e4c3b38b0e3d163a4d9eab149854c0c2cdd5f3b` |
| [#259](https://github.com/neverhuman/JopeDime/pull/259) | fix: verify pins before benchmark and target-maker artifact loads | `8e83c10790f2293a0384c756f57dc97284a7f783` |
| [#255](https://github.com/neverhuman/JopeDime/pull/255) | fix: bound crash recovery and reuse the verifier’s held executable digest | `3d9e675ef9301c3de8253317ec1b12268b18cb81` |
| [#262](https://github.com/neverhuman/JopeDime/pull/262) | fix: pin and seal the embedded fitted runtime before loading | `a3aeb34fae3c7ff7723bdabf0b771104dfbf8bb0` |

## Owner-authorized landing

[#249](https://github.com/neverhuman/JopeDime/pull/249) refuses unknown P3 holdout assignments before admission.
It landed by the owner-authorized squash `8a4b2fdfe00031c635f5d8c5d00fa8cc9039e3cc`; the PR was then closed by pointer.
The squash is an ancestor of this source base. It is included in the notes without claiming a native GitHub merge.

## Pending release queue

The owner’s 2026-10-08 07:30 UTC landing order is below. Each PR remains open at this inventory’s source cut.
Lane A’s full-family-margin correction and green receiving main precede this queue. Preserve pending status until the actual merge is read back.

| Order | PR | Intended change | Full candidate head |
| --- | --- | --- | --- |
| 1 | [#251](https://github.com/neverhuman/JopeDime/pull/251) | Target-tie and LINEAR-DETERMINED admission after each split, using only that context’s outer-train rows. | `ee4eb53a0a7ffc01abc0a7d198e17f35299c5b19` |
| 2 | [#263](https://github.com/neverhuman/JopeDime/pull/263) | Publish owner-designated P3 heldout S1 finals as training only with the pinned ONE-HOLDOUT role map; sealed exclusions stay refused. | `21b1c32799f25b0550f87e4a56b958f2e3fc9dee` |
| 3 | [#257](https://github.com/neverhuman/JopeDime/pull/257) | Train-only feature reduction for fixed-width consumers, including deterministic handling of uninformative inner folds. | `046fea4d499d14d5dc80dd7f03c2744aa9e396a4` |
| 4 | [#265](https://github.com/neverhuman/JopeDime/pull/265) | Share the campaign BIN flock boundary with local FD/ABI/error proofs and document encoder open/OwnedFd ownership; preserve behavior. | `aaa2724e528345ecb20bf91047aa9dbcccdde7e7` |

The separate library-proof PR [#264](https://github.com/neverhuman/JopeDime/pull/264) remains open at
`ba48fb484e5e2a8c2dc9581e4c4024dd70fb39f1`. It is outside the owner’s four-item queue above and is not counted as merged.
Its latest retained CI attempt was cancelled at the existing 45-minute job cap, with partial coverage and timeouts; a healthy exact-head run and independent approval remain required.

No release PR, push, tag or publication is authorized by this local inventory.
