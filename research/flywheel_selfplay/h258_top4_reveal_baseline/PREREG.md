# H258 (`h258_top4_reveal_baseline`) Preregistration

- **Wave ID**: `h258-top4-reveal-baseline-20261103`
- **Study Path**: `research/flywheel_selfplay/h258_top4_reveal_baseline`
- **Owner**: `rd_argon` (`xbabe2` cores `57-60`, `nice 19`, `<= 4` threads, `RSS <= 4 GiB`, `0` GPU)

## 1. Scientific Question & Hypotheses

H254 replicated H252's Stage-1 `ACTIVE_2PLUS2_HURDLE` vs `GATED_025` (`+0.015519` on `DL-Val`), but red-team review (`watcher 14:05Z`) identified two critical blockers before adoption:
1. Revealed `fitness` IS the scored quantity (`utility = -fitness`, `common_h252.py:137`), so `ACTIVE_2PLUS2_HURDLE` is a test-time 4-true-evaluation search policy under the `top_zgap < 0.25` gate (`0` reveals elsewhere), whereas `GATED_025` reveals 4 **random** candidates (`Elo 1496.34 < R882 1500.00`).
2. The trivial strong equal-budget baseline—**`TOP4_PICK`** ("on gated contexts `top_zgap < 0.25`, reveal the 4 lowest-`r882`-cost candidates and pick `argmin` revealed gated cost `fitness if gate else 2.0`, ties broken by `r882` order; `0` reveals elsewhere")—was never evaluated.

**H258 tests two questions**:
1. **Primary (`Holm m = 2`)**: Does `ACTIVE_2PLUS2_HURDLE` beat `TOP4_PICK` at matched 4-reveal budget on `DL-Val` (`Native91`, `91` contexts / `19` families)?
   - **Secondary (`Holm m = 2`)**: Does `TOP4_PICK` beat `GATED_025` at matched 4-reveal budget on `DL-Val`?
2. **Incumbent Transfer (`r3175` Secondary)**: Do `TOP4_PICK@r3175` and `ACTIVE_2PLUS2_HURDLE@r3175` survive on the flywheel incumbent `round-3175.pt` (`sha256 = 1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`, `DOPE c51a56d0`)?

## 2. Frozen Arms & Equal-Budget Contract

All gated arms use the exact same outcome-free trigger `top_zgap < 0.25` (`compute_r882_top_zgap(rcost) < 0.25`): `4` unique candidate reveals when `top_zgap < 0.25` and `0` reveals (`argmin(rcost)`) when `top_zgap >= 0.25`.

1. **`R882`** (`0` reveals): `argmin(r882_cost)`. Must reproduce `Native91 = 1.1377941898164525` (`91/91` contexts) bit-exact before any other arm is scored.
2. **`GATED_025`** (frozen H252/H254): On `top_zgap < 0.25`, reveal `4` random distinct candidates `idx4 = rng.choice(n, size=4, replace=False)`, apply `sem24` RBF kernel adjustment `rcost + 0.05 * (w @ (y4_cost - mean(y4_cost)))`, pick `argmin`.
3. **`ACTIVE_2PLUS2_HURDLE`** (frozen H252/H254): On `top_zgap < 0.25`, reveal `2` initial random probes `idx_a = idx4[:2]`, compute interim Hurdle + `sem24` posterior, actively reveal `2` unrevealed contenders `chosen_b`, and pick `argmin(cost_act)`.
4. **`TOP4_PICK`** (equal-budget search baseline): On `top_zgap < 0.25`, reveal the `4` lowest-`rcost` candidates `top4 = np.argsort(rcost, kind="stable")[:4]`, score each revealed candidate by its true gated cost `y_rev = fitness if gate else 2.0`, and pick `top4[int(np.argmin(y_rev))]` (ties broken in `rcost` order). On `top_zgap >= 0.25`, `0` reveals and `R882` pick.
5. **`TOP2_RAND2_PICK`** (descriptive equal-budget): On `top_zgap < 0.25`, reveal `top2 = order_rcost[:2]` plus `2` random distinct candidates `[i for i in idx4 if i not in top2][:2]`, and pick `argmin` revealed gated cost (`fitness if gate else 2.0`, tie -> `rcost` order).
6. **`TOP4_PICK_UNGATED`** (descriptive only, **unequal budget**): Reveal `order_rcost[:4]` on **all** contexts regardless of `top_zgap` and pick `argmin` revealed gated cost.
7. **`ACTIVE_2PLUS2_PICK`** (descriptive equal-budget mechanism diagnostic): On `top_zgap < 0.25`, reveal the exact 4 candidates `rev4` chosen by `ACTIVE_2PLUS2_HURDLE` and pick `argmin` revealed gated cost (`fitness if gate else 2.0`, tie -> `rcost` order), isolating probe selection (`rev4` vs `top4`) from `cost_act` shrinkage.
8. **`r3175` Secondary Suite (`R3175`, `GATED_025@r3175`, `ACTIVE_2PLUS2_HURDLE@r3175`, `TOP4_PICK@r3175`, `TOP2_RAND2_PICK@r3175`, `TOP4_PICK_UNGATED@r3175`, `ACTIVE_2PLUS2_PICK@r3175`)**: Identical definitions applied to frozen `round-3175.pt` (`sha256 = 1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`) cost caches (`inputs/r3175-secondary-costs.npz` and `inputs/r3175-train-costs.npz`).

## 3. Seeds, Statistical Protocol & Advance Gate

- **Seeds**: `fold_seed = 258140` (`5` family-grouped `DL-Train` folds across `205` families), evaluation seeds `258101, 258102, 258103`, paired family bootstrap `boot_seed = 258150` (`10,000` draws), `400`-iter MM Bradley-Terry Elo anchored at `R882 = 1500.0`.
- **Decision Rule (`ADVANCE` for `ACTIVE_2PLUS2_HURDLE`, ties = `NO_ADVANCE`)**:
  1. `DL-Val` median family gain `ACTIVE_2PLUS2_HURDLE - TOP4_PICK > 0.0`,
  2. Holm-adjusted (`m = 2`) lower confidence bound `> 0.0`,
  3. `Holm p < 0.05`,
  4. `>= 2/3` evaluation seeds positive on `DL-Val`,
  5. Permuted-fitness negative control (`ACTIVE_2PLUS2_HURDLE` with revealed `fitness`/`logratio`/`gate` permuted within context) `95% CI` covers `0`.
