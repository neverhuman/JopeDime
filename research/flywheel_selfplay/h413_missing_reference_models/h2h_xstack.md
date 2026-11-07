**Head to head vs our `h364.XSTACK_2L`** (from `tools/ladder.py --focus <id>`; gain = family-equal mean of u(model) − u(XSTACK_2L); RMSE cut = 1 − exp(−gain); clipped T / sign-flip p / LOFO min = the H335 primary procedure; pass = p < 0.025 and LOFO > 0):

| Model | Gain | RMSE cut | Families W/T/L | Sign-flip p | LOFO min | Passes |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `kumo_tabular_small_e8.v1` | +0.2248 | +20.1% | 16/0/3 | 0.009138 | +0.0948 | yes |
| `tabpfn_v3p5_e1.v1` | +0.0280 | +2.8% | 15/0/4 | 0.2026 | +0.0261 | no |
| `mitra_v2_zeroshot.v1` | -0.0012 | -0.1% | 12/0/7 | 0.3613 | -0.0068 | no |
| `tabicl_v2.v1` | -0.0729 | -7.6% | 10/0/9 | 0.6243 | -0.0387 | no |
| `tabdpt_1p3.v1` | -0.0428 | -4.4% | 12/0/7 | 0.4587 | -0.0172 | no |
