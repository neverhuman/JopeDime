"""Data-free H352 native LR/round grid check; no new fit or baseline relabel."""
def check(pairs,rankers):
 baseline={(lr,n) for lr in [.03,.05,.1] for n in [50,100,200]}
 treatment={(lr*.5,n*2) for lr,n in baseline};observed={tuple(x) for x in pairs}
 good=rankers==972 and bool(observed) and observed<=baseline and not observed<=treatment
 return dict(status='PASS' if good else 'INCOMPLETE_SPEC_MISMATCH',rankers_verified=rankers,observed_baseline_pairs=[list(x) for x in sorted(observed)],baseline_grid=[list(x) for x in sorted(baseline)],treatment_grid=[list(x) for x in sorted(treatment)],mapping=[dict(baseline=[lr,n],treatment=[lr*.5,n*2],product_preserved=lr*n==(lr*.5)*(n*2)) for lr,n in sorted(observed)],already_all_treatment=bool(observed) and observed<=treatment,treatment_fits=0,capacity_probes=0)
