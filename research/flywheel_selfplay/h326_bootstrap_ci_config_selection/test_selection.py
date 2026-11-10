import numpy as np
import pytest
from selection import choose, decide, require_grid_cache, CacheCoverageError

CONFIGS = [(t,k,a) for t in (.05,.1,.15) for k in (5,6,7) for a in (2,3)]


def matrices(value):
    out = [np.zeros((18, 13 + i)) for i in range(4)]
    for x in out:
        x[1] = value
    return out


def test_accepts_uniform_strict_improvement():
    r = choose(matrices(.003), CONFIGS, [10]*18, 0)
    assert r['accepted'] and r['selected_index'] == 1
    assert r['paired_median'] == .003 and r['lcb95'] == .003


@pytest.mark.parametrize('value', [0., .001999, -.003])
def test_ties_small_or_negative_improvement_abstain(value):
    r = choose(matrices(value), CONFIGS, [10]*18, 0)
    assert r['abstained'] and r['selected_index'] == 0


def test_exact_margin_is_accepted_with_positive_lcb():
    assert choose(matrices(.002), CONFIGS, [10]*18, 0)['accepted']


def test_h303_tie_prefers_fewer_switches_then_conservative_config():
    r = choose([np.zeros((18, 5)) for _ in range(4)], CONFIGS, [9]*18, 0)
    assert CONFIGS[r['candidate_index']] == (.05, 7, 3)
    assert r['abstained']
    switches=[9]*18;switches[3]=0
    assert choose([np.zeros((18,5)) for _ in range(4)],CONFIGS,switches,0)['candidate_index']==3


def test_paired_lcb_can_refuse_noisy_positive_point():
    x = matrices(.003)
    for a in x:
        a[1,:a.shape[1]//2] = -.1
    r=choose(x,CONFIGS,[10]*18,0)
    assert r['abstained'] and r['lcb95'] <= 0


def test_selected_only_cache_is_rejected_before_selection():
    with pytest.raises(CacheCoverageError,match='MISSING_GRID_OOF'):
        require_grid_cache(np.zeros(3,dtype=np.int32),['a','b','c'],['a','b','c'],CONFIGS,CONFIGS)


def test_cache_context_closure_and_slot_range():
    with pytest.raises(CacheCoverageError,match='CONTEXT_CLOSURE'):
        require_grid_cache(np.zeros((18,2),np.int32),['a','outer'],['a','b'],CONFIGS,CONFIGS)
    with pytest.raises(CacheCoverageError):
        require_grid_cache(np.full((18,2),8,np.int32),['a','b'],['a','b'],CONFIGS,CONFIGS)
    assert require_grid_cache(np.zeros((18,2),np.int32),['a','b'],['a','b'],CONFIGS,CONFIGS).shape==(18,2)


def test_outer_rule_single_arm_seed_and_abstain_gate():
    good=dict(per_cell=[.003]*9,median=.003,lcb95=.0001)
    g=dict(BASE=[.05]*9,BSEL=[.05]*9);h=dict(BASE=[.1]*9,BSEL=[.1]*9)
    assert decide(good,g,h,29)['verdict']=='ADVANCE'
    r=decide(good,g,h,30);assert r['verdict']=='BSEL_NOT_ADVANCED' and r['label']=='ABSTAIN_DOMINANT'
    one=dict(per_cell=[.01,-.01,-.01]*3,median=.003,lcb95=.0001)
    assert decide(one,g,h,0)['verdict']=='BSEL_NOT_ADVANCED'
    assert decide(dict(good,lcb95=0),g,h,0)['verdict']=='BSEL_NOT_ADVANCED'
    assert decide(dict(good,median=.0019),g,h,0)['verdict']=='BSEL_NOT_ADVANCED'
