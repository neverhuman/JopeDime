#!/usr/bin/env python3
"""Write the probe's 2,689 context ids in probe order (selection_reached, all 10 types) for train_ta.py."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/jopedime_probe_exp/20261007T1710Z/code")
from probe import load
D = load(sys.argv[1]); idx = np.where(D["full"] & ~np.isnan(D["fit"]))[0]
json.dump([D["cids"][i] for i in idx], open(sys.argv[2], "w")); print(len(idx))
