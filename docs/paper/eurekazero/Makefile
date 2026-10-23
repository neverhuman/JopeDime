# EurekaZero north-star draft. All recipes are bash scripts; nothing here needs Python.
.PHONY: all pdf check self-test submission-check synthetic clean
all: pdf

# Source checks, the typeset north-star PDF and the PDF checks (target/eurekazero-paper/).
pdf:
	bash scripts/build.sh

# Inputs, citations, vocabulary, ledgers, and the synthetic-labelling contract; no TeX needed.
check:
	bash scripts/check.sh

# Every refusal in scripts/check.sh fires on a deliberately broken copy of the sources.
self-test:
	bash scripts/check.sh --self-test

# Refuses while any synthetic element remains. This draft is expected to fail it, by design.
submission-check:
	bash scripts/check.sh --submission

# Rewrites data/synthetic/*.csv and generated/numbers.tex from scripts/synthesize.awk.
synthetic:
	bash scripts/synthesize.sh

# Removes only this build's products; never the whole target directory.
clean:
	bash scripts/build.sh --clean
