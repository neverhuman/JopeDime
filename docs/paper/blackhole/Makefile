# EurekaZero: The Tabular Black Hole (docs/paper/blackhole/).
# Modular XeLaTeX build; leaves docs/paper/eurekazero/ v0.1 and docs/whitepaper/ untouched.
.PHONY: all pdf check self-test clean
all: pdf

# Source checks, XeLaTeX + BibTeX build, and PDF/font/layout checks (target/blackhole-paper/).
pdf:
	bash scripts/build.sh

# Inputs, citations, IP vocabulary scrub, privacy guard, and claim-ledger checks; no TeX needed.
check:
	bash scripts/check.sh

# Verifies that every refusal in scripts/check.sh fires on a deliberately broken fixture.
self-test:
	bash scripts/check.sh --self-test

# Removes only this build's products inside target/blackhole-paper/; never touches target/ wholesale.
clean:
	bash scripts/build.sh --clean
