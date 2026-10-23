# InnovationZero publication. All recipes are bash scripts; nothing here needs Python.
.PHONY: all pdf check self-test clean
all: pdf

# Source checks, three typeset volumes and the PDF checks (target/innovationzero-paper/).
pdf:
	bash scripts/build.sh

# Archive hashes, receipt bindings, derived tables, inputs, citations and vocabulary; no TeX needed.
check:
	bash scripts/check.sh

# Every refusal in scripts/check.sh fires on a deliberately broken copy of the sources.
self-test:
	bash scripts/check.sh --self-test

# Removes only this build's products; never the whole target directory.
clean:
	bash scripts/build.sh --clean
