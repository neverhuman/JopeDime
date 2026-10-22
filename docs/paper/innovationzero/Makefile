.PHONY: all pdf check self-test clean
all: pdf

pdf:
	bash scripts/build.sh

check:
	python3 scripts/check.py

self-test:
	python3 scripts/check.py --self-test

# Only files produced by this paper build are removed; no recursive target cleanup.
clean:
	bash scripts/build.sh --clean
