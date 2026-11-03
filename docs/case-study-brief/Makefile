# Case-study briefs (not the white paper).
#   make            the plain-language winners brief (brief.pdf), from cross-validated evidence
#   make technical  the full technical campaign record (campaign-technical.pdf), all 14 cases
BRIEF_WINNERS ?= 09 02 12
BRIEF_CASES ?= 03 09 02 01 07
.PHONY: all winners-data pdf technical-data technical
all: pdf
winners-data:
	BRIEF_WINNERS="$(BRIEF_WINNERS)" ./build-winners.sh
pdf: winners-data
	latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet brief.tex
technical-data:
	BRIEF_CASES="$(BRIEF_CASES)" ./build-data.sh
technical: technical-data
	latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet campaign-technical.tex
