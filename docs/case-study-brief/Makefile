# Case-study brief (not the white paper). `make` regenerates the data from the evidence and the PDF.
BRIEF_CASES ?= 03 09 02 01 07
.PHONY: all data pdf
all: pdf
data:
	BRIEF_CASES="$(BRIEF_CASES)" ./build-data.sh
pdf: data
	latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet brief.tex
