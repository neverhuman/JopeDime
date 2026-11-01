# Rebuilding retained figures

'`make`' builds the paper from committed figures. To regenerate all figures, use
'`make` figs HISTORICAL_TARGET_INPUTS=INPUT_DIR'. The retained historical table
directory must contain comparisons.csv, summary.csv,
selected-historical-cut.csv and carry-forward-audit.json with the exact input
hashes recorded in the target-progress and target-history provenance files.
Input locations are supplied locally; emitted provenance records neutral names
and content hashes. No data or model payload is fetched by these commands.

The default promotion plot retains an earlier development window. Current
main-text promotion and validation panels come from the live-development and
live-admissions generators; the retained prefix plots remain archival views.
