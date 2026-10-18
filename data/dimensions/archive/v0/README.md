# Rubric v0 — superseded, reference only

The dimension configs as they stood before the Phase A alignment (commit `31dd7a0`), kept byte-for-byte for reference. **Never run against this directory**: no experiment used it, and it is not a valid rubric version.

Superseded by `data/dimensions/v1/` because:

- **D1** cited the classic Z (market cap, 1.81 / 2.99 bands), not the Z″ rule the D1 label is computed with.
- **D2** had an ASC citation error (`50-4`), and its MEDIUM level was an analyst-judgement liquidity test with no standard behind it.
- **D3** misstated Item 103: the `$300,000` threshold is environmental-only, and 10% of current assets is a floor for leaving out damages claims.
- **D5** sourced ratings from Item 1A and Item 106, contrary to the Item 1A ≠ Item 9A rule.

## About `ground_truth_labels` in these files

These are **not labels**. They were never hand-verified against the filings or derived from the label rules, and at least one pair contradicts itself. The D2 config calls Carvana FY2022 HIGH ("KPMG AS 2415 going concern qualification"), while a false-positive guard in the same file says Carvana passed ASC 205-40 Step 2, meaning no going-concern qualification.

Treat them as leads for the human labeling pass at most. Real labels go in `labels.json`, from XBRL computation, EDGAR full-text search, or a human reading the filing.
