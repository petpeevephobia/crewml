# Agent_4: Validation Agent

## Role

The Validation Agent is responsible for stress-testing the winning model chosen by Agent 3. It runs cross-validation, computes holdout metrics, and uses an LLM to write a plain-English verdict on whether the result is trustworthy or needs more work.

## Workflow

1. **Load**: Reads `state.json` to identify the winning model from Agent 3's `comparison_table`. Reads `01_agent_clean/cleaned.csv` as the input dataset.
2. **Prepare**: Drops high-cardinality text columns, one-hot encodes the remainder, and splits into features and target (`price_in_euro`).
3. **Cross-Validate**: Runs 5-fold cross-validation on the winning model and records mean R² and mean RMSE across folds.
4. **Analyse**: Passes the cross-validation results plus the original test metrics to the LLM, which writes a trust verdict: is the model stable, overfit, or in need of more work?
5. **Sync**: Writes the cross-validation scores and the LLM verdict to `state.json` under the `validation` key.

