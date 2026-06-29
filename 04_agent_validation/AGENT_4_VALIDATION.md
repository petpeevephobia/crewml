# Agent_4: Validation Agent

## Role

The Validation Agent is responsible for stress-testing the winning model chosen by Agent 3. It runs cross-validation between the output of Agent 3 and verdict of Agent 4, computes holdout metrics, and uses an LLM to write a plain-English verdict on whether the result is trustworthy or needs more work.

## Workflow

1. **Load**: Reads `state.json` to identify the winning model from Agent 3's `comparison_table`. Reads `01_agent_clean/cleaned.csv` as the input dataset.
2. **Prepare**: Drops high-cardinality text columns, one-hot encodes the remainder, and splits into features and target (`price_in_euro`)._load_
3. **Cross-Validate**: Runs 5-fold cross-validation on the winning model and records mean R² and mean RMSE across folds.
4. **Analyse**: Passes the cross-validation results plus the original test metrics to the LLM, which writes a trust verdict: is the model stable, overfit, or in need of more work?
5. **Sync**: Writes the cross-validation scores and the LLM verdict to `state.json` under the `validation` key.

## Agent 3 already measures the metrics of every model. Why measure them again via Agent 4?

Even if Agent 3 calculates every metric, the role of Agent 4 is to independently audit and formalise the result. That is namely called *independent verification*. This way, Agent 4 can further evaluate the winning model without hallucination caused by prior agent processes, hence preventing Confirmation bias.

Therefore, the process without Agent 4 would likely produce an unreliable ML pipeline for the user. If Agent 3 encounters a bug or a silent failure (e.g., it reports a "high score" because it inadvertently tested on training data, a classic leakage error), the system will proceed to production because there is no secondary, objective entity to catch the discrepancy.

Agent 3 answers "Which model is the best?"

Agent 4 answers "Is the winner actually safe/good enough?"

## What is Agent 4 doing a cross-validation with?

The agent tests the model's performance against unseen data, which refers to data that the model was not permitted to access or "see" during its training phase. If the models were evaluated using the same data used for training them, the results would be misleadingly optimistic because the model has essentially memorised patterns there. So with unseen data, Agent 4 measures the winning model's ability to generalise to new, real-world scenarios.

## Why RMSE and not MSE?

RMSE is the root version of the MSE, so that the value can be use in the context of business. That way, the quality of the model measured can be more effectively communicated to non-techncial teams. Additionally, by using RMSE, Agent 4 ensures that if a model makes a few wildly inaccurate predictions, the error metric will increase significantly. The Agent's failure condition will be triggered as a result, which will prevent a substandard model from being approved.

**Example**
| Car | Actual Price ($k) | Model A Prediction ($k) | Model B Prediction ($k) |
|-----|------------------|--------------------------|-------------------------|
1 | 20 | 21 | 20
2 | 25 | 24 | 25
3 | 30 | 31 | 30
4 | 22 | 21 | 22
5 | 28 | 29 | 50 (Wildly inaccurate)

If we calculate the metrics:

Mean Absolute Error (MAE)
- Model A: (1+1+1+1+1) / 5 = 1.0
- Model B: (0+0+0+0+22) / 5 = 4.4
Result: Model B's error is **4.4 times worse** than Model A.

Root Mean Square Error (RMSE)
- Model A: √(1^2+1^2+1^2+1^2+1^2) / 5) = √(1) = 1.0
- Model B: √(0^2+0^2+0^2+0^2+22^2) / 5) = √(484 / 5) = √(96.8) = 9.84
Result: Model B's error is now nearly **10 times worse** than Model A.
