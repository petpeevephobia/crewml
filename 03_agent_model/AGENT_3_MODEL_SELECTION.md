# Agent_3: Model Selection Agent
## Role
The Model Selection Agent is responsible for training various machine learning models, evaluating their performance, and determining the optimal configuration for the final pipeline based on metrics.

## Workflow
1. **Load/Prepare**: Reads the `featured.csv`, performs final preprocessing (e.g., one-hot encoding, train-test splitting).
2. **Train & Evaluate**: Trains multiple models (e.g., Linear Regression, Random Forest, Gradient Boosting) and calculates performance metrics like R-squared Score and RMSE.
3. **Analyse**: Uses an LLM to evaluate the performance metrics of the models and declare a "winner" based on the comparison table.
4. **Sync**: Writes the final performance comparison and the model analysis/selection rationale to state.json under the `model_selection` key.