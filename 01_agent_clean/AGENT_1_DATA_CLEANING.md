# Agent_1: Data Cleaning Agent
## Role
The Data Cleaning Agent is responsible for the initial processing of the raw dataset. Its primary objective is to prepare the raw automotive CSV for further analysis by handling missing values and ensuring data quality.

## Workflow
1. **Load**: Reads the raw input dataset (e.g., germany-used-cars-dataset-2023).
2. **Clean**: Performs operations such as removing null values (`dropna()`) and standardizing data formats.
3. **Output**: Saves the processed data to `01_agent_clean/cleaned.csv` and generates a `cleaning_report.md`.
4. **Sync**: Updates the shared `state.json` with metadata regarding the cleaning process (e.g., input/output row counts).