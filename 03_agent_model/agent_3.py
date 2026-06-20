"""Agent 3: Load cleaned.csv -> train 3 models -> calculate metrics -> append results to state.json"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import json

# Define paths relative to agent_3.py's location (to keep the project modular)
_DIR = Path(__file__).resolve().parent                # where is agent_3.py?
_ROOT = _DIR.parent                                   # root project location
INPUT_CSV = _ROOT / "01_agent_clean/cleaned.csv"      # input data
STATE_FILE = _ROOT / "state.json"                     # shared agent brain

# agent 3 prompt
SYSTEM_PROMPT = """You are the Model Selection Agent.
We just trained 3 machine learning models to predict car prices. 
Here are their actual performance metrics on the test set:

{metrics}

Write a brief, 1-paragraph analysis justifying which model performed best and why we should select it for the final pipeline.
Return ONLY plain text."""



def update_state(agent_id: str, new_data: dict):
    """
    Safely reads state.json, updates the entry for this agent, 
    and writes the entire state back.
    """
    state_file = _ROOT / "state.json"
    
    # Load existing state or start with empty dict
    data = {}
    if state_file.exists() and state_file.stat().st_size > 0:
        try:
            with open(state_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # File exists but is corrupt/empty; start fresh
            
    # Update state for this specific agent
    data[agent_id] = new_data
    
    # Save back to file
    with open(state_file, 'w') as f:
        json.dump(data, f, indent=2)



# Train 3 models -> evaluate -> update state.json
"""
If developing Agent 3 and want to test its data handling entirely by itself:
`agent_3.run(call_llm, input_path="tests/mock_cleaned_data.csv")`
"""
def run(call_llm, input_path: str | None = None) -> dict:
    # 1. load dataset
    target_path = _ROOT / input_path if input_path else INPUT_CSV
    df = pd.read_csv(target_path)
    df = df.dropna()

    # MEMORY TOO BIG (30.9 GiB): drop text cols with very high cardinality, aka >50 unique values
    object_cols = df.select_dtypes(include=["object"]).columns
    high_cardinality_cols = [col for col in object_cols if df[col].nunique() > 50]
    df = df.drop(columns=high_cardinality_cols)

    # one-hot encode safely the remaining text cols
    df = pd.get_dummies(df, drop_first=True)

    target_col = "price_in_euro"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. train models (42 is always the answer lol)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # 3. measure metrics for every model
        r2 = r2_score(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5           # raise standard MSE to power of 0.5 to get RMSE (root MSE). better to compare against other models
        results[name] = {"R2_Score": round(r2, 4), "RMSE": round(rmse, 2)}

    # 4. inject metrics into LLM prompt template
    # json.dumps() turns our Python dictionary into a cleanly formatted string for the LLM to read
    formatted_prompt = SYSTEM_PROMPT.format(metrics=json.dumps(results, indent=2))

    # 5. call LLM to evaluate the three models
    analysis = call_llm(formatted_prompt, "Analyze these model metrics and declare a winner.")

    # 6. package final payload -> save to state.json
    agent_output = {
        "comparison_table": results,
        "analysis": analysis.strip()
    }

    state = {}
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

    state['Agent_3_Model_Selection'] = agent_output

    # Use the helper function instead of manual open/read/write logic
    update_state("Agent_3_Model_Selection", agent_output)

    return agent_output