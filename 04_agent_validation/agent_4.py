"""Agent 4: Load winning model from state.json -> cross-validate -> LLM trust verdict -> state.json"""

import json
from pathlib import Path

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
STATE_FILE = _ROOT / "state.json"
INPUT_CSV = _ROOT / "01_agent_clean" / "cleaned.csv"
N_FOLDS = 5

# Maps the model name strings Agent 3 wrote into state.json
# to the actual sklearn classes, with the same hyperparameters Agent 3 used.
MODEL_REGISTRY = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
}

SYSTEM_PROMPT = """You are the Validation Agent in an AutoML pipeline.

Agent 3 selected a winning model based on a single train/test split.
You now have {n_folds}-fold cross-validation results for that same model.

Original test metrics (single split):
{original_metrics}

Cross-validation results ({n_folds} folds):
{cv_results}

Write a concise, plain-English verdict (2–3 paragraphs) covering:
1. Whether the cross-validation scores are consistent with the original test metrics (stable vs. overfit/underfit).
2. Whether the variance across folds is acceptable or concerning.
3. A final recommendation: deploy as-is, tune further, or collect more data.

Return ONLY plain text."""



def _load_state() -> dict:
    if not STATE_FILE.exists():
        raise FileNotFoundError("state.json not found. Run Agents 1-3 first.")
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)



def _pick_winner(state: dict) -> tuple[str, dict]:
    """
    Reads the comparison_table from Agent 3'soutput and returns
    the model name with the highest R2_Score.
    """
    comparison = state.get("model_selection", {}).get("comparison_table", {})
    if not comparison:
        raise ValueError("No comparison_table in state.json. Run Agent 3 first.")

    # Pick the model with the best R2_Score. Same logic that Agent 3 used
    # print(comparison)
    winner_name = max(comparison, key=lambda name: comparison[name]["R2_Score"])
    return winner_name, comparison[winner_name]



def _prepare_data(csv_path: Path) -> tuple:
    """
    Mirrors the exact preprocessing steps Agent 3 used to the
    validation is a fair comparison against the same feature space
    """
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # Drop high-cardinality test cols (>50 unique values) - same threshold as Agent 3
    object_cols = df.select_dtypes(include=["object"]).columns
    high_cardinality = [col for col in object_cols if df[col].nunique() > 50]
    df = df.drop(columns=high_cardinality)

    df = pd.get_dummies(df, drop_first=True)

    target_col = "price_in_euro"
    if target_col not in df.columns:
        raise ValueError(f"Target columns '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y



# root mean squared error to make each model comparable to one another
def _rmse_scorer(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5



def update_state(agent_id: str, new_data: dict):
    """Same safe read-modify-write helper used by every other agent"""
    data = {}
    if STATE_FILE.exists() and STATE_FILE.stat().st_size > 0:
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass
            
    data[agent_id] = new_data
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)



def run(call_llm, input_path: str | None = None) -> dict:
    # 1. Load state and identify the winning model
    print("\tReading state.json for winning model ...")
    state = _load_state()
    winner_name, original_metrics = _pick_winner(state)
    print(f"\tWinner identified: {winner_name}")

    # 2. Prep data with the same steps Agent 3 used
    print("\tPreparing dataset ...")
    csv_path = _ROOT / input_path if input_path else INPUT_CSV
    X, y = _prepare_data(csv_path)

    # 3. Instantiate the winnong model from the registry
    if winner_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Winner '{winner_name}' not in MODEL_REGISTRY. "
            f"Add it to agent_4.py to validate it."
        )
    model = MODEL_REGISTRY[winner_name]

    # 4. Run cross-validation with two scorer simultaneaously.
    # cross-validate passes multiple scoring functions in one call (instead of running the CV loop twice)
    print(f"\tRunning {N_FOLDS}-fold cross-validation on {winner_name} ...")
    rmse_scorer = make_scorer(_rmse_scorer, greater_is_better=False)

    cv_results = cross_validate(
        model, X, y,
        cv=N_FOLDS,
        scoring={
            "r2": "r2",
            "rmse": rmse_scorer
        },
        n_jobs=-1       # use all CPU cores - loss metrics are returned negative so that "higher is always better" holds for all scorers internally.
    )

    # cross_validate returns negative scores for loss metrics to follow sklearn convention
    # Flip the sign back so the numbers makes sense to humans and LLM
    fold_r2_scores = cv_results["test_r2"].tolist()
    fold_rmse_scores = (-cv_results['test_rmse']).tolist()

    cv_summary = {
        "model": winner_name,
        "n_folds": N_FOLDS,
        "fold_r2_scores": [round(v, 4) for v in fold_r2_scores],
        "mean_r2": round(float(np.mean(fold_r2_scores)), 4),
        "std_r2": round(float(np.std(fold_r2_scores)), 4),      # A low std means the model generalizes consistently; a high std means performance is dataset-slice-dependent
        "fold_rmse_scores": [round(v, 2) for v in fold_rmse_scores],
        "mean_rmse": round(float(np.mean(fold_rmse_scores)), 2),
        "std_rmse": round(float(np.std(fold_rmse_scores)), 2)
    }

    # 5. Pass everything to LLM for a verdict
    print("\tAsking LLM for trust verdict ...")
    prompt = SYSTEM_PROMPT.format(
        n_folds=N_FOLDS,
        original_metrics=json.dumps(original_metrics, indent=2),
        cv_results=json.dumps(cv_summary, indent=2)
    )
    verdict = call_llm(prompt, "Write the validation verdict.")

    # 6. Package
    print("\tWriting results to state.json ...")
    agent_output = {
        "winner_model": winner_name,
        "original_test_metrics": original_metrics,
        "cross_validation": cv_summary,
        "verdict": verdict.strip()
    }
    update_state("validation", agent_output)

    return agent_output





# # TO TEST AGENT 4 ONLY
# def _load_dotenv(path=".env"):
#     if not os.path.isfile(path):
#         return
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#") or "=" not in line:
#                 continue
#             key, _, value = line.partition("=")
#             os.environ.setdefault(key.strip(), value.strip())
# _load_dotenv()



# if __name__ == "__main__":
#     from openai import OpenAI
#     import os

#     def call_llm(system, user):
#         client = OpenAI(
#             api_key=os.getenv("DASHSCOPE_API_KEY"),
#             base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
#         )
#         completion = client.chat.completions.create(
#             model="qwen3.6-plus",
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user},
#             ],
#         )
#         return completion.choices[0].message.content

#     result = run(call_llm)
#     print(result)