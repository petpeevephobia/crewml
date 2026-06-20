"""Agent 1: profile CSV → LLM plan → pandas apply → cleaned CSV + report."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

AGENT_ID = "data_cleaning"
INPUT_CSV = "data.csv"
OUTPUT_CSV = "01_agent_clean/cleaned.csv"
REPORT_PATH = "01_agent_clean/cleaning_report.md"
SAMPLE_ROWS = 15

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
_SCHEMA_PATH = _DIR / "agent_1_prompt.json"

SYSTEM_PROMPT = """You are a data-cleaning agent for an ML pipeline.

You receive a JSON profile of a CSV (shape, dtypes, null counts, numeric summary, sample rows).
Identify nulls, sentinel values, outliers, and wrong dtypes.

Respond with ONLY valid JSON (no markdown fences) using this structure:
- "issues": list of problems found (column, type, detail, severity)
- "transformations": ordered list of fixes your code will run (see schema example)
- "report_markdown": markdown report of what you found and what you changed

Allowed transformation actions:
- replace_sentinels: {"action","column","from":[strings],"to":null}
- parse_european_decimal: strip units, comma decimals → numeric (column)
- astype: {"action","column","dtype":"float"|"int"|"str"}
- clip_outliers: {"action","column","method":"iqr","factor":3}

Use exact column names from the profile. Prefer fixing values over dropping rows."""


def build_profile(csv_path: str, sample_rows: int = SAMPLE_ROWS):
    """Load CSV and build a small summary for the LLM (not the full file)."""
    df = pd.read_csv(csv_path, low_memory=False)
    null_counts = df.isna().sum()
    profile = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": null_counts.astype(int).to_dict(),
        "null_pct": (null_counts / len(df) * 100).round(2).to_dict(),
        "sample_rows": df.head(sample_rows).fillna("").astype(str).to_dict(orient="records"),
    }
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        profile["numeric_summary"] = numeric.describe().round(2).to_dict()
    return df, profile


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


def _parse_european_decimal(series: pd.Series) -> pd.Series:
    as_str = series.astype(str)
    extracted = as_str.str.extract(r"([\d]+[,.]?\d*)", expand=False)
    extracted = extracted.str.replace(",", ".", regex=False)
    return pd.to_numeric(extracted, errors="coerce")


def apply_transformations(df: pd.DataFrame, transformations: list) -> tuple[pd.DataFrame, list[str]]:
    log = []
    for step in transformations or []:
        action = step.get("action")
        col = step.get("column")
        if col not in df.columns:
            log.append(f"skip {action}: unknown column {col!r}")
            continue

        if action == "replace_sentinels":
            from_vals = step.get("from", [])
            to_val = step.get("to")
            replacement = np.nan if to_val is None else to_val
            before = df[col].isin(from_vals).sum()
            df[col] = df[col].replace(from_vals, replacement)
            log.append(f"replace_sentinels on {col}: {before} values → missing")

        elif action == "parse_european_decimal":
            before_nulls = df[col].isna().sum()
            df[col] = _parse_european_decimal(df[col])
            log.append(
                f"parse_european_decimal on {col}: "
                f"nulls {before_nulls} → {df[col].isna().sum()}"
            )

        elif action == "astype":
            dtype = step.get("dtype", "float")
            if dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df[col] = df[col].astype(str)
            log.append(f"astype on {col} → {dtype}")

        elif action == "clip_outliers":
            factor = float(step.get("factor", 3))
            series = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - factor * iqr, q3 + factor * iqr
            clipped = series.clip(lower=low, upper=high)
            n_changed = int(((series != clipped) & series.notna()).sum())
            df[col] = clipped
            log.append(f"clip_outliers on {col} (IQR×{factor}): {n_changed} values capped")

        else:
            log.append(f"skip unknown action: {action}")

    return df, log



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


def run(call_llm, input_path: str | None = None) -> dict:
    """Profile → LLM JSON plan → apply on full CSV → write outputs."""
    input_path = str(_ROOT / input_path) if input_path else str(_ROOT / INPUT_CSV)
    df, profile = build_profile(input_path)

    schema_example = _SCHEMA_PATH.read_text(encoding="utf-8")
    system = f"{SYSTEM_PROMPT}\n\nExample output shape:\n{schema_example}"
    user = json.dumps({"data_profile": profile}, indent=2)

    raw = call_llm(system, user)
    spec = _parse_llm_json(raw)

    df_clean, apply_log = apply_transformations(df.copy(), spec.get("transformations"))

    report = spec.get("report_markdown", "# Data cleaning report\n")
    report += f"\n\n## Run stats\n\n- Input rows: {len(df):,}\n- Output rows: {len(df_clean):,}\n"
    if spec.get("issues"):
        report += "\n## Issues identified\n\n"
        for issue in spec["issues"]:
            report += (
                f"- **{issue.get('column')}** ({issue.get('type')}, "
                f"{issue.get('severity')}): {issue.get('detail')}\n"
            )
    report += "\n## Steps applied (code)\n\n"
    for line in apply_log:
        report += f"- {line}\n"

    out_csv = _ROOT / OUTPUT_CSV
    out_report = _ROOT / REPORT_PATH
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(out_csv, index=False)
    out_report.write_text(report, encoding="utf-8")

    result = {
        "agent": AGENT_ID,
        "input": input_path,
        "cleaned_csv": str(out_csv),
        "report": str(out_report),
        "rows_in": len(df),
        "rows_out": len(df_clean),
    }

    # Call the helper to update the central state
    update_state(AGENT_ID, result)

    return result