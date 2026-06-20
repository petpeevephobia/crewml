"""Agent 2: profile cleaned CSV → LLM feature plan → pandas apply → featured CSV + report."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

AGENT_ID = "feature_engineering"
INPUT_CSV = "01_agent_clean/cleaned.csv"
OUTPUT_CSV = "02_agent_feature/featured.csv"
REPORT_PATH = "02_agent_feature/feature_report.md"
SAMPLE_ROWS = 15
MIN_FEATURES = 3
MAX_FEATURES = 5

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
_SCHEMA_PATH = _DIR / "agent_2_prompt.json"

SYSTEM_PROMPT = f"""You are a feature-engineering agent for an ML pipeline on tabular data.

You receive a JSON profile of a cleaned CSV (shape, dtypes, null counts, numeric summary, sample rows).
Propose {MIN_FEATURES}–{MAX_FEATURES} new derived features that would help predict price or similar targets.

Respond with ONLY valid JSON (no markdown fences) using this structure:
- "features": ordered list of feature definitions (see schema example)
- "report_markdown": markdown summary of your feature strategy

Each feature must include:
- "name": new column name (snake_case, not colliding with existing columns)
- "action": one of the allowed actions below
- "justification": 1–3 sentences explaining why this feature helps modeling
- "pandas_code": a single-line or short snippet showing the intended pandas/sklearn logic (for the report; Python executes via "action")

Allowed actions:
- ratio: {{"action","name","column_a","column_b","epsilon":1e-6}} — column_a / (column_b + epsilon)
- log1p: {{"action","name","column"}} — np.log1p of non-negative values
- sqrt: {{"action","name","column"}} — sqrt of non-negative values
- standardize: {{"action","name","column"}} — z-score (mean 0, std 1) on the column
- difference: {{"action","name","column_a","column_b"}} — column_a - column_b
- product: {{"action","name","column_a","column_b"}} — column_a * column_b
- year_delta: {{"action","name","column","reference_year":2023}} — reference_year - column
- bin: {{"action","name","column","n_bins":5}} — equal-width bins as categorical codes (int)
- pca: {{"action","columns":[...],"n_components":2,"prefix":"pca"}} — adds prefix_1, prefix_2, ... (fit on rows with finite values in those columns)

Prefer ratios, log transforms, age/usage features, and optionally one PCA block over many redundant pairs.
Use exact column names from the profile. Reference columns you create earlier in the same list (e.g. vehicle_age before mileage_per_year)."""


def build_profile(csv_path: str, sample_rows: int = SAMPLE_ROWS):
    """Load CSV and build a compact summary for the LLM."""
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
        profile["numeric_columns"] = list(numeric.columns)
    return df, profile


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


def apply_features(df: pd.DataFrame, features: list) -> tuple[pd.DataFrame, list[str]]:
    log = []
    for feat in features or []:
        action = feat.get("action")
        name = feat.get("name")

        if action == "pca":
            cols = feat.get("columns", [])
            prefix = feat.get("prefix", "pca")
            n_components = int(feat.get("n_components", 2))
            missing = [c for c in cols if c not in df.columns]
            if missing:
                log.append(f"skip pca: unknown columns {missing}")
                continue
            block = df[cols].apply(pd.to_numeric, errors="coerce")
            medians = block.median()
            filled = block.fillna(medians)
            valid = filled.notna().all(axis=1)
            if valid.sum() < n_components:
                log.append(f"skip pca: insufficient valid rows ({valid.sum()})")
                continue
            scaler = StandardScaler()
            scaled = scaler.fit_transform(filled.loc[valid])
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(scaled)
            for i in range(n_components):
                col_name = f"{prefix}_{i + 1}"
                df[col_name] = np.nan
                df.loc[valid, col_name] = components[:, i]
            explained = pca.explained_variance_ratio_.round(3).tolist()
            log.append(
                f"pca on {cols} → {prefix}_1..{n_components} "
                f"(explained variance ratios: {explained})"
            )
            continue

        if not name:
            log.append(f"skip {action}: missing name")
            continue
        if name in df.columns:
            log.append(f"skip {action}: column {name!r} already exists")
            continue

        if action == "ratio":
            a, b = feat.get("column_a"), feat.get("column_b")
            if a not in df.columns or b not in df.columns:
                log.append(f"skip ratio {name}: unknown column(s)")
                continue
            eps = float(feat.get("epsilon", 1e-6))
            denom = pd.to_numeric(df[b], errors="coerce") + eps
            df[name] = pd.to_numeric(df[a], errors="coerce") / denom
            log.append(f"ratio {name} = {a} / ({b} + {eps})")

        elif action == "log1p":
            col = feat.get("column")
            if col not in df.columns:
                log.append(f"skip log1p {name}: unknown column {col!r}")
                continue
            values = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
            df[name] = np.log1p(values)
            log.append(f"log1p {name} from {col}")

        elif action == "sqrt":
            col = feat.get("column")
            if col not in df.columns:
                log.append(f"skip sqrt {name}: unknown column {col!r}")
                continue
            values = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
            df[name] = np.sqrt(values)
            log.append(f"sqrt {name} from {col}")

        elif action == "standardize":
            col = feat.get("column")
            if col not in df.columns:
                log.append(f"skip standardize {name}: unknown column {col!r}")
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            std = values.std()
            if std == 0 or np.isnan(std):
                df[name] = 0.0
            else:
                df[name] = (values - values.mean()) / std
            log.append(f"standardize {name} from {col}")

        elif action == "difference":
            a, b = feat.get("column_a"), feat.get("column_b")
            if a not in df.columns or b not in df.columns:
                log.append(f"skip difference {name}: unknown column(s)")
                continue
            df[name] = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")
            log.append(f"difference {name} = {a} - {b}")

        elif action == "product":
            a, b = feat.get("column_a"), feat.get("column_b")
            if a not in df.columns or b not in df.columns:
                log.append(f"skip product {name}: unknown column(s)")
                continue
            df[name] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")
            log.append(f"product {name} = {a} * {b}")

        elif action == "year_delta":
            col = feat.get("column")
            ref = int(feat.get("reference_year", 2023))
            if col not in df.columns:
                log.append(f"skip year_delta {name}: unknown column {col!r}")
                continue
            df[name] = ref - pd.to_numeric(df[col], errors="coerce")
            log.append(f"year_delta {name} = {ref} - {col}")

        elif action == "bin":
            col = feat.get("column")
            n_bins = int(feat.get("n_bins", 5))
            if col not in df.columns:
                log.append(f"skip bin {name}: unknown column {col!r}")
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            df[name] = pd.cut(values, bins=n_bins, labels=False, duplicates="drop")
            log.append(f"bin {name} from {col} ({n_bins} bins)")

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
    """Profile → LLM JSON plan → apply features → write outputs."""
    print("\tLoading cleaned CSV ...")
    input_path = str(_ROOT / input_path) if input_path else str(_ROOT / INPUT_CSV)
    if not Path(input_path).is_file():
        raise FileNotFoundError(
            f"Input not found: {input_path}. Run Agent 1 first to produce cleaned.csv."
        )

    df, profile = build_profile(input_path)

    print("\tSuggesting new features in CSV ...")
    schema_example = _SCHEMA_PATH.read_text(encoding="utf-8")
    system = f"{SYSTEM_PROMPT}\n\nExample output shape:\n{schema_example}"
    user = json.dumps(
        {
            "data_profile": profile,
            "task": f"Propose {MIN_FEATURES}–{MAX_FEATURES} features with justification and pandas_code for each.",
        },
        indent=2,
    )

    raw = call_llm(system, user)
    print("\tParsing LLM output to JSON ...")
    spec = _parse_llm_json(raw)

    original_cols = set(df.columns)
    df_feat, apply_log = apply_features(df.copy(), spec.get("features"))
    new_cols = [c for c in df_feat.columns if c not in original_cols]

    print("\tReporting in state.json ...")
    report = spec.get("report_markdown", "# Feature engineering report\n")
    report += f"\n\n## Run stats\n\n- Input rows: {len(df):,}\n- Output rows: {len(df_feat):,}\n"
    report += f"- New columns ({len(new_cols)}): {', '.join(new_cols) if new_cols else '(none)'}\n"

    if spec.get("features"):
        report += "\n## Proposed features\n\n"
        for feat in spec["features"]:
            report += f"### `{feat.get('name', feat.get('prefix', 'pca'))}`\n\n"
            report += f"- **Action:** `{feat.get('action')}`\n"
            report += f"- **Justification:** {feat.get('justification', '—')}\n"
            if feat.get("pandas_code"):
                report += f"- **Pandas code:**\n\n```python\n{feat['pandas_code']}\n```\n\n"

    report += "\n## Steps applied (code)\n\n"
    for line in apply_log:
        report += f"- {line}\n"

    out_csv = _ROOT / OUTPUT_CSV
    out_report = _ROOT / REPORT_PATH
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_feat.to_csv(out_csv, index=False)
    out_report.write_text(report, encoding="utf-8")

    result = {
        "agent": AGENT_ID,
        "input": input_path,
        "featured_csv": str(out_csv),
        "report": str(out_report),
        "rows_in": len(df),
        "rows_out": len(df_feat),
        "new_columns": new_cols,
    }

    # Call the helper to update the central state
    update_state(AGENT_ID, result)

    return result