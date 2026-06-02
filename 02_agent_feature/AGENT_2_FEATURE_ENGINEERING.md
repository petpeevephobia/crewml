# Agent 2 — Feature Engineering

Agent 2 is the second step in the CrewML pipeline. It reads the cleaned CSV from Agent 1, asks an LLM to propose **3–5** derived features (ratios, log transforms, PCA components, etc.), documents each choice with **justification** and **pandas code**, applies features deterministically in Python, and writes an enriched dataset plus a report for downstream modeling agents.

## What it does

| Input | Output |
|-------|--------|
| `01_agent_clean/cleaned.csv` | `featured.csv` — all original columns plus new features |
| | `feature_report.md` — strategy, per-feature rationale, code snippets, apply log |

The agent is designed to produce:

- **Ratios** — e.g. price per kW, mileage per year (value and usage signals)
- **Transforms** — `log1p`, `sqrt`, z-score (`standardize`) for skewed or scaled modeling
- **Derived age/usage** — `year_delta` (vehicle age), differences and products across numeric columns
- **Binning** — equal-width numeric bins as integer codes
- **PCA** — orthogonal components from correlated numeric groups (efficiency, mileage, etc.)

Row count is unchanged; only new columns are added. Features are applied **in list order**, so later steps can use columns created earlier (e.g. `vehicle_age` before `mileage_per_year`).

## How it works

The LLM does **not** receive the full CSV. Python loads every row locally; only a compact **profile** is sent to the model. The model returns a JSON plan; Python executes whitelisted `action` types — it does **not** `exec()` arbitrary generated code.

```
01_agent_clean/cleaned.csv
    │
    ▼
┌──────────────────────────────────────┐
│ 1. build_profile()          [Python] │
│    Load full CSV, build JSON summary │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 2. call_llm()               [Qwen]   │
│    Profile + schema → JSON plan      │
│    (features + report_markdown)      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 3. apply_features()         [Python] │
│    Run each feature action in order  │
└──────────────────┬───────────────────┘
                   │
                   ▼
        featured.csv + feature_report.md
```

### Division of responsibility

| Role | Component | Responsibility |
|------|-----------|----------------|
| Analyst | LLM (Qwen via `main.call_llm`) | Inspect profile; propose 3–5 `features` with `justification` and `pandas_code`; draft `report_markdown` |
| Worker | `apply_features()` | Build new columns via fixed `action` handlers on the full dataframe |
| Orchestrator | `main.py` | Run Agent 1, then `_load_agent(..., "02_agent_feature", "agent_2.py")` and `agent_2.run(call_llm)` |

The model **proposes and explains**; pandas/sklearn **execute**. `pandas_code` in the JSON is for humans and the report only.

## Running

From the project root (with `DASHSCOPE_API_KEY` in `.env`):

```bash
python main.py
```

Agent 2 runs automatically after Agent 1. If `cleaned.csv` is missing, `run()` raises `FileNotFoundError` with a message to run Agent 1 first.

## Files in this folder

| File | Purpose |
|------|---------|
| `agent_2.py` | Implementation: profile, LLM call, apply, save |
| `agent_2_prompt.json` | Example JSON shape the LLM should return |
| `featured.csv` | Written at runtime — dataset with new feature columns |
| `feature_report.md` | Written at runtime — per-feature docs and apply log |
| `AGENT_2_FEATURE_ENGINEERING.md` | This document |

### `agent_2.py`

Main module. Exposes `run(call_llm, input_path=None)` for the orchestrator.

| Piece | Role |
|-------|------|
| `build_profile()` | Loads cleaned CSV; returns `(dataframe, profile_dict)` including `numeric_columns` when present |
| `_parse_llm_json()` | Strips optional markdown fences; parses JSON from the model |
| `apply_features()` | Iterates `features` list; dispatches on `action`; returns `(dataframe, log_lines)` |
| `run()` | Profile → LLM → apply → write outputs → return metadata including `new_columns` |

**Module constants** (top of file):

| Constant | Default | Meaning |
|----------|---------|---------|
| `AGENT_ID` | `feature_engineering` | Identifier in return dict |
| `INPUT_CSV` | `01_agent_clean/cleaned.csv` | Default input (project-relative) |
| `OUTPUT_CSV` | `02_agent_feature/featured.csv` | Featured dataset output |
| `REPORT_PATH` | `02_agent_feature/feature_report.md` | Human-readable report |
| `SAMPLE_ROWS` | `15` | Rows included in LLM profile |
| `MIN_FEATURES` / `MAX_FEATURES` | `3` / `5` | Bounds baked into `SYSTEM_PROMPT` |

`SYSTEM_PROMPT` lists every allowed `action` and required JSON fields. `_SCHEMA_PATH` points at `agent_2_prompt.json` and is appended to the system message as a few-shot shape example.

**`apply_features()` behavior highlights:**

- Skips unknown columns, duplicate `name`s, and unknown `action`s (with log lines).
- **`pca`**: median-imputes listed columns, `StandardScaler` + `PCA` on rows with finite values; writes `prefix_1`, `prefix_2`, …; logs explained variance ratios.
- **`ratio`**: `column_a / (column_b + epsilon)` with numeric coercion.
- **`log1p` / `sqrt`**: non-negative clip before transform.
- **`year_delta`**: `reference_year - column` (default reference `2023`).
- Order matters: features that use another derived column must appear **after** its creator in the LLM’s `features` array.

### `agent_2_prompt.json`

Static example of valid LLM output. Shipped with five illustrative features:

1. `price_per_kw` — `ratio` of price to power  
2. `log_price` — `log1p` on price  
3. `vehicle_age` — `year_delta` from `year`  
4. `mileage_per_year` — `ratio` using the new `vehicle_age` column  
5. PCA block — `pca_eff_1`, `pca_eff_2` from fuel and mileage columns  

Each entry includes `justification` and `pandas_code` strings. The file is read at runtime and embedded in the system prompt; it is not executed as code.

### `featured.csv`

Runtime artifact. Same rows as `cleaned.csv`; additional columns from successful `apply_features()` steps. Intended input for Agent 3 (model selection).

### `feature_report.md`

Runtime artifact assembled in `run()`:

1. LLM `report_markdown` (overall strategy)  
2. **Run stats** — row counts, list of new column names  
3. **Proposed features** — per feature: action, justification, fenced `pandas_code`  
4. **Steps applied** — machine log from `apply_features()` (what actually ran or was skipped)

### `AGENT_2_FEATURE_ENGINEERING.md`

Agent manual (this file). Describes architecture, file roles, JSON contract, and configuration — distinct from the project root `README.md`.

## Step 1 — Profile (`build_profile`)

`pandas.read_csv` loads the full cleaned file. The profile sent to the LLM includes:

- `shape` — row and column counts  
- `columns` — column names after cleaning  
- `dtypes` — types per column  
- `null_counts` / `null_pct` — missing value stats  
- `sample_rows` — first `SAMPLE_ROWS` rows as strings  
- `numeric_summary` — `describe()` for numeric columns, when present  
- `numeric_columns` — list of numeric column names (helps the model pick PCA inputs and ratios)

The user message also includes a `task` string restating the 3–5 feature requirement.

## Step 2 — LLM plan

The model must return **only JSON** with two top-level keys (see `agent_2_prompt.json`):

### `features`

Ordered list of feature definitions. Each item should include:

| Field | Required | Purpose |
|-------|----------|---------|
| `action` | Yes | Whitelisted builder (see table below) |
| `name` | Yes* | New column name (`snake_case`) |
| `justification` | Yes | 1–3 sentences for the report |
| `pandas_code` | Yes | Illustrative snippet (not executed) |
| action-specific keys | Varies | e.g. `column_a`, `column_b`, `columns`, `prefix` |

\*PCA uses `prefix` + component indices instead of a single `name`.

Supported `action` values:

| Action | Parameters | What Python does |
|--------|------------|------------------|
| `ratio` | `name`, `column_a`, `column_b`, `epsilon` (optional) | Numeric ratio with safe denominator |
| `log1p` | `name`, `column` | `np.log1p` after clipping to ≥ 0 |
| `sqrt` | `name`, `column` | `np.sqrt` after clipping to ≥ 0 |
| `standardize` | `name`, `column` | Z-score; `0` if zero variance |
| `difference` | `name`, `column_a`, `column_b` | Numeric subtraction |
| `product` | `name`, `column_a`, `column_b` | Numeric multiplication |
| `year_delta` | `name`, `column`, `reference_year` | `reference_year - column` |
| `bin` | `name`, `column`, `n_bins` | `pd.cut` → integer bin codes |
| `pca` | `columns`, `n_components`, `prefix` | Scaled PCA; adds `prefix_1`, … |

Example ratio feature:

```json
{
  "name": "price_per_kw",
  "action": "ratio",
  "column_a": "price_in_euro",
  "column_b": "power_kw",
  "epsilon": 1e-6,
  "justification": "Normalizes price by engine power...",
  "pandas_code": "df['price_per_kw'] = df['price_in_euro'] / (df['power_kw'] + 1e-6)"
}
```

### `report_markdown`

High-level markdown from the LLM (feature strategy, themes). Python appends run stats, per-feature sections, and the apply log.

## Step 3 — Apply (`apply_features`)

Runs on a **copy** of the full dataframe. Each feature in the LLM list is attempted in order; the log records successes and skips (unknown column, duplicate name, insufficient rows for PCA, etc.).

After apply, `run()` computes `new_columns` as set difference between output and input column names.

## Step 4 — Write outputs

- **`02_agent_feature/featured.csv`** — enriched data for Agent 3+  
- **`02_agent_feature/feature_report.md`** — combined narrative + machine log  

`run()` returns:

```python
{
    "agent": "feature_engineering",
    "input": "<path to cleaned.csv>",
    "featured_csv": "<path>",
    "report": "<path>",
    "rows_in": int,
    "rows_out": int,
    "new_columns": ["price_per_kw", ...],
}
```

## Configuration

To change LLM behavior, edit `SYSTEM_PROMPT` or `agent_2_prompt.json`. To add a new feature type:

1. Add a branch in `apply_features()`.  
2. Document the `action` and JSON fields in `SYSTEM_PROMPT`.  
3. Add an example entry to `agent_2_prompt.json`.

To point at a different cleaned file, pass `input_path` to `run()` or change `INPUT_CSV`.

## Dependencies

- `pandas`, `numpy` — load, profile, transforms, save  
- `scikit-learn` — `StandardScaler`, `PCA` for the `pca` action  
- `openai` — used from `main.py` for Qwen (DashScope compatible API)

See project `requirements.txt`.

## Typical results (Germany used cars dataset)

On a full run (~251k rows) after Agent 1, Agent 2 commonly adds features such as:

- `price_per_kw` — value relative to power  
- `log_price` — variance-stabilized target-friendly column  
- `vehicle_age` and `mileage_per_year` — depreciation and usage intensity  
- `pca_eff_1`, `pca_eff_2` — compressed fuel/mileage signal  

Exact columns depend on the LLM plan for that run. Row count matches `cleaned.csv`; only column count grows.

## Related docs

- Project overview: `README.md` (repo root)  
- Agent 1 manual: `01_agent_clean/AGENT_1_DATA_CLEANING.md`  
- Orchestrator: `main.py`
