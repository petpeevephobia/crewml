# Agent 1 — Data Cleaning

Agent 1 is the first step in the CrewML pipeline. It takes the raw project CSV, uses an LLM to **find** data-quality problems, uses pandas to **fix** them on the full dataset, and writes a cleaned file plus a human-readable report for downstream agents.

## What it does

| Input | Output |
|-------|--------|
| `data.csv` (project root) | `cleaned.csv` — same rows, cleaned values |
| | `cleaning_report.md` — what was wrong and what changed |

The agent is designed to handle:

- **Nulls** — real missing values and placeholder strings (e.g. `"- (g/km)"`)
- **Wrong dtypes** — numbers stored as strings (`price_in_euro`, `year`, etc.)
- **Messy formats** — European decimals and units (`"10,9 l/100 km"`)
- **Outliers** — extreme numeric values capped with an IQR rule (not row deletion)

By default it **does not drop rows**; it coerces, replaces sentinels, and clips values so later agents keep the full dataset size.

## How it works

The LLM does **not** receive the entire CSV (too large for context). Python loads every row locally; only a compact **profile** is sent to the model.

```
data.csv
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
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ 3. apply_transformations()  [Python] │
│    Run each step on all rows         │
└──────────────────┬───────────────────┘
                   │
                   ▼
        cleaned.csv + cleaning_report.md
```

### Division of responsibility

| Role | Component | Responsibility |
|------|-----------|----------------|
| Analyst | LLM (Qwen via `main.call_llm`) | Inspect profile; list `issues`; propose ordered `transformations`; draft `report_markdown` |
| Worker | `apply_transformations()` | Execute fixes deterministically on the full dataframe |
| Orchestrator | `main.py` | Load API config, call `agent_1.run(call_llm)` |

The model **plans**; pandas **executes**. That keeps cleaning reproducible and safe (no arbitrary generated code on your data).

## Running

From the project root (with `DASHSCOPE_API_KEY` in `.env`):

```bash
python main.py
```

`main.py` dynamically loads `01_agent_clean/agent_1.py` and prints row counts and output paths when finished.

## Files in this folder

| File | Purpose |
|------|---------|
| `agent_1.py` | Implementation: profile, LLM call, apply, save |
| `agent_1_prompt.json` | Example JSON shape the LLM should return |
| `cleaned.csv` | Written at runtime — cleaned dataset |
| `cleaning_report.md` | Written at runtime — run-specific report |
| `README.md` | This document |

## Step 1 — Profile (`build_profile`)

`pandas.read_csv` loads the full input file. The profile sent to the LLM includes:

- `shape` — row and column counts
- `columns` — column names
- `dtypes` — inferred types per column
- `null_counts` / `null_pct` — missing value stats
- `sample_rows` — first 15 rows (strings; blanks for NaN)
- `numeric_summary` — `describe()` for numeric columns, when present

Config: `SAMPLE_ROWS = 15` at the top of `agent_1.py`.

## Step 2 — LLM plan

The system prompt tells the model to return **only JSON** with three top-level keys (see `agent_1_prompt.json`):

### `issues`

Audit list of problems found. Each item:

```json
{"column": "price_in_euro", "type": "dtype", "detail": "...", "severity": "high"}
```

Used in the final report; not executed directly.

### `transformations`

Ordered list of cleaning steps. Python runs them in sequence. Only these `action` values are supported:

| Action | What it does |
|--------|----------------|
| `replace_sentinels` | Replace listed strings (e.g. `"- (g/km)"`) with `NaN` |
| `parse_european_decimal` | Extract first number, `,` → `.`, strip units → numeric |
| `astype` | Cast column to `float`, `int`, or `str` |
| `clip_outliers` | Cap values outside Q1 − factor×IQR … Q3 + factor×IQR (default factor `3`) |

Example:

```json
{"action": "replace_sentinels", "column": "fuel_consumption_g_km", "from": ["- (g/km)"], "to": null}
```

Unknown columns or actions are skipped and logged.

### `report_markdown`

Markdown narrative from the LLM (summary of findings and changes). Python appends run stats, the `issues` list, and a line-by-line apply log.

## Step 3 — Apply (`apply_transformations`)

Runs on a **copy** of the full dataframe. Each transformation mutates columns in place. A text log records counts (e.g. how many sentinels replaced, how many outliers clipped).

## Step 4 — Write outputs

- **`01_agent_clean/cleaned.csv`** — cleaned data for Agent 2+
- **`01_agent_clean/cleaning_report.md`** — combined LLM report + machine apply log

`run()` also returns a small dict (`rows_in`, `rows_out`, paths) to the orchestrator.

## Configuration

Constants at the top of `agent_1.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `INPUT_CSV` | `data.csv` | Input path (relative to project root) |
| `OUTPUT_CSV` | `01_agent_clean/cleaned.csv` | Cleaned output |
| `REPORT_PATH` | `01_agent_clean/cleaning_report.md` | Report output |
| `SAMPLE_ROWS` | `15` | Sample rows in LLM profile |

To change LLM behavior, edit `SYSTEM_PROMPT` or `agent_1_prompt.json`. To add new cleaning operations, implement a new branch in `apply_transformations()` and document the action in `SYSTEM_PROMPT`.

## Dependencies

- `pandas`, `numpy` — load, profile, transform, save
- `openai` — used from `main.py` for Qwen (DashScope compatible API)

See project `requirements.txt`.

## Typical results (Germany used cars dataset)

On a full run (~251k rows), Agent 1 commonly:

- Parses `fuel_consumption_*` columns from strings with units to floats
- Replaces CO₂ sentinels with missing values
- Casts `price_in_euro`, `power_kw`, `power_ps`, `year` to numeric types
- Clips extreme `mileage_in_km` values

Row count stays the same; value quality improves for modeling steps later in the pipeline.
