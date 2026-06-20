# CrewML

Agents that collaborate to build and validate an ML pipeline for you. Each agent covers a step you would recognize from classical ML (cleaning, features, model choice, validation), wired together as a live AutoML demo on a real automotive dataset.

## What this is

A multi-agent system that takes a raw CSV and runs it through specialized LLM-backed agents: clean data, engineer features, compare models, and validate results. Agents share reasoning and outputs via a central `state.json`, so you can see each step and how disagreements get resolved.

The goal is a pipeline you can demo end-to-end — including a deliberate conflict (e.g. PCA vs raw features) resolved by a Mediator agent — and compare against a single-prompt “build me an ML pipeline” baseline.

**Dataset:** [Germany used cars (Kaggle, 2023)](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023)

## How it’s built

| Piece | Role |
|-------|------|
| `main.py` | Orchestrator — runs agents in sequence |
| One folder per agent | Data Cleaning, Feature Engineering, Model Selection, Validation, Mediator |
| `state.json` | Shared read/write state for reports, metrics, and arguments |
| `call_llm(system, user)` | Single Qwen Cloud API helper used by every agent |

Each agent folder has its own manual (not named `README.md`, to avoid clashing with this file):

| Agent | Manual |
|-------|--------|
| Data cleaning | `01_agent_clean/AGENT_1_DATA_CLEANING.md` |
| Feature engineering | `02_agent_feature/AGENT_2_FEATURE_ENGINEERING.md` |
| Model selection | `03_agent_model/AGENT_3_MODEL_SELECTION.md` |

**LLM:** [Qwen Cloud](https://docs.qwencloud.com/developer-guides/getting-started/introduction)

**Stack:** Python, pandas, scikit-learn (and optional XGBoost) for execution; agents use the LLM for analysis, code generation, and written justification.

### Agent flow

1. **Data Cleaning** — nulls, outliers, dtypes → cleaned frame + change report  
2. **Feature Engineering** — 3–5 proposed features (ratios, transforms, PCA, etc.) with code and rationale  
3. **Model Selection** — run candidate models (e.g. RF, XGBoost, linear regression) → comparison in `state.json`  
4. **Validation** — cross-validation, F1/RMSE, plain-English trust verdict  
5. **Mediator** (on conflict) — reads opposing arguments from `state.json`, picks a winner, unblocks the pipeline  

**Phase 3+:** Hardcoded disagreement for the demo; single-agent baseline for accuracy comparison. **Phase 4:** Minimal HTML dashboard polling `state.json` for live status during the run.

## References

- Qwen Cloud docs: https://docs.qwencloud.com/developer-guides/getting-started/introduction
