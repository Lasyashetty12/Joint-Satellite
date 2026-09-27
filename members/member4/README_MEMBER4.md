# Member 4 workspace

| Item | Value |
|---|---|
| Branch | `member4/full-pipeline` |
| Workspace | `members/member4/` |
| File prefix | `m4_` (tests: `test_m4_*`) |
| Diversity focus | Visualization, explainability (α weights, SHAP for XGBoost), dashboard, integration cleanliness |

Member 4 still implements the **complete** pipeline (generator, MLTS, EO baselines, fusion, evaluation, dashboard).

## Setup (local, run from repository root)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r members/member4/requirements_member4.txt
python members/member4/m4_env_check.py --save-report
python -m pytest members/member4/tests -q
```

## Colab smoke test

```python
!git clone <REPOSITORY_URL>
%cd multimodal-satellite-ai
!git checkout member4/full-pipeline
!pip install -q -r members/member4/requirements_member4.txt
!python members/member4/m4_env_check.py
!python -m pytest members/member4/tests -q
```
Colab: Runtime → Change runtime type → T4 GPU (if available) to see `cuda_available: True`.

## Milestone log

| Milestone | Status |
|---|---|
| P1-M1 Environment + Git + Colab smoke test | in progress |
