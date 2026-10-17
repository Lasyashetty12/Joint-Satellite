

\# ResearchRadar AI — Member 3



\## Role

Member 3 implements the ResearchRadar AI pipeline, with emphasis on rigorous forecasting evaluation, backtesting, uncertainty estimation, and early-warning scoring.



\## Environment Setup



Requirements:

\- Python 3.11 or compatible

\- Git

\- Internet connection



From the repository root, create and activate the environment:



```cmd

py -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements\_member3.txt

```



\## Current Data Collector



Collect a small sample of research papers from OpenAlex:



```cmd

python members\\member3\\m3\_collect\_openalex.py

```



Output:



`members/member3/data/sample\_openalex.json`



\## Run Tests



```cmd

python -m pytest members\\member3\\tests -q

```



\## Current Progress



\- Member 3 branch: `member3/full-pipeline`

\- OpenAlex sample collector implemented

\- JSON output validated

\- Automated test passing

\- Initial setup commit: `ce76eb3`



\## Planned Pipeline



1\. Collect, clean, deduplicate, and validate papers.

2\. Generate semantic embeddings and discover research topics.

3\. Track topic evolution and create historical time series.

4\. Forecast topic growth and evaluate models using chronological backtesting.

5\. Estimate uncertainty and identify emerging topics.

6\. Integrate evidence retrieval, agents, API, and dashboard.

7\. Test and document the complete pipeline.



\## Data and Security Rules



\- Keep executable Python files in this workspace prefixed with `m3\_`.

\- Never commit `.env` files, API keys, or large raw datasets.

\- Use chronological splits for forecasting evaluation.

\- Keep outputs reproducible and document dataset filters and limitations.



