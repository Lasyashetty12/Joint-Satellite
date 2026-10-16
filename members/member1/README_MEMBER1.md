# Member 1 — Full Pipeline

Workspace: `members/member1/`  
Branch: `member1/full-pipeline`  
Executable filename prefix: `m1_`

## Current milestone
Phase 1 — Environment and dataset validation. Start with a tiny sample and verify the dataset's official documentation, license, modalities, timestamps, labels, alignment, and storage requirements before training.

## Dataset inventory
Run from the repository root:

```bash
python members/member1/data/m1_dataset_inventory.py --data-dir path/to/small_dataset
```

The inventory lists supported image files and byte sizes. It does not verify that files are paired, aligned, labeled, or suitable for training.
