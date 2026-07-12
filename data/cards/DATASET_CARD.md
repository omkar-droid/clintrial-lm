# Dataset Card — ClinTrial Instruction Set

## Summary
An instruction-tuning dataset for clinical-trial understanding, built from the
public [ClinicalTrials.gov v2 API](https://clinicaltrials.gov/data-api/api). Each
registry study is converted into several instruction/response examples with
**programmatic ground truth** (labels come from structured registry fields, not
from a teacher LLM), so targets are auditable and hallucination-free.

## Tasks
| Task | Input | Target (ground truth source) |
|------|-------|------------------------------|
| `extract_eligibility` | Free-text eligibility criteria | JSON `{inclusion[], exclusion[]}` parsed from the criteria text |
| `plain_language_summary` | Title + conditions + interventions | Registry brief summary |
| `condition_qa` | Brief summary | Conditions list |
| `phase_classification` | Brief summary | Trial phase(s) |

## Splits
Split **by trial (NCT id)**, not by example — no trial appears in more than one
split, so multiple examples from the same trial can't leak across train/val/test.
Default ratio 90 / 5 / 5.

## Preference data (for DPO)
`dpo_train.jsonl` / `dpo_val.jsonl` contain `{prompt, chosen, rejected}` triples.
`chosen` is the faithful gold answer; `rejected` is a degraded variant produced by
perturbation — dropping inclusion criteria and injecting a hallucinated one, or
substituting an unrelated / truncated answer. This encodes the preference
"faithful & complete > incomplete or hallucinated." In a production setting you
would replace these with human preference labels or reward-model scores.

## Provenance & licensing
Source records are U.S. government works from ClinicalTrials.gov (public domain).
This dataset is derived programmatically. It is **for research/education only and
is not medical advice.**

## Reproduce
```bash
python data/build_dataset.py --max-studies 8000 --out-dir data/processed
# or, offline smoke test:
python data/build_dataset.py --offline --out-dir data/processed
```
