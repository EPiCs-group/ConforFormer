# Reviewer Fingerprint Summary

Date: 2026-04-20

This note summarizes the follow-up screening of alternative 2D fingerprints for the XGBoost baseline used in the paper.

## Setup

- Data: Uni-Mol molecular property benchmark splits
- Model family: XGBoost on RDKit Morgan fingerprints
- Fast screen: one-seed runs using the previously selected XGBoost configs
  - classification: `x03`
  - regression: `x07`
- Variants screened:
  - `ECFP4_2048`
  - `ECFP4_16384`
  - `ECFP6_2048`
  - `ECFP6_16384`

## Screen Ranking

Ranking below uses the average of the screen classification and regression normalized scores.

| Fingerprint | Classification norm | Regression norm | Combined norm | Within range |
| --- | ---: | ---: | ---: | ---: |
| `ECFP6_16384` | 0.4084 | 0.0198 | 0.2141 | 10/14 |
| `ECFP4_2048` | 0.1864 | -0.0454 | 0.0705 | 8/14 |
| `ECFP6_2048` | 0.1922 | -0.0740 | 0.0591 | 7/14 |
| `ECFP4_16384` | 0.0909 | 0.0176 | 0.0543 | 9/14 |

Main takeaway: `ECFP6_16384` was the strongest fingerprint in the screen, mainly because of stronger classification performance.

## Classification vs Current `ECFP4_1024` Baseline

The current paper baseline is the 5-seed `ECFP4_1024` result. The screen winner (`ECFP6_16384`) improved some tasks but not all:

| Task | `ECFP4_1024` | `ECFP6_16384` | Delta |
| --- | ---: | ---: | ---: |
| BBBP | 0.6817 | 0.6908 | +0.0090 |
| BACE | 0.8045 | 0.8328 | +0.0283 |
| ClinTox | 0.8386 | 0.9154 | +0.0768 |
| Tox21 | 0.7238 | 0.7398 | +0.0160 |
| ToxCast | 0.5893 | 0.5996 | +0.0103 |
| SIDER | 0.6179 | 0.6171 | -0.0008 |
| HIV | 0.7618 | 0.7441 | -0.0178 |
| MUV | 0.7735 | 0.7220 | -0.0515 |

Interpretation:

- The improvement is not a qualitative across-the-board change.
- `ClinTox` and `BACE` improve the most.
- `HIV` and `MUV` are worse than the current `ECFP4_1024` baseline.
- Because the new fingerprint numbers come from a one-seed screen, they should be treated as directional rather than final.

## Regression Notes

- Regression differences were smaller than the classification spread.
- `ECFP6_16384` and `ECFP4_16384` were close on regression.
- `ECFP6_16384` had the best overall combined screen score, but the regression margin alone was small.

## Tuning Status

A narrowed follow-up tuning setup for `ECFP6_16384` was prepared:

- classification configs: `x03`, `x01`, `x10`
- regression configs: `x07`, `x06`, `x05`

At the time of writing, the tuning scaffolding exists but no completed tuning result tables were written yet.

## Practical Conclusion

For the paper, the screen does not justify replacing the current reference baseline with a different fingerprint in the main text. The stronger `ECFP6_16384` classification screen result is interesting, but it is not a clean qualitative win once task-by-task tradeoffs are considered.
