| Method | Disk | Peak VRAM | TTFT | Decode | Eligibility F1 | JSON valid | Quality kept |
|---|---|---|---|---|---|---|---|
| bf16 (baseline) | 15.24 GB | 15.36 GB | 21.1 ms | 56.9 tok/s | 0.9673 | 1.0 | 100.0% |
| INT8 (bitsandbytes) | 15.24 GB | 9.14 GB | 110.9 ms | 11.0 tok/s | 0.9676 | 1.0 | 100.0% |
| NF4 4-bit (bitsandbytes) | 15.24 GB | 5.81 GB | 49.3 ms | 30.8 tok/s | 0.9632 | 1.0 | 99.6% |
| AWQ 4-bit | 5.58 GB | 6.72 GB | 62.4 ms | 16.1 tok/s | 0.9509 | 0.9839 | 98.3% |

_Baseline = bf16 (F1 0.9673). 'Quality kept' = F1 relative to baseline._
