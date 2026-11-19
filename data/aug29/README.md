# Additional session for the recording-level validation (manuscript Section 3.5, Table 3)

Eleven random-motion recordings from a separate session, recorded with the same robot, sensor, tool
holder and Simulink/Python logging pipeline as the main data (see `../README.md` for the setup and
the field definitions, which are identical). They are used only for the recording-level validation
in `scripts/07_recording_level.py`; none of the main results use them.

| File | Records | Recorded |
|---|---|---|
| `2_DyF_L1.jsonl.gz` | 49,810 | 2025-08-29 |
| `2_DyF_L2.jsonl.gz` | 48,745 | 2025-08-29 |
| `2_DyF_L3.jsonl.gz` | 49,077 | 2025-08-29 |
| `2_DyF_L4.jsonl.gz` | 52,731 | 2025-08-29 |
| `2_DyF_L5.jsonl.gz` | 52,526 | 2025-08-29 |
| `2_DyF_L6.jsonl.gz` | 48,721 | 2025-08-29 |
| `2_DyF_L7.jsonl.gz` | 48,131 | 2025-08-29 |
| `2_DyF_L8.jsonl.gz` | 54,028 | 2025-08-29 |
| `2_DyF_L9.jsonl.gz` | 58,915 | 2025-08-29 |
| `2_DyF_L10.jsonl.gz` | 57,161 | 2025-08-29 |
| `2_DyF_L11.jsonl.gz` | 44,752 | 2025-08-29 |

Total: 564,597 records at 250 Hz.

Notes:

- Model input and target are the same as for the main dynamic data: `POS[3:6] + X_DoubleDot_Cmds`
  (roll, pitch, yaw and the 6-DOF commanded acceleration) and `FF_EE_Dynamic`.
- As in the main data, `FF_EE_Quasi` was computed online during logging by the quasi-static model
  in use at that time, so `FF_EE_Dynamic` is the offset remaining after that online subtraction.
- Recording 11 contains much faster motion than recordings 1-10 (99th percentile of |linear
  acceleration| 0.53 vs 0.10-0.16, of |angular acceleration| 1.92 vs 0.13-0.31, in driver units).

## Integrity

MD5 of the original uncompressed files (CR line endings removed):

```
2304651e1e21ced57a7d271620e846f3  2_DyF_L1.jsonl
3752ad46137296b0907ede11d33d3c17  2_DyF_L2.jsonl
f3a6df394b780d1db07a91897de3fa4d  2_DyF_L3.jsonl
75b9016e0d621a93b3222f8e4b900293  2_DyF_L4.jsonl
6a72e700996787ffd93b6adc77433dea  2_DyF_L5.jsonl
50ec7ceda6382aeadd3dca6a22801db4  2_DyF_L6.jsonl
63d021ace00289ba7837a68f90cb6334  2_DyF_L7.jsonl
e6df797e441b9e4074bbc07669d89e2b  2_DyF_L8.jsonl
3e31637c1e655d32ac214176c88bd366  2_DyF_L9.jsonl
986c3a620735da7e7cfd7784e61fb206  2_DyF_L10.jsonl
86e75c799b23fbd3de010255179cb446  2_DyF_L11.jsonl
```
