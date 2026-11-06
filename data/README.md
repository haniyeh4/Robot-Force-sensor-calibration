# Data

All data were recorded on a KUKA KR 600 with an ATI Gamma IP60 SI-130-10 force/torque sensor and a
custom tool holder with surgical forceps (single tool configuration). Logging ran at 250 Hz through
the Simulink KUKA driver, which streams robot state and sensor readings over TCP to a Python logger.
Files are gzip-compressed JSON Lines (one record per line) and keep their original names.

| File | Used for | Records | Recorded |
|---|---|---|---|
| `1_Quasi_Static_Log2.jsonl.gz` | Quasi-static training/test set (362 poses) | 362 | 2025-07-28 |
| `1_TheRightWay.csv` | Commanded joint angles [deg] of the same 362 poses, same order | 362 | – |
| `manual_fresh_log.jsonl.gz` | Quasi-static benchmark (manually jogged poses); 56 stationary frames are used | 100 | 2025-07-28 |
| `2_DyF_L1.jsonl.gz` | Dynamic training pool (random motion) | 32,279 | 2025-07-23 |
| `2_DyF_L2.jsonl.gz` | Dynamic training pool (random motion) | 55,944 | 2025-07-23 |
| `2_DyF_L3.jsonl.gz` | Additional random-motion recording (used only in recording-level checks) | 51,698 | 2025-07-23 |
| `2_DyF_L4.jsonl.gz` | Dynamic benchmark (random motion, recorded separately) | 34,363 | 2025-07-23 |

## Fields

Quasi-static logs:

- `X_Pos` [6]: end-effector x, y, z [m] and roll, pitch, yaw [deg]
- `FRaw` [6]: unfiltered sensor wrench Fx, Fy, Fz [N], Tx, Ty, Tz [N·m]
- `FF_EE` [6]: wrench after the Simulink 5 Hz low-pass filter, end-effector frame (**model target**)
- `FF_GB` [6]: filtered wrench in the robot base frame
- `X_Dot_Cmds` [6], `X_DoubleDot_Cmds` [6]: commanded Cartesian velocity and acceleration (linear + angular)
- `Foot_Pedal`: teleoperation pedal state

Dynamic logs add:

- `timestamp`, `delta_time` [s]
- `POS` [6]: same as `X_Pos`
- `FF_EE_Quasi` [6]: quasi-static model prediction, computed online during logging
- `FF_EE_Dynamic` [6]: `FF_EE - FF_EE_Quasi` (**dynamic model target**)

Dynamic model input: `POS[3:6] + X_DoubleDot_Cmds`, i.e. roll, pitch, yaw and the 6-DOF commanded
Cartesian acceleration (3 linear + 3 angular).

Note: `FF_EE_Quasi` in the dynamic logs was produced online by an earlier quasi-static model fitted on a
previous 362-pose session, not by the degree-3 model in `scripts/01_quasi_static.py`. The dynamic models
learn whatever offset remains after that online subtraction.

## Integrity

MD5 of the original uncompressed files (CR line endings removed):

```
6659a5696de1c312737371bdd8d01e9b  1_Quasi_Static_Log2.jsonl
eaec5e90dcbc4f6f4f146dc2686434a8  manual_fresh_log.jsonl
be9b0cec1fee4e4d7f0d47afee12a1b8  2_DyF_L1.jsonl
c60fdee348c56c3cdb4246f76302867f  2_DyF_L2.jsonl
8f7ca3713c1d17831a7b0a335855e164  2_DyF_L3.jsonl
c8fb6eede2832cb3f5f6301ff6ad5cde  2_DyF_L4.jsonl
```
