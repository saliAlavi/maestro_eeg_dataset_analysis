# Head IMU (accel+gyro) attended-direction over the decision-window curve — head-orienting branch

Per-window IMU summary (pose + net head-turn) -> LDA/MLP, 4-way (chance .25). **DIRECTION** = attended physical loudspeaker (does head pose/motion reveal *where* the listener attends); **CONTENT** = attended permuted slot (must stay ~chance). Characterisation branch, **not** in the EEG-only headline.

| window | DIRECTION within (lda) | DIRECTION loso (lda) | CONTENT within | CONTENT loso |
|---|---|---|---|---|
| 5s | **0.384** | **0.315** | 0.248 | 0.249 |
| 10s | **0.393** | **0.328** | 0.237 | 0.247 |
| 15s | **0.409** | **0.342** | 0.239 | 0.245 |
| 20s | **0.452** | **0.396** | 0.237 | 0.245 |
| 30s | **0.465** | **0.381** | 0.242 | 0.246 |

- Head orienting is a **weaker** azimuth cue than gaze/scene video (the IMU has no magnetometer, so absolute head yaw toward a speaker is only partly recoverable); **content stays at chance** — head motion carries no talker content.

