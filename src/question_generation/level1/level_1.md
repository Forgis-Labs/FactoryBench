# Level 1 Question Templates

Level 1 questions focus on **State Identification**. They evaluate the model's ability to extract and interpret raw sensor data and control signals at specific timestamps or over short intervals.

## Core Topics
The questions are categorized into 6 main topics, each with three variants: **True/False (TF)**, **Multiple Choice (MC)**, and **Open-Ended (Open)**.

### 1. Joint Position Movement
Checks if a specific joint's position has changed between two points in time.
- **TF (`state_joint_moved_tf`)**: "Is joint {axis} at the same position at {t1}ms and {t2}ms?"
- **MC (`state_joint_moved_mc`)**: "What is the status of the position of joint {axis} between {t1}ms and {t2}ms?"
- **Open (`state_joint_moved_open`)**: "What is the absolute difference in position of joint {axis}... in rad?"

### 2. Friction Proxy
Evaluates changes in a friction-related proxy (ratio of $|effort\_current| / |feedback\_speed|$) over a time window.
- **TF (`state_friction_increase_tf`)**: "Has the friction proxy for joint {axis} increased from {t1}ms to {t2}ms?"
- **MC (`state_friction_increase_mc`)**: "How did the friction proxy for joint {axis} change from {t1}ms to {t2}ms?"
- **Open (`state_friction_increase_open`)**: "What is the ratio of the friction proxy at {t2}ms compared to {t1}ms?"

### 3. End-Effector Acceleration
Measures the acceleration of the end-effector using vibration sensor data.
- **TF (`state_acceleration_tf`)**: "Is the end-effector acceleration magnitude at {t_ms}ms greater than {threshold}?"
- **MC (`state_acceleration_mc`)**: "Which axis (X, Y, or Z) has the highest absolute acceleration at {t_ms}ms?"
- **Open (`state_acceleration_open`)**: "What is the acceleration of the end effector at {t_ms}ms? (X_Y_Z vector)"

### 4. External Force Detection
Detects if an external force (magnitude) is present at a given timestamp.
- **TF (`state_external_force_detected_tf`)**: "Is an external force detected at {t_ms}ms?"
- **MC (`state_external_force_detected_mc`)**: "What is the detected state of external forces at {t_ms}ms?"
- **Open (`state_external_force_detected_open`)**: "What is the magnitude of the external force detected at {t_ms}ms in N?"

### 5. Joint Jerk
Calculates the numerical jerk (rate of change of acceleration) for a joint using a central difference approximation.
- **TF (`state_jerk_tf`)**: "Does the jerk of joint {axis} at {t_ms}ms exceed {threshold}?"
- **MC (`state_jerk_mc`)**: "What is the magnitude range (Low/Medium/High) of the jerk?"
- **Open (`state_jerk_open`)**: "What is the jerk of joint {axis} at {t_ms}ms in rad/s^3?"

### 6. Torque Magnitude
Measures the absolute torque around a specific axis (X, Y, or Z) derived from wrench data.
- **TF (`state_torque_magnitude_tf`)**: "Is the torque magnitude about the {axis}-axis at {t_ms}ms greater than {threshold}?"
- **MC (`state_torque_magnitude_mc`)**: "Compare the torque magnitude about the {axis}-axis at {t_ms}ms to a threshold."
- **Open (`state_torque_magnitude_open`)**: "What is the torque magnitude about the {axis}-axis at {t_ms}ms in Nm?"

### 7. Tracking Error (Position)
Compares the commanded setpoint and the actual feedback position to identify control deviations.
- **TF (`state_tracking_error_tf`)**: "Is the position tracking error of joint {axis} at {t}ms above {threshold} rad?"
- **MC (`state_tracking_error_mc`)**: "How has the position tracking error of joint {axis} evolved between {t1}ms and {t2}ms?"
- **Open (`state_tracking_error_open`)**: "What is the position tracking error of joint {axis} at {t}ms in rad?"

### 8. Joint Speed & Semantic Limits
Evaluates joint speed against absolute values or semantic limits (from the Machine KG).
- **TF (`state_joint_speed_tf`)**: "Is the speed of joint {axis} at {t}ms below {threshold} rad/s?"
- **Ranking (`state_joint_speed_ranking`)**: "Rank the joints {joints_list} by their absolute speed at time {t_ms}ms."
- **Semantic (`state_joint_within_rated_speed`)**: "Which of the following joints are operating within their rated maximum speed at T={t_ms}ms?" **(Requires KG)**

### 9. Motor Current & Semantic Limits
Direct indicator of actuator effort. High values suggest overload; low values suggest free movement.
- **TF (`state_motor_current_tf`)**: "Does the motor current of joint {axis} at {t}ms exceed {threshold} A?"
- **Semantic (`state_current_within_rated`)**: "Is the motor current of joint {axis} at T={t_ms}ms within the rated continuous current limit?" **(Requires KG)**

### 10. Signal Description (Free-form)
Evaluates the model's ability to provide a natural language description of signal behavior.
- **Free-form (`state_signal_description`)**: "Describe in one sentence the behaviour of {signal} for joint {axis} over the window [T={t1}ms, T={t2}ms]."

The semantic meaning is evaluated by LLM-as-a-judge technique.

## Data Source
Level 1 generation is uniquely integrated with the **Hugging Face Hub**. It consumes raw **Parquet** files directly from the cloud.

- **Default Repository**: `the authors/FactoryNet_Dataset`
- **Format**: The script downloads Parquet files, parses them via `pandas`, and splits them into monotonic segments to ensure time-series consistency.

## Availability Table
| Category | AURSAD | voraus-AD | FactoryWave |
| :--- | :---: | :---: | :---: |
| 1. Joint Position | ✓ | ✓ | ✓ |
| 2. Friction Proxy | ✓ | ✓ | ✓ |
| 3. EE Acceleration | ✓ | - | - |
| 4. External Force | ✓ | - | ✓ |
| 5. Joint Jerk | ✓ | ✓ | ✓ |
| 6. Torque Magnitude | ✓ | - | ✓ |
| 7. Tracking Error | ✓ | ✓ | ✓ |
| 8. Joint Speed | ✓ | ✓ | ✓ |
| 9. Motor Current | ✓ | ✓ | ✓ |
| 10. Signal Description | ✓ | ✓ | ✓ |

## Answer Formats
| Type | Format |
| :--- | :--- |
| **TF** | Multiple Choice (Single Select) with "Yes" or "No" options. |
| **MC** | Multiple Choice (Single Select) with 2-3 categorical options. |
| **Ranking** | Permutation string (e.g., `BADC`) ranking joints by a signal. |
| **Multi-Select**| Bitstring (e.g., `TFTF`) for per-joint boolean checks. |
| **Open** | Numerical values or Tensors (e.g., `X_Y_Z`). |
| **Free-form** | Natural language description, evaluated via LLM-as-judge. |

## Semantic Priors (Machine KG)
Advanced Level 1 questions require knowing the specific physical limits of the robot. This information is stored in `data/labelling/machines.json` (the Machine Knowledge Graph).
- **Joint Speed Limits**: Max angular velocity (rad/s) per joint.
- **Rated Current**: Continuous current limits (A) per motor.

## Genericity and Adaptability
Level 1 is designed to be **robot-agnostic** by following these principles:

1. **Physical Foundations**: The templates use universal physical concepts (position, velocity, effort, jerk, torque). These apply to any mechanical system with actuators and sensors.
2. **Standardized Data Schema**: The implementation (`mc_truth.py`) assumes a normalized mapping (e.g., `feedback_pos_{i}`). This allows different robots (UR5, Franka, mobile bases) to use the same logic as long as their raw data is pre-processed into this common format.
3. **Parametric Logic**: All questions use configurable thresholds (`eps`, `thresholds`). This ensures that "significant movement" or "high jerk" can be tuned based on the specific accuracy and dynamics of each robot.
4. **Flexible Indexing**: Using `{axis}` and `{axis_label}` allows the templates to scale to robots with varying numbers of joints.

## Usage
To generate Level 1 questions, run the following command from the project root:

```powershell
python -m src.question_generation.level1.level1 -n 100 --seed 27
```

### Options:
- `-n`: Number of questions to generate (default: 100).
- `--seed`: Random seed for reproducibility.
- `--dataset-repo`: Specify a different Hugging Face repository (default: `the authors/FactoryNet_Dataset`).
- `--test-mode`: **Fast development mode**. Only downloads the first 5000 rows of an episode to verify logic without waiting for full dataset downloads.
