# ColdGuard AI - Demonstration & Showcase Guide

This guide provides a step-by-step walkthrough to demonstrate the full capabilities of **ColdGuard AI** to reviewers, stakeholders, or examiners.

---

## 1. Quick Launch

Launch the Streamlit web application:
```bash
.venv/bin/streamlit run app.py
```
Open your web browser at **`http://localhost:8501`**.

The application will immediately load with high-polish navy-blue styling and automatically pre-populate with the authentic **Zenodo 15130001 Cold Storage Research Extract** so that all metrics, charts, anomaly scores, and risk classifications are functional from the very first second.

---

## 2. Tour of Application Pages

### 📊 Page 1: Overview (`app.py`)
- **Key Talking Points**:
  - Point out the **Cold-Chain Status Banner** at the top showing the current risk level (e.g. `HIGH RISK` or `MODERATE RISK`) and overall score.
  - Highlight the **Execution Mode Badge** (`LOCAL DEMO MODE (Scikit-Learn CPU)` vs `AWS CLOUD MODE`).
  - Review the **6 Core Metric Cards**: Total Sensors, Observations, Average Temperature, Average Humidity, ML Anomalies, and Excursion Count.
  - Show the **Multi-Sensor Temperature Trajectory** with the dashed red upper safe limit (8.0°C) and blue lower safe limit (2.0°C).
  - Walk through the **Risk Score Factor Contribution** horizontal bar chart showing how duration, peak severity, ML anomalies, and thermal trends add points to the overall risk score.
  - Scroll down to review the **Recent Operational Alerts** feed.

---

### 📁 Page 2: Data Upload (`pages/1_📁_Data_Upload.py`)
- **Key Talking Points**:
  - Click on **Upload Custom CSV** tab:
    - You can drag-and-drop any cold storage CSV.
    - ColdGuard AI automatically detects whether the file is semicolon-separated (`;`) or comma-separated (`,`).
    - It removes duplicate header rows caused by Arduino logger restarts (a known real-world issue in the Zenodo dataset).
    - It parses European date formats (`dd.mm.yyyy HH:MM:SS`) and handles missing values.
  - Switch to the **Load Demonstration Datasets** tab:
    - **Option A - Zenodo Research Sample**: Authentic data recorded from DHT22 sensors in an industrial cold room, containing real door opening events.
    - **Option B - [SIMULATED] Demonstration Dataset**: Synthetic data clearly labeled as simulated with normal compressor cycling, gradual refrigerant drift, sudden door opening spikes, and sub-zero overcooling.
  - Click **Load [SIMULATED] Demonstration Dataset** to observe how the system responds to deliberate synthetic anomalies!

---

### 📈 Page 3: Temperature Monitoring (`pages/2_📈_Temperature_Monitoring.py`)
- **Key Talking Points**:
  - Show the **Safe Operating Band** (shaded green region between 2.0°C and 8.0°C).
  - Toggle the **Highlight Excursion Points** checkbox: Notice how red `x` markers appear exactly where sensor temperatures break out of the safe envelope.
  - Use the **Filter Sensors** multi-select to compare individual zones:
    - Compare `SENSOR_SIM_A` (door bay spike) against `SENSOR_SIM_B` (slow refrigerant drift) and `SENSOR_SIM_C` (sub-zero freeze hazard).
  - Inspect the **Relative Humidity Trajectory** chart below, noting how humidity fluctuates during temperature events.
  - Review the **Sensor-Wise Comparative Analytics** summary table at the bottom.

---

### 🤖 Page 4: AI Anomaly Detection (`pages/3_🤖_AI_Anomaly_Detection.py`)
- **Key Talking Points**:
  - Explain why unsupervised anomaly detection is essential: it detects abnormal patterns *before* temperatures necessarily exceed hard static thresholds.
  - Point out the **Anomaly Score Trajectory**:
    - Normal cycling produces low scores (10 - 25).
    - Uncharacteristic disruptions trigger sharp score spikes (70 - 100+).
  - Demonstrate the **Dynamic Heuristic Threshold** (dashed red line):
    - Calculated as $\mu + k\sigma$.
    - Move the **Anomaly Sensitivity (k-Sigma)** slider in the sidebar from 3.0 to 2.0 and click **Recompute Local Anomaly Model**: Notice how the threshold updates reactively and captures subtler patterns.
  - Inspect the **Score Distribution Histogram** showing how anomalies form a clear tail separated from normal operational readings.
  - Scroll down to the **Plain-English Explanation Table**:
    - Each anomaly is explained in human-readable terms (e.g., *"Rapid temperature shift (+3.20°C/step); Strong deviation from baseline (+4.15°C)"*).
  - Expand the **AWS Cloud Actions** card:
    - Demonstrates transparent cloud integration. If AWS credentials are not yet configured, beginner-friendly instructions explain how to configure S3 and SageMaker without confusing stack traces.

---

### 🛡️ Page 5: Risk Assessment (`pages/4_🛡️_Risk_Assessment.py`)
- **Key Talking Points**:
  - Emphasize that this is a **transparent rule-based risk assessment engine**, NOT a black-box predictive model.
  - Review the **Risk Category Banner** (**LOW**, **MODERATE**, **HIGH**, **CRITICAL**).
  - Walk through the **Transparent Classification Factors**:
    - Explains exact point additions (e.g. +35 pts for excursion duration exceeding allowed limits, +25 pts for peak temperature exceeding 16°C).
  - Review the **Recommended Operator Actions**:
    - Tailored actionable steps (e.g., inspect door seals, verify compressor cycling, quarantine sensitive product batches for QA review).
  - Review the **Generated Operational Alerts Feed**:
    - Structured alerts categorizing events into High Excursion, Low Excursion, Rapid Shift, and Persistent Anomaly Patterns with severity levels (`CRITICAL`, `WARNING`, `INFO`).

---

### 📑 Page 6: Reports & Data Exports (`pages/5_📑_Reports.py`)
- **Key Talking Points**:
  - Review the **Sensor-Wise Operational Performance** table and accompanying bar charts comparing mean temperatures and anomaly rates across zones.
  - Demonstrate the **3 Downloadable Export Artifacts**:
    1. **Download Processed CSV**: Full dataset with all engineered features (`rolling_temperature_average`, `rolling_temperature_std`, `temperature_change`, `temperature_deviation`, and `anomaly_score`).
    2. **Download Anomaly Log CSV**: Filtered table containing only anomalous events with severity and plain-English reasons.
    3. **Download Summary Audit Report**: Formal plain-text compliance audit document summarizing executive risk status, excursion metrics, and recommended actions.
  - Click on the expander to preview the formal audit document.

---

## 3. Demonstrating Dual Execution (Local vs AWS)

### Local Demo Mode
- **Status**: Ready immediately out-of-the-box.
- **Under the Hood**: Uses `scikit-learn` Isolation Forest and tree depth scoring running on the local laptop CPU.
- **Key Message**: Enables offline development, testing, and continuous operation without incurring AWS cloud spend.

### AWS Cloud Mode
- **Status**: Select **AWS** in the sidebar.
- **Workflow**:
  1. Ingests data through the Python pipeline.
  2. Uploads numeric features to Amazon S3.
  3. Triggers an official Amazon SageMaker AI Random Cut Forest (RCF) training job.
  4. Executes SageMaker Batch Transform on ephemeral instances (auto-shuts down to prevent costs).
  5. Pulls real RCF anomaly scores from S3 into the Streamlit UI.
- **Prerequisites Handling**: If AWS credentials or S3 bucket are missing, ColdGuard AI displays exact setup steps rather than faking a connection.
