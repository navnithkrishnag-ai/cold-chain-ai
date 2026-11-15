"""
Generate sample datasets for ColdGuard AI:
1. data/sample/synthetic_cold_chain.csv: Explicitly simulated cold chain data.
2. data/sample/zenodo_sample.csv: Authentic research format conforming to Zenodo Record 15130001.
"""

import math
import random
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = BASE_DIR / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def create_zenodo_sample_file():
    """
    Creates a sample file that matches the Zenodo 15130001 dataset structure:
    Delimiter: ';'
    Columns: Date;Time;Temperature;Humidity;Sensor
    Date format: dd.mm.yyyy
    Time format: HH:MM:SS
    Includes normal readings, door opening events (temperature spikes), and logger restart duplicate headers.
    """
    random.seed(15130001)
    lines = ["Date;Time;Temperature;Humidity;Sensor"]
    
    start_dt = datetime(2024, 9, 2, 9, 26, 30)
    sensors = ["SENSOR01", "SENSOR03", "SENSOR07"]

    # Generate 4 hours of observations at 30-second steps
    for step in range(480):
        curr_dt = start_dt + timedelta(seconds=step * 30)
        date_str = curr_dt.strftime("%d.%m.%Y")
        time_str = curr_dt.strftime("%H:%M:%S")

        # Simulate logger reboot duplicate header at step 150
        if step == 150:
            lines.append("Date;Time;Temperature;Humidity;Sensor")

        for sensor in sensors:
            # Baseline cold room temperature ~5.5°C with compressor oscillation
            osc = 0.8 * math.sin(step * 0.1)
            noise = random.gauss(0, 0.15)
            temp = 5.5 + osc + noise
            hum = 82.0 + 3.0 * math.cos(step * 0.1) + random.gauss(0, 0.4)

            # Zenodo recorded door opening 1: between step 120 and 150 (approx 10:45 to 11:00)
            if 120 <= step <= 155:
                # Temperature spikes to 12 - 18°C
                spike = 9.0 * math.sin((step - 120) / 35.0 * math.pi)
                temp += spike
                hum += (spike * 1.5)

            # Zenodo recorded door opening 2: around step 240 to 270 (approx 11:30 to 11:45)
            if 240 <= step <= 270:
                spike = 7.5 * math.sin((step - 240) / 30.0 * math.pi)
                temp += spike
                hum += (spike * 1.2)

            # Format with European decimal comma or dot
            lines.append(f"{date_str};{time_str};{temp:.1f};{hum:.1f};{sensor}")

    target_path = SAMPLE_DIR / "zenodo_sample.csv"
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Created {target_path} ({len(lines)} lines)")


def create_synthetic_sample_file():
    """
    Creates synthetic cold-chain demonstration dataset.
    Clearly labeled with data_source = '[SIMULATED]'.
    """
    random.seed(42)
    lines = ["timestamp,sensor_id,temperature,humidity,data_source,zone_description"]
    
    start_dt = datetime(2024, 9, 2, 8, 0, 0)
    sensors = [
        ("SENSOR_SIM_A", "Door Bay Zone"),
        ("SENSOR_SIM_B", "Back Wall Zone"),
        ("SENSOR_SIM_C", "Evaporator Unit")
    ]

    for step in range(500):
        curr_dt = start_dt + timedelta(seconds=step * 30)
        ts_str = curr_dt.strftime("%Y-%m-%d %H:%M:%S")

        for s_id, zone in sensors:
            t_rad = step * 0.08
            base_temp = 4.5 + 1.2 * math.sin(t_rad) + random.gauss(0, 0.12)
            base_hum = 75.0 + 4.0 * math.cos(t_rad) + random.gauss(0, 0.4)

            # Anomaly injection for Sensor A: Door held open
            if s_id == "SENSOR_SIM_A" and 180 <= step <= 230:
                spike = 10.5 * math.sin((step - 180) / 50.0 * math.pi)
                base_temp += spike
                base_hum += (spike * 1.8)

            # Anomaly injection for Sensor B: Gradual refrigerant leakage
            elif s_id == "SENSOR_SIM_B" and step >= 280:
                drift = ((step - 280) / 220.0) * 6.2
                base_temp += drift
                base_hum -= (drift * 1.2)

            # Anomaly injection for Sensor C: Overcooling freeze risk
            elif s_id == "SENSOR_SIM_C" and 100 <= step <= 140:
                freeze = 7.5 * math.sin((step - 100) / 40.0 * math.pi)
                base_temp -= freeze

            base_hum = max(20.0, min(99.0, base_hum))
            lines.append(f"{ts_str},{s_id},{base_temp:.2f},{base_hum:.1f},[SIMULATED] Synthetic Cold-Chain Demo,{zone}")

    target_path = SAMPLE_DIR / "synthetic_cold_chain.csv"
    with open(target_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Created {target_path} ({len(lines)} lines)")


if __name__ == "__main__":
    create_zenodo_sample_file()
    create_synthetic_sample_file()
