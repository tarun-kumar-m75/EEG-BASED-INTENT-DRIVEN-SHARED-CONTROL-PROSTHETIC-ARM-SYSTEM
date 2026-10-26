# EEG Prosthetic Arm Interface — Teammate Setup Guide

This package contains the complete standalone interface and real-time control system for our final-year project:
**EEG-Based Intent-Driven Shared-Control Prosthetic Arm System**.

---

## 1. Quick Start (Run Locally on Your Machine)

### Step 1: Install Python Requirements
Open PowerShell or Command Prompt in this folder and run:
```powershell
pip install -r requirements.txt
```

### Step 2: Launch the Interface
Simply **double-click** the file:
```text
run_interface.bat
```
*(Or run `streamlit run dashboard/app.py` in your terminal)*

It will immediately open the dashboard in your web browser at:
`http://localhost:8501`

---

## 2. Connecting the Physical Prosthetic Arm (STM32 / Arduino)

### Serial Communication Protocol:
* **Baud Rate**: `115200 bps` (8 Data bits, No Parity, 1 Stop bit)
* **Line Ending**: Newline (`\n`)

### Commands Sent by the Interface to Your Hardware:
| Hand Action | Packet Sent Over USB Serial | Description |
| :--- | :--- | :--- |
| **Close Hand** | `<CMD:GRASP,CRC:0x4A>\n` | Actuate servo/motor to close fingers |
| **Open Hand** | `<CMD:RELEASE,CRC:0x5B>\n` | Actuate servo/motor to open fingers |
| **Hold / Idle** | `<CMD:HOLD,CRC:0x3F>\n` | Maintain current mechanical grip |

### Firmware Ready to Flash:
Inside the `firmware/` folder, you will find:
* `firmware/prosthetic_arm_controller.ino`: Complete Arduino/STM32 C++ firmware ready to upload via Arduino IDE / STM32Duino.

### How to Connect in the Dashboard:
1. Plug your STM32 / Arduino into the PC via USB.
2. In the dashboard sidebar, switch **Actuator Mode** from `Simulation Mode` to `Physical Hardware (STM32/UART)`.
3. Select your COM port (e.g. `COM3` or `COM4`) from the dropdown.
4. Click **Initialize Port**.
5. You can immediately test the motor using the **`[Close Hand]`** and **`[Open Hand]`** manual buttons in the sidebar!

---

## 3. How the EEG Intent Pipeline Operates

1. **EEG Source**: Plays pre-recorded $160\text{ Hz}$ multi-channel EEG recordings from Subject 001 in the PhysioNet EEGBCI benchmark (`data/test_replay_stream.npz`).
2. **Preprocessing**: Applies Common Average Referencing (CAR), $50\text{ Hz}$ notch filter, and $8–30\text{ Hz}$ zero-phase Butterworth bandpass filter.
3. **ML Classifier**: Evaluates the sliding window using **Filter-Bank Common Spatial Patterns (FBCSP: 5 sub-bands)** + **Shrinkage LDA** to output exact probabilities (`predict_proba`).
4. **Shared-Control & Safety Layer**:
   * **Confidence Gate**: Only triggers if top probability $\ge 75\%$.
   * **Temporal Debounce ($N=2$)**: Requires 2 consecutive matching windows before sending a command to eliminate false noise.
   * **Mechanical Lockout**: If the hand is already closed, duplicate grasp commands are suppressed to avoid motor stall/overheat.

---

## 4. Package Structure

```text
interface_eeg/
│
├── dashboard/
│   ├── app.py                         # Main Streamlit Dashboard
│   └── hand_visualizer.py             # Vector kinematic hand graphic
│
├── src/
│   ├── classifier.py                  # FBCSP Deployment Pipeline class
│   ├── decision.py                    # Shared-Control & Safety Layer
│   ├── features.py                    # Filter-Bank CSP feature extraction
│   ├── preprocessing.py               # CAR + Filter + Welch PSD
│   ├── replay.py                      # Real-time sliding window streamer
│   ├── serial_controller.py           # USB/UART hardware serial manager
│   └── train_fbcsp.py                 # FBCSP training pipeline
│
├── data/
│   ├── test_replay_stream.npz         # 160 Hz recorded PhysioNet EEG stream
│   └── dataset_manifest.json          # Dataset metadata
│
├── models/
│   ├── fbcsp_deployment_pipeline.joblib  # Trained model weights
│   ├── pipeline_metadata.json         # Channel and sampling metadata
│   └── training_report.json           # Accuracy & validation metrics
│
├── firmware/
│   └── prosthetic_arm_controller.ino  # Ready-to-flash STM32/Arduino firmware
│
├── requirements.txt                   # Python dependencies
├── run_interface.bat                  # 1-click Windows launcher
├── HARDWARE_INTEGRATION_GUIDE.md      # Pinouts, wiring diagrams, and troubleshooting
└── README_FOR_TEAMMATE.md             # This documentation
```

---

## 5. Additional Documentation
- See [`HARDWARE_INTEGRATION_GUIDE.md`](HARDWARE_INTEGRATION_GUIDE.md) for complete wiring schematics, STM32 vs. Arduino pinouts, power safety, and USB-UART debugging tips.
