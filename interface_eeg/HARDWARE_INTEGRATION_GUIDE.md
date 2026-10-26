# Hardware Integration & Wiring Guide: Prosthetic Arm Actuation

This guide provides technical specifications, wiring schematics, and serial communication details for integrating the physical prosthetic arm with the EEG dashboard.

---

## 1. System Overview Architecture

```
[PhysioNet EEG Dataset Stream (160 Hz)]
               │
               ▼
[Streamlit Processing Engine (PC / Laptop)]
   ├─ Preprocessing (CAR + 8-30 Hz Filter)
   ├─ Feature Extraction (5-Subband FBCSP)
   ├─ Classifier (Shrinkage Regularized LDA)
   └─ Shared-Control Layer (Confidence Gate + Debounce + Lockout)
               │
               ▼ USB Virtual COM Port (115200 Baud, 8-N-1)
[Microcontroller: STM32F103 / NUCLEO-F401RE / Arduino Uno]
   ├─ Serial Ring Buffer & CRC Packet Parser
   ├─ Motor / Servo PWM Generator
   └─ Soft End-Stop & Overcurrent Protection
               │
               ▼ PWM / H-Bridge Driver
[Prosthetic Arm Actuators (Linear Actuator / DC Gearmotor / SG90 / MG996R Servos)]
```

---

## 2. Microcontroller Pinout & Wiring

The firmware `firmware/prosthetic_arm_controller.ino` supports both STM32 (STM32Duino) and standard Arduino boards.

### Option A: STM32 Blue Pill (`STM32F103C8T6`) or NUCLEO-F401RE
| Function | STM32 Pin | Connects To | Notes |
| :--- | :--- | :--- | :--- |
| **USB Serial** | Built-in USB / PA9 (TX), PA10 (RX) | PC USB Port | 115200 Baud |
| **Motor Forward (Close)** | `PA0` (TIM2_CH1 PWM) | H-Bridge `IN1` / `PWM_A` | Closing drive signal |
| **Motor Reverse (Open)** | `PA1` (TIM2_CH2 PWM) | H-Bridge `IN2` / `PWM_B` | Opening drive signal |
| **Servo Signal (Alternate)** | `PA8` (TIM1_CH1 PWM) | Servo Signal (Orange/Yellow) | $0^\circ = \text{Open}$, $90^\circ = \text{Closed}$ |
| **Heartbeat LED** | `PC13` (Onboard LED) | Built-in LED | Blinks every 500ms when alive |
| **Current Sensor (Optional)** | `PA4` (ADC1_IN4) | ACS712 / Shunt Output | Overcurrent stall protection |
| **Logic Ground** | `GND` | Motor Driver `GND` & Sensor `GND` | **Common Ground is Mandatory!** |

### Option B: Arduino Uno / Mega / Nano
| Function | Arduino Pin | Connects To | Notes |
| :--- | :--- | :--- | :--- |
| **USB Serial** | Built-in USB / D0 (RX), D1 (TX) | PC USB Port | 115200 Baud |
| **Motor Forward (Close)** | `Pin 5` (PWM) | H-Bridge `IN1` | Closing drive signal |
| **Motor Reverse (Open)** | `Pin 6` (PWM) | H-Bridge `IN2` | Opening drive signal |
| **Servo Signal (Alternate)** | `Pin 9` (PWM) | Servo Signal | $0^\circ = \text{Open}$, $90^\circ = \text{Closed}$ |
| **Heartbeat LED** | `Pin 13` | Built-in LED | Blinks every 500ms when alive |
| **Current Sensor** | `Pin A0` | Current Sensor Analog Out | Optional |
| **Logic Ground** | `GND` | Driver `GND` | **Common Ground is Mandatory!** |

> [!CAUTION]
> **Power Supply Isolation**: Never power DC motors or high-torque servos directly from the 5V/3.3V pin of the microcontroller or PC USB! Use an external power supply (e.g. 5V–7.4V battery pack or 12V bench supply with step-down buck converter) to power the motors, and connect all Ground (`GND`) wires together.

---

## 3. Communication Protocol Specification

- **Baud Rate**: `115200 bps`
- **Data Bits**: 8
- **Parity**: None
- **Stop Bits**: 1
- **Line Ending**: Line Feed (`\n`) or Carriage Return + Line Feed (`\r\n`)

### Packets Sent by PC Interface to Microcontroller:
```text
<CMD:GRASP,CRC:0x4A>\n      -> Closes hand / fingers
<CMD:RELEASE,CRC:0x5B>\n    -> Opens hand / fingers
<CMD:HOLD,CRC:0x3F>\n       -> Disables motor drive / maintains current hold
<CMD:STATUS,CRC:0x12>\n     -> Requests telemetry packet
```

### Packets Sent by Microcontroller back to PC Interface:
```text
<ACK:GRASP_ACCEPTED>\n              -> Acknowledges grasp command and begins actuation
<ACK:RELEASE_ACCEPTED>\n            -> Acknowledges release command and begins actuation
<ACK:IGNORED_ALREADY_CLOSED>\n      -> Safety lockout: hand is already closed, command suppressed
<ACK:IGNORED_ALREADY_OPEN>\n        -> Safety lockout: hand is already open, command suppressed
<STATUS:STATE=CLOSED,ANGLE=90>\n    -> Telemetry response
```

---

## 4. How Actuation is Triggered

There are **two distinct methods** to trigger physical movement from the interface:

1. **Autonomous EEG-Driven Triggering**:
   - The user clicks **`[Start Stream]`** in the dashboard.
   - The system plays the calibrated continuous PhysioNet motor imagery recording.
   - When the subject imagines closing the hand, the FBCSP+LDA classifier detects the ERD/ERS oscillatory patterns in the $\mu$ (8–12 Hz) and $\beta$ (16–24 Hz) bands.
   - When confidence exceeds $75\%$ for 2 consecutive windows, the interface automatically transmits `<CMD:GRASP,CRC:0x4A>\n` over USB.
   - The microcontroller receives the command and drives the fingers closed.

2. **Direct Manual Hardware Override (Testing Mode)**:
   - In the sidebar, select your COM port and initialize the serial connection.
   - Click the manual **`[Close Hand]`** or **`[Open Hand]`** buttons.
   - The interface sends the exact same serial packets immediately, bypassing the EEG classifier.
   - Use this to verify that the motor drivers, limit switches, and mechanical tendons are functioning properly.

---

## 5. Troubleshooting & Checklist

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **"Access Denied" or Port Error** | The COM port is opened in Arduino Serial Monitor or another program. | Close the Arduino IDE Serial Monitor / PuTTY before initializing the port in the dashboard. |
| **No COM Ports in Dropdown** | USB-to-UART drivers (CH340, CP2102, FTDI, or STM32 VCP) not installed. | Install the appropriate Virtual COM Port driver for your board; replug USB cable. |
| **LED Blinks but Motors Don't Move** | Motor power supply is disconnected or common ground is missing. | Verify external power supply voltage and ensure microcontroller GND is connected to driver GND. |
| **Hand Stutters or Resets Board** | Motor inrush current is causing brownout reset on the microcontroller. | Add a $470\,\mu\text{F}–1000\,\mu\text{F}$ electrolytic capacitor across motor power rails. |
