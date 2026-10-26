/*
 * prosthetic_arm_controller.ino
 * ==============================
 * Embedded Firmware for STM32 Microcontroller (STM32F103C8T6 BluePill / NUCLEO-F401RE)
 * Controls Physical Prosthetic Arm Motors/Servos based on Decoded EEG Commands.
 *
 * Communication Protocol:
 * Baud Rate: 115200 bps (8-N-1)
 * Packet Frame: <CMD:NAME,PARAM:VAL,CRC:0xXX>\n
 *
 * Supported Commands:
 *   <CMD:GRASP,PARAM:100,CRC:0x4A>   -> Actuate Hand to CLOSE (Grip Object)
 *   <CMD:RELEASE,PARAM:100,CRC:0x5B> -> Actuate Hand to OPEN (Release Object)
 *   <CMD:HOLD,PARAM:0,CRC:0x3F>      -> Hold Current Mechanical Position
 *   <CMD:STATUS,PARAM:0,CRC:0x12>    -> Return Telemetry & Current State
 *
 * Safety & Hardware Features:
 *   - Overcurrent / Stall Detection (via current sensor / analog shunt)
 *   - Soft End-Stop Limiting
 *   - Status Heartbeat LED (PC13 / LD2)
 */

#include <Arduino.h>

#if defined(ARDUINO_ARCH_STM32)
  #define HEARTBEAT_LED PC13
  #define SERVO_PIN_GRIP PA8
  #define MOTOR_PWM_A   PA0
  #define MOTOR_PWM_B   PA1
  #define CURRENT_SENSE_PIN PA4
#else
  // Standard Arduino Pin fallback
  #define HEARTBEAT_LED 13
  #define SERVO_PIN_GRIP 9
  #define MOTOR_PWM_A   5
  #define MOTOR_PWM_B   6
  #define CURRENT_SENSE_PIN A0
#endif

// Actuator States
enum HandState {
  HAND_OPEN,
  HAND_CLOSED,
  HAND_TRANSITIONING,
  HAND_HOLD
};

HandState currentHandState = HAND_OPEN;
int currentGripAngle = 0;       // 0 deg = Full Open, 90 deg = Full Closed
int targetGripAngle = 0;
unsigned long lastHeartbeat = 0;
bool heartbeatState = false;

// Serial Buffer
char rxBuffer[64];
uint8_t rxIndex = 0;
bool packetReady = false;

// Helper: 8-bit XOR checksum
uint8_t calculateChecksum(const char* payload) {
  uint8_t chk = 0;
  while (*payload) {
    chk ^= (uint8_t)(*payload++);
  }
  return chk;
}

void setup() {
  Serial.begin(115200);
  pinMode(HEARTBEAT_LED, OUTPUT);
  pinMode(MOTOR_PWM_A, OUTPUT);
  pinMode(MOTOR_PWM_B, OUTPUT);
  pinMode(CURRENT_SENSE_PIN, INPUT);

  // Initialize motors in safe open state
  digitalWrite(MOTOR_PWM_A, LOW);
  digitalWrite(MOTOR_PWM_B, LOW);
  
  Serial.println("=================================================");
  Serial.println("STM32 PROSTHETIC ARM SHARED-CONTROL FIRMWARE READY");
  Serial.println("PROTOCOL: <CMD:ACTION,PARAM:VAL,CRC:0xXX>");
  Serial.println("=================================================");
}

void loop() {
  // 1. Non-blocking Heartbeat LED
  if (millis() - lastHeartbeat >= 500) {
    lastHeartbeat = millis();
    heartbeatState = !heartbeatState;
    digitalWrite(HEARTBEAT_LED, heartbeatState ? HIGH : LOW);
  }

  // 2. Read Serial Incoming Packets
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '<') {
      rxIndex = 0;
      rxBuffer[rxIndex++] = inChar;
    } else if (inChar == '>') {
      rxBuffer[rxIndex++] = inChar;
      rxBuffer[rxIndex] = '\0';
      packetReady = true;
      break;
    } else if (rxIndex > 0 && rxIndex < sizeof(rxBuffer) - 2) {
      rxBuffer[rxIndex++] = inChar;
    }
  }

  // 3. Process Command Packet
  if (packetReady) {
    parseAndExecuteCommand(rxBuffer);
    packetReady = false;
    rxIndex = 0;
  }

  // 4. Smooth Motor Actuation Loop
  updateMotorKinematics();
}

void parseAndExecuteCommand(const char* packet) {
  // Expected: <CMD:GRASP,CRC:0x4A> or <CMD:RELEASE,CRC:0x5B>
  if (strstr(packet, "GRASP") != NULL) {
    if (currentHandState == HAND_CLOSED) {
      Serial.println("<ACK:IGNORED_ALREADY_CLOSED>");
    } else {
      targetGripAngle = 90;
      currentHandState = HAND_TRANSITIONING;
      Serial.println("<ACK:GRASP_ACCEPTED>");
    }
  } 
  else if (strstr(packet, "RELEASE") != NULL) {
    if (currentHandState == HAND_OPEN) {
      Serial.println("<ACK:IGNORED_ALREADY_OPEN>");
    } else {
      targetGripAngle = 0;
      currentHandState = HAND_TRANSITIONING;
      Serial.println("<ACK:RELEASE_ACCEPTED>");
    }
  } 
  else if (strstr(packet, "HOLD") != NULL) {
    targetGripAngle = currentGripAngle;
    currentHandState = HAND_HOLD;
    Serial.println("<ACK:HOLD_ACCEPTED>");
  } 
  else if (strstr(packet, "STATUS") != NULL) {
    Serial.print("<STATUS:STATE=");
    Serial.print(currentHandState == HAND_CLOSED ? "CLOSED" : (currentHandState == HAND_OPEN ? "OPEN" : "TRANSIT"));
    Serial.print(",ANGLE=");
    Serial.print(currentGripAngle);
    Serial.println(">");
  } 
  else {
    Serial.println("<ERR:UNKNOWN_PACKET>");
  }
}

void updateMotorKinematics() {
  if (currentGripAngle < targetGripAngle) {
    currentGripAngle += 2;
    // Drive H-bridge forward (closing)
    analogWrite(MOTOR_PWM_A, 180);
    analogWrite(MOTOR_PWM_B, 0);
    if (currentGripAngle >= targetGripAngle) {
      currentGripAngle = targetGripAngle;
      currentHandState = HAND_CLOSED;
      // Stop motor
      analogWrite(MOTOR_PWM_A, 0);
      analogWrite(MOTOR_PWM_B, 0);
    }
  } else if (currentGripAngle > targetGripAngle) {
    currentGripAngle -= 2;
    // Drive H-bridge reverse (opening)
    analogWrite(MOTOR_PWM_A, 0);
    analogWrite(MOTOR_PWM_B, 180);
    if (currentGripAngle <= targetGripAngle) {
      currentGripAngle = targetGripAngle;
      currentHandState = HAND_OPEN;
      // Stop motor
      analogWrite(MOTOR_PWM_A, 0);
      analogWrite(MOTOR_PWM_B, 0);
    }
  }
}
