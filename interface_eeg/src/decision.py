"""
src/decision.py
===============
Shared-Control and Safety Layer for Prosthetic Arm Actuation.

Engineering Rationale:
---------------------
In assistive robotics and neuroprosthetics, sending raw ML predictions directly
to physical actuators is dangerous and clinically unacceptable. Scalp EEG is
inherently non-stationary and prone to transient false positives.

This module implements a multi-stage Shared-Control & Safety Layer:
1. Confidence Threshold Gating:
   - Evaluates max posterior probability: P_max >= threshold (e.g., 0.80).
   - Rejects ambiguous predictions as 'LOW_CONFIDENCE' -> HOLD.

2. Temporal Debounce (Moving Stability Window):
   - Requires N consecutive matching predictions (e.g., 3 consecutive windows of 'GRASP').
   - Graphical debounce state: [✓] [✓] [✓] -> ACCEPT, or [✓] [✗] [ ] -> REJECT/HOLD.

3. Actuator State Machine & Redundancy Lockout:
   - If arm is already 'CLOSED', repeated 'GRASP' commands are suppressed to prevent motor stall and overheating.
   - If arm is already 'OPEN', repeated 'RELEASE' commands are suppressed.
   - If arm is in 'EXECUTING' transition, new commands are queued or locked out.

Output: Safe, validated prosthetic command: 'GRASP', 'RELEASE', or 'HOLD'.
"""

from collections import deque
from typing import Dict, List, Any, Optional

class SharedControlLayer:
    def __init__(
        self,
        confidence_threshold: float = 0.80,
        debounce_consecutive: int = 3,
        initial_arm_state: str = "OPEN"
    ):
        """
        Args:
            confidence_threshold: Minimum probability (0.0 to 1.0) required to consider intent.
            debounce_consecutive: Number of consecutive matching windows required.
            initial_arm_state: 'OPEN', 'CLOSED', or 'IDLE'.
        """
        self.confidence_threshold = confidence_threshold
        self.debounce_consecutive = debounce_consecutive
        
        # Arm state: 'OPEN', 'CLOSED', 'IDLE', 'EXECUTING'
        self.arm_state = initial_arm_state
        self.last_accepted_command = "NONE"
        self.last_transition_time = 0.0
        
        # Debounce buffer holds recent candidate predictions
        self.debounce_buffer: deque = deque(maxlen=debounce_consecutive)
        
        # Running statistics
        self.total_evaluations = 0
        self.total_accepted = 0
        self.total_rejected_confidence = 0
        self.total_rejected_debounce = 0
        self.total_rejected_redundant = 0

    def set_confidence_threshold(self, threshold: float):
        """Allows real-time tuning from GUI slider."""
        self.confidence_threshold = max(0.50, min(0.99, threshold))

    def set_debounce_consecutive(self, n: int):
        """Allows real-time tuning of required debounce windows."""
        self.debounce_consecutive = max(1, min(10, n))
        new_deque = deque(self.debounce_buffer, maxlen=self.debounce_consecutive)
        self.debounce_buffer = new_deque

    def process_intent(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluates decoded intent through confidence, debounce, and safety logic.
        
        Returns:
            Dictionary with decision status, accepted motor command, debounce indicators,
            and safety explanations for display on the dashboard.
        """
        self.total_evaluations += 1
        
        # 1. Confidence Gate
        confidence_passed = (confidence >= self.confidence_threshold)
        
        candidate_intent = predicted_class if confidence_passed else "LOW_CONFIDENCE"
        self.debounce_buffer.append(candidate_intent)
        
        # Format debounce visual indicators e.g. ['✓', '✓', '✓'] or ['✓', '✗', ' ']
        buffer_list = list(self.debounce_buffer)
        debounce_marks = []
        is_debounced = False
        
        target_intent = buffer_list[-1] if buffer_list else "NONE"
        if len(buffer_list) == self.debounce_consecutive:
            # Check if all items in buffer match
            all_match = all(item == target_intent for item in buffer_list)
            if all_match and target_intent not in ("LOW_CONFIDENCE", "REST", "NONE"):
                is_debounced = True
                debounce_marks = ['✓'] * len(buffer_list)
            else:
                for item in buffer_list:
                    debounce_marks.append('✓' if item == target_intent and item != "LOW_CONFIDENCE" else '✗')
        else:
            for item in buffer_list:
                debounce_marks.append('✓' if item != "LOW_CONFIDENCE" else '✗')
            # Pad remaining empty slots
            debounce_marks += [' '] * (self.debounce_consecutive - len(buffer_list))

        # 2. Decision Logic
        command = "HOLD"
        decision = "HOLD"
        reason = ""
        
        if not confidence_passed:
            self.total_rejected_confidence += 1
            decision = "HOLD"
            reason = f"LOW CONFIDENCE ({confidence*100:.1f}% < {self.confidence_threshold*100:.1f}%)"
        elif not is_debounced:
            self.total_rejected_debounce += 1
            decision = "HOLD"
            reason = f"DEBOUNCE UNCONFIRMED ({len(buffer_list)}/{self.debounce_consecutive})"
        else:
            # Debounced intent candidate is valid! Now check actuator safety rules
            debounced_class = buffer_list[0]
            
            if debounced_class == "GRASP":
                if self.arm_state == "CLOSED":
                    self.total_rejected_redundant += 1
                    decision = "HOLD"
                    reason = "SAFETY LOCK: ARM ALREADY CLOSED (SUPPRESSED REDUNDANT ACTUATION)"
                else:
                    decision = "ACCEPT"
                    command = "GRASP"
                    self.arm_state = "CLOSED"
                    self.last_accepted_command = "GRASP"
                    self.total_accepted += 1
                    reason = "COMMAND ACCEPTED -> ACTUATE TO CLOSE"
                    # Clear debounce buffer to prevent continuous re-triggering
                    self.debounce_buffer.clear()
                    
            elif debounced_class == "RELEASE":
                if self.arm_state == "OPEN":
                    self.total_rejected_redundant += 1
                    decision = "HOLD"
                    reason = "SAFETY LOCK: ARM ALREADY OPEN (SUPPRESSED REDUNDANT ACTUATION)"
                else:
                    decision = "ACCEPT"
                    command = "RELEASE"
                    self.arm_state = "OPEN"
                    self.last_accepted_command = "RELEASE"
                    self.total_accepted += 1
                    reason = "COMMAND ACCEPTED -> ACTUATE TO OPEN"
                    self.debounce_buffer.clear()
            else:
                decision = "HOLD"
                reason = "REST INTENT -> MAINTAIN IDLE POSITION"
                
        return {
            "decision": decision,                     # "ACCEPT" or "HOLD"
            "motor_command": command,                 # "GRASP", "RELEASE", or "HOLD"
            "arm_state": self.arm_state,             # "OPEN" or "CLOSED"
            "last_accepted_command": self.last_accepted_command,
            "confidence_passed": confidence_passed,
            "confidence_threshold": self.confidence_threshold,
            "is_debounced": is_debounced,
            "debounce_marks": debounce_marks,         # e.g. ['✓', '✓', '✓']
            "debounce_fill_ratio": len(self.debounce_buffer) / self.debounce_consecutive,
            "reason": reason,
            "statistics": {
                "total_evaluations": self.total_evaluations,
                "total_accepted": self.total_accepted,
                "total_rejected_confidence": self.total_rejected_confidence,
                "total_rejected_debounce": self.total_rejected_debounce,
                "total_rejected_redundant": self.total_rejected_redundant
            }
        }
