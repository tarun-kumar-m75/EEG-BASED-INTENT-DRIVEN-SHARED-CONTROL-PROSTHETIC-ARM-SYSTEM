"""
src/serial_controller.py
========================
Serial Interface for STM32 Microcontroller & Prosthetic Actuator.
Provides both Physical Hardware Mode (USB/UART) and Software Simulation Mode.
Measures true transmission latency without faking.
"""

import time
import serial
import serial.tools.list_ports
from typing import Dict, List, Optional, Tuple, Any

def compute_checksum(packet_str: str) -> int:
    """Calculates 8-bit XOR checksum across packet string."""
    chk = 0
    for ch in packet_str:
        chk ^= ord(ch)
    return chk

class STM32ProstheticController:
    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 0.1,
        mode: str = "SIMULATION"
    ):
        """
        Args:
            port: Serial COM port (e.g. 'COM3' or '/dev/ttyUSB0').
            baudrate: 115200 default.
            mode: 'SIMULATION' or 'HARDWARE'.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.mode = mode
        
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        
        self.last_command_sent = "NONE"
        self.last_packet_raw = "NONE"
        self.last_comm_latency_ms = 0.0
        self.total_packets_sent = 0
        self.last_ack_response = "SIMULATED_ACK"
        
        if self.mode == "HARDWARE" and self.port:
            self.connect(self.port)

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """Returns list of all available serial ports on the system."""
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append({
                "device": p.device,
                "description": p.description,
                "hwid": p.hwid,
                "is_stm32": ("STM" in p.description or "STLink" in p.description or "USB Serial" in p.description)
            })
        return ports

    def connect(self, port_name: str) -> bool:
        """Opens serial port connection."""
        self.port = port_name
        if self.mode == "SIMULATION":
            self.is_connected = True
            return True
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(0.05)  # Allow port settling
            self.is_connected = True
            print(f"[OK] Connected to hardware port: {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            print(f"[!] Failed to connect to serial port {port_name}: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Closes serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False

    def send_command(self, command: str) -> Dict[str, Any]:
        """
        Sends structured command packet to STM32.
        Command: 'GRASP', 'RELEASE', or 'HOLD'.
        
        Measures actual round-trip or transmission latency.
        """
        t_start = time.perf_counter_ns()
        
        # Build protocol frame: <CMD:GRASP,CRC:0xXX>\n
        crc = compute_checksum(command)
        packet = f"<CMD:{command},CRC:0x{crc:02X}>\n"
        
        ack_received = "NONE"
        comm_latency_ms = 0.0
        
        if self.mode == "HARDWARE" and self.serial_conn and self.serial_conn.is_open:
            try:
                # Transmit bytes
                self.serial_conn.write(packet.encode('ascii'))
                self.serial_conn.flush()
                
                # Check for ACK with timeout
                line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
                t_end = time.perf_counter_ns()
                comm_latency_ms = (t_end - t_start) / 1e6
                
                ack_received = line if line else "TIMEOUT_NO_ACK"
            except Exception as e:
                t_end = time.perf_counter_ns()
                comm_latency_ms = (t_end - t_start) / 1e6
                ack_received = f"ERROR: {e}"
        else:
            # Simulation Mode: Measure pure software framing & mock ACK
            time.sleep(0.001)  # 1ms simulated UART delay
            t_end = time.perf_counter_ns()
            comm_latency_ms = (t_end - t_start) / 1e6
            ack_received = f"<ACK:{command}>"
            
        self.last_command_sent = command
        self.last_packet_raw = packet.strip()
        self.last_comm_latency_ms = round(comm_latency_ms, 3)
        self.last_ack_response = ack_received
        self.total_packets_sent += 1
        
        return {
            "mode": self.mode,
            "command": command,
            "packet": packet.strip(),
            "ack": ack_received,
            "latency_ms": self.last_comm_latency_ms,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    controller = STM32ProstheticController(mode="SIMULATION")
    ports = controller.list_available_ports()
    print("[*] Available Ports Detected:", ports)
    res = controller.send_command("GRASP")
    print(f"[OK] Simulation Command Result: {res}")
