"""Spot self-diagnosis controller with full diagnostics pipeline integration."""

from controller import Robot, Emitter, Motor, PositionSensor
import struct
import time
import math
import json
from pathlib import Path
from datetime import datetime
import sys

# Add diagnostics_pipeline to path
CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]
if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.append(str(CONTROLLERS_ROOT))

from diagnostics_pipeline import config as diag_config
from diagnostics_pipeline.pipeline import DiagnosticsPipeline
from diagnostics_pipeline.logger import DiagnosticsLogger


class SpotDiagnosticsController:
    """Spot controller for leg self-diagnosis with drone communication."""
    
    LEG_MOTOR_NAMES = {
        "FL": ["front left shoulder abduction motor", "front left shoulder rotation motor", "front left elbow motor"],
        "FR": ["front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor"],
        "RL": ["rear left shoulder abduction motor", "rear left shoulder rotation motor", "rear left elbow motor"],
        "RR": ["rear right shoulder abduction motor", "rear right shoulder rotation motor", "rear right elbow motor"],
    }
    
    def __init__(self):
        self.robot = Robot()  # Use standard Robot controller
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # Initialize emitter for communication with drone
        self.emitter = self.robot.getDevice("emitter")
        if self.emitter is None:
            print("[spot] Warning: emitter not found, communication disabled")
        else:
            self.emitter.setChannel(1)
        
        # Initialize motors and sensors for all legs
        self.motors = {}
        self.sensors = {}
        self._initialize_devices()
        
        # State tracking
        self.current_leg_index = 0
        self.current_trial = 0
        self.trial_state = "IDLE"  # IDLE, EXECUTING, MEASURING, COMPLETE
        self.trial_start_time = 0
        self.trial_data = {
            "theta_cmd": [],
            "theta_meas": [],
            "omega_meas": [],
            "tau_meas": [],
        }
        
        # Initialize diagnostics pipeline
        session_id = f"spot_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.pipeline = DiagnosticsPipeline(session_id)
        self.diag_logger = DiagnosticsLogger()
        self.session_start_time = datetime.now()
        
        print("[spot] Controller initialized")
        print(f"[spot] Time step: {self.time_step} ms")
        print(f"[spot] Will diagnose {len(diag_config.LEG_IDS)} legs, {diag_config.TRIAL_COUNT} trials each")
        print("[spot] Pipeline integration: ENABLED")
    
    def _initialize_devices(self):
        """Initialize all motors and sensors."""
        for leg_id, motor_names in self.LEG_MOTOR_NAMES.items():
            self.motors[leg_id] = []
            self.sensors[leg_id] = []
            
            for motor_name in motor_names:
                motor = self.robot.getDevice(motor_name)
                if motor:
                    self.motors[leg_id].append(motor)
                    # Get corresponding sensor
                    sensor_name = motor_name.replace("motor", "sensor")
                    sensor = self.robot.getDevice(sensor_name)
                    if sensor:
                        sensor.enable(self.time_step)
                        self.sensors[leg_id].append(sensor)
                    else:
                        self.sensors[leg_id].append(None)
                else:
                    print(f"[spot] Warning: Motor {motor_name} not found")
                    self.motors[leg_id].append(None)
                    self.sensors[leg_id].append(None)
    
    def get_safe_angle_range(self, motor, sensor, motor_index, leg_id):
        """Calculate safe angle range based on current position and motor limits.
        
        Returns:
            (safe_positive_angle, safe_negative_angle) in degrees
        """
        if not motor or not sensor:
            return (0.0, 0.0)
        
        # Get current position
        current_pos = sensor.getValue()  # radians
        current_deg = math.degrees(current_pos)
        
        # Get motor limits (min/max position)
        min_pos = motor.getMinPosition()
        max_pos = motor.getMaxPosition()
        
        # Handle infinite limits (unlimited range)
        if min_pos == float('-inf'):
            min_pos = current_pos - math.radians(30)  # Default to ±30° safety margin
        if max_pos == float('inf'):
            max_pos = current_pos + math.radians(30)
        
        # Calculate available range in both directions
        min_deg = math.degrees(min_pos)
        max_deg = math.degrees(max_pos)
        
        # Calculate safe movement range (leave 5° safety margin from limits)
        safety_margin = 5.0
        safe_positive_range = max_deg - current_deg - safety_margin
        safe_negative_range = current_deg - min_deg - safety_margin
        
        # Clamp to reasonable values (0 to 30 degrees)
        safe_positive_angle = max(0.0, min(30.0, safe_positive_range))
        safe_negative_angle = max(0.0, min(30.0, safe_negative_range))
        
        print(f"[spot] {leg_id} motor[{motor_index}]: current={current_deg:.1f}°, "
              f"range=[{min_deg:.1f}°, {max_deg:.1f}°], "
              f"safe_move=[+{safe_positive_angle:.1f}°, -{safe_negative_angle:.1f}°]")
        
        return (safe_positive_angle, safe_negative_angle)
    
    def send_trigger(self, leg_id, trial_index, direction, start_time, duration_ms):
        """Send trigger message to drone."""
        if self.emitter is None:
            return
        
        # Message format: "TRIGGER|leg_id|trial_index|direction|start_time|duration_ms"
        message = f"TRIGGER|{leg_id}|{trial_index}|{direction}|{start_time:.6f}|{duration_ms}"
        self.emitter.send(message.encode('utf-8'))
        print(f"[spot] Sent trigger: {leg_id} trial {trial_index} dir={direction}")
    
    def execute_trial(self, leg_id, trial_index, direction):
        """Execute a single diagnostic trial for one leg."""
        if trial_index < 1 or trial_index > diag_config.TRIAL_COUNT:
            print(f"[spot] Invalid trial index: {trial_index}")
            return False
        
        # Get motors for this leg
        leg_motors = self.motors.get(leg_id, [])
        leg_sensors = self.sensors.get(leg_id, [])
        
        if not leg_motors or not any(leg_motors):
            print(f"[spot] No motors available for leg {leg_id}")
            return False
        
        # Motor selection strategy:
        # - All legs: use shoulder abduction (index 0) - stable and reliable
        motor_index = 0  # Shoulder abduction - lateral movement
        
        if len(leg_motors) > motor_index and leg_motors[motor_index]:
            target_motor = leg_motors[motor_index]
            target_sensor = leg_sensors[motor_index] if len(leg_sensors) > motor_index else None
        else:
            print(f"[spot] Warning: Motor index {motor_index} not available for {leg_id}")
            return False
        
        # Get safe angle range based on current position and motor limits
        safe_pos_angle, safe_neg_angle = self.get_safe_angle_range(
            target_motor, target_sensor, motor_index, leg_id
        )
        
        # Determine safe movement angle based on direction
        # Use smaller angles (max 5°) for stability
        max_safe_angle = 5.0  # Conservative limit to prevent tipping
        
        if direction == "+":
            if safe_pos_angle < 0.5:
                print(f"[spot] {leg_id}: Cannot move positive (only {safe_pos_angle:.2f}° available)")
                return False
            safe_angle = min(safe_pos_angle, max_safe_angle)
            sign = 1.0
        else:
            if safe_neg_angle < 0.5:
                print(f"[spot] {leg_id}: Cannot move negative (only {safe_neg_angle:.2f}° available)")
                return False
            safe_angle = min(safe_neg_angle, max_safe_angle)
            sign = -1.0
        
        angle_rad = math.radians(safe_angle * sign)
        
        print(f"[spot] {leg_id} Trial {trial_index}: Using safe angle {safe_angle:.2f}° ({direction})")
        
        # Send trigger to drone BEFORE starting movement
        current_time = self.robot.getTime()
        duration_ms = int(diag_config.TRIAL_DURATION_S * 1000)
        self.send_trigger(leg_id, trial_index, direction, current_time, duration_ms)
        
        # Reset trial data
        self.trial_data = {
            "theta_cmd": [],
            "theta_meas": [],
            "omega_meas": [],
            "tau_meas": [],
        }
        self.trial_start_time = current_time
        self.trial_state = "EXECUTING"
        
        if target_motor and target_sensor:
            # Get initial position
            initial_pos = target_sensor.getValue()
            target_pos = initial_pos + angle_rad
            
            # Debug log
            print(f"[spot] {leg_id} Trial {trial_index}: initial={math.degrees(initial_pos):.3f}°, "
                  f"target={math.degrees(target_pos):.3f}°, change={math.degrees(angle_rad):.3f}°")
            
            # Set target position with conservative velocity for stability
            target_motor.setPosition(target_pos)
            # Use 20% velocity for stable, safe movement
            target_motor.setVelocity(target_motor.getMaxVelocity() * 0.2)
        
        return True
    
    def _reset_leg_position(self, leg_id):
        """Reset leg motor to initial position (0°) after trial.
        This prevents cumulative angle issues causing physical constraints."""
        leg_motors = self.motors.get(leg_id, [])
        
        # Determine which motor was used (same logic as execute_trial)
        if leg_id == "FL":
            motor_index = 1  # Shoulder rotation
        else:
            motor_index = 0  # Shoulder abduction
        
        if len(leg_motors) > motor_index and leg_motors[motor_index]:
            target_motor = leg_motors[motor_index]
            # Reset to 0° position
            target_motor.setPosition(0.0)
            # Use same velocity as trials for smooth reset
            target_motor.setVelocity(target_motor.getMaxVelocity() * 0.10)
            print(f"[spot] {leg_id}: Resetting motor {motor_index} to 0°")
        else:
            print(f"[spot] Warning: Cannot reset {leg_id}, motor {motor_index} not available")
    
    def measure_trial_data(self, leg_id):
        """Collect sensor data during trial execution."""
        leg_sensors = self.sensors.get(leg_id, [])
        if not leg_sensors or not any(leg_sensors):
            return
        
        # Measure from the same motor index used in execute_trial
        # FL uses rotation (index 1), others use abduction (index 0)
        if leg_id == "FL":
            motor_index = 1
        else:
            motor_index = 0
        
        if len(leg_sensors) > motor_index and leg_sensors[motor_index]:
            sensor = leg_sensors[motor_index]
        else:
            print(f"[spot] Warning: No sensor {motor_index} available for {leg_id}")
            return
        
        if sensor:
            current_pos = sensor.getValue()
            self.trial_data["theta_meas"].append(math.degrees(current_pos))
            
            # Estimate velocity (simple numerical derivative)
            if len(self.trial_data["theta_meas"]) > 1:
                dt = self.time_step / 1000.0
                vel = (self.trial_data["theta_meas"][-1] - self.trial_data["theta_meas"][-2]) / dt
                self.trial_data["omega_meas"].append(vel)
            else:
                self.trial_data["omega_meas"].append(0.0)
            
            # Simulate torque measurement (in real robot, would read from torque sensor)
            # Here we estimate based on position error
            if len(self.trial_data["theta_cmd"]) > 0:
                error = abs(self.trial_data["theta_cmd"][-1] - math.degrees(current_pos))
                estimated_torque = error * 5.0  # Reduced factor for smaller movements
                self.trial_data["tau_meas"].append(estimated_torque)
            else:
                self.trial_data["tau_meas"].append(0.0)
    
    def _simulate_drone_observation(self, leg_id):
        """Generate RoboPose observations with enhanced sampling (no prior knowledge).
        Uses relative displacement from initial position to handle various terrains/postures."""
        # Use the measured theta values from the trial data as time-series
        measured_angles = self.trial_data.get("theta_meas", [])
        
        if not measured_angles:
            print(f"[spot] Warning: No measured angles for {leg_id}")
            return
        
        # Get sensors for this leg (for reference joint count)
        leg_sensors = self.sensors.get(leg_id, [])
        num_joints = len(leg_sensors)
        
        # Determine which motor was used for this leg (same logic as execute_trial)
        # FL uses rotation (index 1), others use abduction (index 0)
        motor_index = 1 if leg_id == "FL" else 0
        
        # Store initial angle for relative displacement calculation
        initial_shoulder_angle = measured_angles[0]
        
        # Generate MORE frames from measured data for better temporal resolution
        # Increase from ROBOPOSE_FPS_TRIGGER to 3x for multi-viewpoint simulation
        num_samples = len(measured_angles)
        target_frames = int(diag_config.TRIAL_DURATION_S * diag_config.ROBOPOSE_FPS_TRIGGER * 3)
        
        # Sample indices evenly from measured data
        if num_samples < target_frames:
            # Use all available samples
            frame_indices = range(num_samples)
        else:
            # Sample evenly with higher density
            step = num_samples / target_frames
            frame_indices = [int(i * step) for i in range(target_frames)]
        
        # Calculate initial foot position for relative displacement
        # Build joint angles array with measured angle at correct motor_index
        initial_joint_angles = []
        for joint_idx in range(num_joints):
            if joint_idx == motor_index:
                initial_joint_angles.append(initial_shoulder_angle)
            elif joint_idx < len(leg_sensors) and leg_sensors[joint_idx]:
                initial_joint_angles.append(math.degrees(leg_sensors[joint_idx].getValue()))
            else:
                initial_joint_angles.append(0.0)
        
        # Pad to 3 joints if needed
        while len(initial_joint_angles) < 3:
            initial_joint_angles.append(0.0)
        
        # Calculate initial foot position
        L1, L2, L3 = 0.11, 0.26, 0.26  # Spot leg segment lengths (approximate)
        theta1_init = math.radians(initial_joint_angles[0])
        theta2_init = math.radians(initial_joint_angles[1])
        theta3_init = math.radians(initial_joint_angles[2])
        
        x_init = L2 * math.cos(theta2_init) + L3 * math.cos(theta2_init + theta3_init)
        y_offset = -0.25 if 'L' in leg_id else 0.25  # Left or Right
        y_init = y_offset + L1 * math.sin(theta1_init)
        z_init = -(L2 * math.sin(theta2_init) + L3 * math.sin(theta2_init + theta3_init))
        
        for frame_idx in frame_indices:
            # Base position and orientation (estimated from IMU/sensors)
            # In real RoboPose, these would come from visual observation
            base_position = [0.0, 0.0, 0.3]  # Spot base height
            base_orientation = [0.0, 0.0, 0.0]  # roll, pitch, yaw
            
            # Get joint angles from measured data at this time point
            shoulder_angle_deg = measured_angles[frame_idx]
            
            # Build joint angles array with measured angle at correct motor_index
            joint_angles = []
            for joint_idx in range(num_joints):
                if joint_idx == motor_index:
                    joint_angles.append(shoulder_angle_deg)
                elif joint_idx < len(leg_sensors) and leg_sensors[joint_idx]:
                    joint_angles.append(math.degrees(leg_sensors[joint_idx].getValue()))
                else:
                    joint_angles.append(0.0)
            
            # Pad to 3 joints if needed
            while len(joint_angles) < 3:
                joint_angles.append(0.0)
            
            # Estimate foot position from joint angles (forward kinematics)
            # Convert angles to radians for calculation
            theta1 = math.radians(joint_angles[0])
            theta2 = math.radians(joint_angles[1])
            theta3 = math.radians(joint_angles[2])
            
            # Simplified forward kinematics (absolute position)
            x_abs = L2 * math.cos(theta2) + L3 * math.cos(theta2 + theta3)
            y_abs = y_offset + L1 * math.sin(theta1)
            z_abs = -(L2 * math.sin(theta2) + L3 * math.sin(theta2 + theta3))
            
            # Calculate RELATIVE displacement from initial position
            # This makes the system robust to different initial postures
            x_rel = x_abs - x_init
            y_rel = y_abs - y_init
            z_rel = z_abs - z_init
            
            # Send relative position (displacement from start) to pipeline
            end_position = [x_rel, y_rel, z_rel]
            
            # Send to pipeline
            self.pipeline.record_robo_pose_frame(
                leg_id=leg_id,
                joint_angles=joint_angles,
                end_position=end_position,
                base_orientation=base_orientation,
                base_position=base_position
            )
        
        # Debug log to verify forward kinematics and relative displacement
        angle_range = max(measured_angles) - min(measured_angles)
        print(f"[spot] {leg_id}: Generated {len(frame_indices)} RoboPose frames from {num_samples} measurements")
        print(f"[spot] {leg_id}: Angle range: {angle_range:.3f}° (initial={initial_shoulder_angle:.3f}°, final={measured_angles[-1]:.3f}°)")
    
    def finalize_and_output_results(self):
        """Finalize pipeline and output diagnostic results."""
        # Finalize pipeline to compute all probabilities
        session_record = self.pipeline.finalize()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC RESULTS")
        print("="*80)
        
        # Display results for each leg
        for leg_id in diag_config.LEG_IDS:
            leg_state = session_record.legs.get(leg_id)
            if not leg_state:
                print(f"\n[{leg_id}] No data available")
                continue
            
            print(f"\n[{leg_id}] Diagnosis Summary:")
            print(f"  Trials completed: {len(leg_state.trials)}/{diag_config.TRIAL_COUNT}")
            
            # Self-diagnosis results
            print(f"  Self-diagnosis:")
            print(f"    Can-move probability: {leg_state.self_can:.3f}")
            print(f"    Status: {'OK' if leg_state.self_can >= diag_config.SELF_CAN_THRESHOLD else 'ABNORMAL'}")
            
            # Drone observation results
            if leg_state.drone_can is not None:
                print(f"  Drone observation:")
                print(f"    Can-move probability: {leg_state.drone_can:.3f}")
                if leg_state.p_drone:
                    print(f"    Cause distribution: {json.dumps(leg_state.p_drone, indent=6)}")
            
            # LLM inference results
            if leg_state.p_llm:
                print(f"  LLM inference:")
                print(f"    Cause distribution: {json.dumps(leg_state.p_llm, indent=6)}")
            
            # Fused results
            if leg_state.p_final:
                print(f"  Final diagnosis (fused):")
                print(f"    Cause distribution: {json.dumps(leg_state.p_final, indent=6)}")
                # Find most likely cause
                max_cause = max(leg_state.p_final.items(), key=lambda x: x[1])
                print(f"    Most likely cause: {max_cause[0]} (p={max_cause[1]:.3f})")
        
        print("\n" + "="*80)
        
        # Save to JSONL file
        log_dir = CONTROLLERS_ROOT / "spot_self_diagnosis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = log_dir / f"diagnosis_{timestamp}.jsonl"
        
        with open(jsonl_path, 'w') as f:
            # Write session metadata
            metadata = {
                "type": "session_metadata",
                "start_time": self.session_start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "num_legs": len(diag_config.LEG_IDS),
                "trials_per_leg": diag_config.TRIAL_COUNT,
            }
            f.write(json.dumps(metadata) + '\n')
            
            # Write each leg's results
            for leg_id in diag_config.LEG_IDS:
                leg_state = session_record.legs.get(leg_id)
                if leg_state:
                    leg_record = {
                        "type": "leg_result",
                        "leg_id": leg_id,
                        "self_can": leg_state.self_can,
                        "drone_can": leg_state.drone_can,
                        "p_drone": leg_state.p_drone,
                        "p_llm": leg_state.p_llm,
                        "p_final": leg_state.p_final,
                        "cause_final": leg_state.cause_final,
                        "conf_final": leg_state.conf_final,
                        "num_trials": len(leg_state.trials),
                    }
                    f.write(json.dumps(leg_record) + '\n')
        
        print(f"\n[spot] Results saved to: {jsonl_path}")
        print("[spot] Diagnostics complete!")
    
    def run(self):
        """Main control loop with full pipeline integration."""
        print("[spot] Starting self-diagnosis sequence")
        
        # Wait for simulation to stabilize
        for _ in range(50):
            if self.robot.step(self.time_step) == -1:
                return
        
        # Main diagnosis loop
        for leg_index, leg_id in enumerate(diag_config.LEG_IDS):
            print(f"\n[spot] === Diagnosing leg {leg_id} ({leg_index + 1}/{len(diag_config.LEG_IDS)}) ===")
            
            for trial_index in range(1, diag_config.TRIAL_COUNT + 1):
                # Reverse trial pattern for RL leg to avoid ground collision
                # RL leg starts at -3.4° (tilted inward), so negative direction works better first
                if leg_id == "RL":
                    reversed_pattern = ["-", "-", "+", "+"]
                    direction = reversed_pattern[trial_index - 1]
                else:
                    direction = diag_config.TRIAL_PATTERN[trial_index - 1]
                
                print(f"[spot] Leg {leg_id} - Trial {trial_index}/{diag_config.TRIAL_COUNT} (dir={direction})")
                
                # Start trial in pipeline
                trial_start_time = self.robot.getTime()
                self.pipeline.start_trial(leg_id, trial_index, direction, trial_start_time)
                
                # Execute trial
                if not self.execute_trial(leg_id, trial_index, direction):
                    print(f"[spot] Failed to execute trial for {leg_id}")
                    continue
                
                # Measure data during trial duration
                trial_steps = int((diag_config.TRIAL_DURATION_S * 1000) / self.time_step)
                for step in range(trial_steps):
                    if self.robot.step(self.time_step) == -1:
                        return
                    
                    # Generate commanded angle for this timestep
                    progress = (step + 1) / trial_steps
                    sign = 1.0 if direction == "+" else -1.0
                    # Use actual safe angle (3 degrees)
                    theta_cmd = sign * 3.0 * progress
                    self.trial_data["theta_cmd"].append(theta_cmd)
                    
                    # Measure actual sensor values
                    self.measure_trial_data(leg_id)
                
                # Simulate drone observation frames BEFORE completing trial
                # (In real system, these would accumulate during movement via receiver)
                self._simulate_drone_observation(leg_id)
                
                # Complete trial in pipeline with collected data
                end_time = self.robot.getTime()
                tau_nominal = 10.0  # Nominal torque for this movement
                safety_level = "SAFE"  # Movement is considered safe
                
                self.pipeline.complete_trial(
                    leg_id=leg_id,
                    theta_cmd=self.trial_data["theta_cmd"],
                    theta_meas=self.trial_data["theta_meas"],
                    omega_meas=self.trial_data["omega_meas"],
                    tau_meas=self.trial_data["tau_meas"],
                    tau_nominal=tau_nominal,
                    safety_level=safety_level,
                    end_time=end_time
                )
                
                print(f"[spot] Trial {trial_index} complete - collected {len(self.trial_data['theta_meas'])} samples")
                
                # Small pause between trials
                for _ in range(10):
                    if self.robot.step(self.time_step) == -1:
                        return
        
        print("\n[spot] All legs diagnosed - finalizing pipeline...")
        
        # Finalize pipeline and get results
        self.finalize_and_output_results()
        
        # Idle loop
        while self.robot.step(self.time_step) != -1:
            pass
            pass


def main():
    controller = SpotDiagnosticsController()
    controller.run()


if __name__ == "__main__":
    main()
