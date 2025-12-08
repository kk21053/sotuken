"""Drone controller with RoboPose integration and trigger-based observation."""

from controller import Supervisor, Receiver, Emitter
import math
import sys
from pathlib import Path
import time as time_module

# Add diagnostics_pipeline to path
CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]
if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.append(str(CONTROLLERS_ROOT))

from diagnostics_pipeline import config as diag_config
from diagnostics_pipeline.pipeline import DiagnosticsPipeline
from datetime import datetime
import configparser


class RoboPoseSimulator:
    """Simulates RoboPose output from Webots sensors."""
    
    def __init__(self, supervisor, spot_def_name="SPOT"):
        self.supervisor = supervisor
        self.spot_node = supervisor.getFromDef(spot_def_name)
        if self.spot_node is None:
            print("[drone] Warning: Spot node not found")
    
    def get_observation(self):
        """Get RoboPose-style observation from Spot."""
        if self.spot_node is None:
            return None
        
        # Get base position and orientation
        position = self.spot_node.getPosition()
        orientation = self.spot_node.getOrientation()
        
        # Convert orientation matrix to roll, pitch, yaw
        roll = math.atan2(orientation[7], orientation[8])
        pitch = math.asin(-orientation[6])
        yaw = math.atan2(orientation[3], orientation[0])
        
        # Get velocity (if available)
        velocity = self.spot_node.getVelocity()
        
        # Simulate joint angles (in real RoboPose, these would be estimated from image)
        # For simulation, we'll use placeholder values
        joint_angles = [0.0, 0.0, 0.0]  # Simplified
        
        # Calculate end effector position (simplified for simulation)
        # In real system, this would come from forward kinematics
        end_position = [
            position[0] + 0.3,  # Approximate leg reach
            position[1],
            position[2] - 0.3
        ]
        
        return {
            "base_position": list(position),
            "base_orientation": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
            "joint_angles": joint_angles,
            "end_position": end_position,
            "timestamp": self.supervisor.getTime()
        }


class DroneCircularController:
    """Drone controller with RoboPose and communication."""
    
    def __init__(self):
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())
        
        # Initialize diagnostics pipeline (drone owns the pipeline)
        session_id = f"drone_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.pipeline = DiagnosticsPipeline(session_id)
        print(f"[drone] Pipeline initialized: {session_id}")
        
        # Load expected causes from scenario.ini
        self.expected_causes = self.load_expected_causes()
        print(f"[drone] Expected causes: {self.expected_causes}")
        
        # Set expected causes in pipeline
        self.pipeline.set_expected_causes(self.expected_causes)
        
        # Parse command line arguments
        arguments = sys.argv[1:]
        self.offset_x, self.offset_y, self.offset_z, self.center_def = self.parse_arguments(arguments)
        
        # Initialize RoboPose simulator
        self.robopose = RoboPoseSimulator(self.supervisor, self.center_def)
        
        # Get Spot node for customData communication
        self.spot_node = self.supervisor.getFromDef("SPOT")
        if self.spot_node is None:
            print("[drone] Warning: SPOT node not found")
            self.spot_custom_data_field = None
        else:
            self.spot_custom_data_field = self.spot_node.getField("customData")
        
        # Battery simulation (Mavic 2 Pro specs)
        # Official specs: 59.29Wh battery, 29min hover time
        # Power consumption: 59.29Wh / (29/60)h = 122.66W hovering
        # Energy per minute: 122.66W * (1/60)h = 2.044Wh/min
        self.battery_capacity_wh = 59.29  # Wh
        self.battery_current_wh = 59.29   # Start at full charge
        self.battery_hover_power_w = 122.66  # Watts during hover
        self.battery_move_power_w = 200.0  # Watts during movement (estimated)
        self.battery_last_update = 0.0
        self.battery_log_interval = 10.0  # Log every 10 seconds
        self.battery_last_log_time = 0.0  # Track last log time
        
        # Track last processed message to avoid duplicates
        self.last_custom_data = ""
        
        # Get drone node and fields for positioning
        self.drone_node = self.supervisor.getSelf()
        self.translation_field = self.drone_node.getField("translation")
        self.rotation_field = self.drone_node.getField("rotation")
        
        # Get center (Spot) node
        self.center_node = self.supervisor.getFromDef(self.center_def)
        
        # State tracking
        self.active_trials = {}  # leg_id -> trial_info
        self.observation_mode = "IDLE"  # IDLE or ACTIVE
        self.fps_current = 10.0  # 観測フレームレート（固定）
        self.last_observation_time = 0
        self.observation_interval = 1.0 / self.fps_current
        
        # Diagnosis tracking
        self.completed_legs = set()  # Legs that have completed all trials
        self.total_legs = len(diag_config.LEG_IDS)
        self.total_trials_per_leg = diag_config.TRIAL_COUNT
        self.trials_completed = {}  # leg_id -> count
        for leg_id in diag_config.LEG_IDS:
            self.trials_completed[leg_id] = 0
        
        # Load expected causes from scenario.ini
        self.expected_causes = self.load_expected_causes()
        print(f"[drone] Expected causes: {self.expected_causes}")
        
        print("[drone] Controller initialized")
        print(f"[drone] Offset: ({self.offset_x:.2f}, {self.offset_y:.2f}, {self.offset_z:.2f})")
        print(f"[drone] Center: {self.center_def}")
        print(f"[drone] RoboPose FPS: {self.fps_current}")
    
    def load_expected_causes(self):
        """Load expected causes from scenario.ini."""
        config_path = Path(__file__).parent.parent.parent / "config" / "scenario.ini"
        expected = {}
        
        if not config_path.exists():
            print(f"[drone] Warning: scenario.ini not found at {config_path}")
            return expected
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Map leg IDs to config keys
        leg_mapping = {
            "FL": "fl_environment",
            "FR": "fr_environment",
            "RL": "rl_environment",
            "RR": "rr_environment"
        }
        
        for leg_id, config_key in leg_mapping.items():
            if config.has_option("DEFAULT", config_key):
                env_state = config.get("DEFAULT", config_key)
                expected[leg_id] = env_state
            else:
                expected[leg_id] = "NONE"
        
        return expected
    
    def parse_arguments(self, arguments):
        """Parse command line arguments."""
        offset_x = 0.0
        offset_y = -2.0
        offset_z = 3.0
        center_def = "SPOT"
        
        for arg in arguments:
            if arg.startswith("--offset-x="):
                offset_x = float(arg.split("=", 1)[1])
            elif arg.startswith("--offset-y="):
                offset_y = float(arg.split("=", 1)[1])
            elif arg.startswith("--offset-z="):
                offset_z = float(arg.split("=", 1)[1])
            elif arg.startswith("--center-def="):
                center_def = arg.split("=", 1)[1]
            elif arg.startswith("--radius="):
                radius = float(arg.split("=", 1)[1])
                offset_y = -abs(radius)
            elif arg.startswith("--height="):
                offset_z = float(arg.split("=", 1)[1])
        
        return offset_x, offset_y, offset_z, center_def
    
    def process_triggers(self):
        """Process trigger messages from Spot via customData field."""
        if self.spot_custom_data_field is None:
            return
        
        # Read customData field
        custom_data = self.spot_custom_data_field.getSFString()
        
        # Skip if no new data
        if not custom_data or custom_data == self.last_custom_data:
            return
        
        # Update last processed data
        self.last_custom_data = custom_data
        
        # Process each message (newline separated)
        messages = custom_data.strip().split("\n")
        
        for message in messages:
            if not message:
                continue
            
            try:
                parts = message.split("|")
                
                if parts[0] == "TRIGGER":
                    # Trial start trigger
                    leg_id = parts[1]
                    trial_index = int(parts[2])
                    direction = parts[3]
                    start_time = float(parts[4])
                    duration_ms = int(parts[5])
                    
                    end_time = start_time + (duration_ms / 1000.0)
                    
                    # Start trial in pipeline
                    motor_index = diag_config.TRIAL_MOTOR_INDICES[trial_index - 1]
                    self.pipeline.start_trial(
                        leg_id=leg_id,
                        trial_index=trial_index,
                        direction=direction,
                        start_time=start_time,
                        duration=(duration_ms / 1000.0)
                    )
                    
                    # Store trial info
                    self.active_trials[leg_id] = {
                        "trial_index": trial_index,
                        "direction": direction,
                        "start_time": start_time,
                        "end_time": end_time,
                        "observations": [],
                        "body_positions": [],
                        "initial_body_pos": None,
                        "motor_index": motor_index  # Store for later use
                    }
                
                elif parts[0] == "JOINT_ANGLES":
                    # Joint angle observation frame
                    # Format: JOINT_ANGLES|leg_id|trial_index|angle0|angle1|angle2
                    leg_id = parts[1]
                    trial_index = int(parts[2])
                    angle0 = float(parts[3])
                    angle1 = float(parts[4])
                    angle2 = float(parts[5])
                    
                    # Store joint angles in active trial
                    if leg_id in self.active_trials:
                        trial_info = self.active_trials[leg_id]
                        if trial_info["trial_index"] == trial_index:
                            current_time = self.supervisor.getTime()
                            
                            # Create observation if none exists (to capture all joint angle updates)
                            # time_step is in milliseconds, convert to seconds for comparison
                            if not trial_info["observations"] or \
                               current_time - trial_info["observations"][-1].get("timestamp", 0) > (self.time_step / 1000.0):
                                # Create new observation with joint angles
                                observation = {
                                    "timestamp": current_time,
                                    "joint_angles": [angle0, angle1, angle2],
                                    "end_position": [0.0, 0.0, 0.0],  # Will be calculated later
                                    "base_orientation": [0.0, 0.0, 0.0],
                                }
                                trial_info["observations"].append(observation)
                                
                                # Record body position
                                if self.robopose.spot_node:
                                    body_pos = self.robopose.spot_node.getPosition()
                                    trial_info["body_positions"].append(list(body_pos))
                                    
                                    # Set initial body position if first observation
                                    if trial_info["initial_body_pos"] is None:
                                        trial_info["initial_body_pos"] = list(body_pos)
                                        print(f"[drone] Initial body position for {leg_id}: {body_pos}")
                            else:
                                # Update existing observation with joint angles
                                trial_info["observations"][-1]["joint_angles"] = [angle0, angle1, angle2]
                            
                            # Debug: print first joint angle update
                            if not hasattr(self, '_received_first_joints'):
                                self._received_first_joints = {}
                            if leg_id not in self._received_first_joints:
                                print(f"[drone] Received joint angles for {leg_id} trial {trial_index}: [{angle0:.1f}°, {angle1:.1f}°, {angle2:.1f}°]")
                                self._received_first_joints[leg_id] = True
                
                elif parts[0] == "SELF_DIAG":
                    # Self-diagnosis data from Spot
                    # Message format: SELF_DIAG|leg_id|trial_index|theta_samples|theta_avg|theta_final|tau_avg|tau_max|tau_nominal|safety|self_can_raw
                    leg_id = parts[1]
                    trial_index = int(parts[2])
                    theta_samples = int(parts[3])
                    theta_avg = float(parts[4])
                    theta_final = float(parts[5])
                    tau_avg = float(parts[6])
                    tau_max = float(parts[7])
                    tau_nominal = float(parts[8])
                    safety_level = parts[9]
                    self_can_raw = float(parts[10]) if len(parts) > 10 else 0.35  # Default if not provided
                    
                    # Store self-diagnosis data
                    # Note: trial may still be active or just completed
                    if leg_id in self.active_trials:
                        self.active_trials[leg_id]["self_diag_data"] = {
                            "theta_samples": theta_samples,
                            "theta_avg": theta_avg,
                            "theta_final": theta_final,
                            "tau_avg": tau_avg,
                            "tau_max": tau_max,
                            "tau_nominal": tau_nominal,
                            "safety_level": safety_level,
                            "self_can_raw": self_can_raw,
                        }
                        print(f"[drone] Self-diagnosis received: {leg_id} trial {trial_index}, samples={theta_samples}, self_can={self_can_raw:.3f}")
                    else:
                        print(f"[drone] Warning: Self-diagnosis for inactive trial: {leg_id}")
                
                elif parts[0] == "SPOT_CAN":
                    # 仕様ステップ5: Spotから送られたspot_canを受信
                    # Message format: SPOT_CAN|leg_id|spot_can
                    leg_id = parts[1]
                    spot_can = float(parts[2])
                    
                    # Store spot_can in pipeline's session
                    leg_state = self.pipeline.session.ensure_leg(leg_id)
                    leg_state.spot_can = spot_can
                    
                    print(f"[drone] 仕様ステップ5受信: {leg_id}のspot_can={spot_can:.3f}")
                    
                    # Track that this leg's spot_can has been received
                    if not hasattr(self, 'spot_can_received'):
                        self.spot_can_received = {}
                    self.spot_can_received[leg_id] = True
            
            except Exception as e:
                print(f"[drone] Error processing message: {e}")
                import traceback
                traceback.print_exc()
    
    def should_observe(self):
        """Determine if we should make an observation now."""
        current_time = self.supervisor.getTime()
        
        # Check if enough time has passed since last observation
        if current_time - self.last_observation_time < self.observation_interval:
            return False
        
        return True
    
    def make_observation(self):
        """Make a RoboPose observation and store it."""
        if not self.should_observe():
            return
        
        current_time = self.supervisor.getTime()
        observation = self.robopose.get_observation()
        
        if observation is None:
            return
        
        self.last_observation_time = current_time
        
        # Store observation for active trials
        for leg_id, trial_info in list(self.active_trials.items()):
            if trial_info["start_time"] <= current_time <= trial_info["end_time"]:
                # Note: joint_angles will be filled by JOINT_ANGLES message from Spot
                # Initialize with default values, will be updated when message arrives
                observation_with_joints = observation.copy()
                observation_with_joints["joint_angles"] = [0.0, 0.0, 0.0]  # Will be updated
                
                trial_info["observations"].append(observation_with_joints)
                
                # Record body position
                if self.robopose.spot_node:
                    body_pos = self.robopose.spot_node.getPosition()
                    trial_info["body_positions"].append(list(body_pos))
                    
                    # Set initial body position if first observation
                    if trial_info["initial_body_pos"] is None:
                        trial_info["initial_body_pos"] = list(body_pos)
                        print(f"[drone] Initial body position for {leg_id}: {body_pos}")
                
            elif current_time > trial_info["end_time"]:
                # Trial complete - send data
                self.send_trial_data(leg_id, trial_info)
                del self.active_trials[leg_id]
        
        # Switch back to idle mode if no active trials
        if len(self.active_trials) == 0 and self.observation_mode == "ACTIVE":
            self.observation_mode = "IDLE"
            # FPSは固定なので変更不要
            print("[drone] Switched to idle observation mode")
    
    def send_trial_data(self, leg_id, trial_info):
        """Send collected observation data to pipeline."""
        obs_count = len(trial_info["observations"])
        body_positions = trial_info["body_positions"]
        
        print(f"[drone] Processing {obs_count} observations for {leg_id} trial {trial_info['trial_index']}")
        
        if obs_count == 0:
            print(f"[drone] Warning: No observations collected for {leg_id}")
            return
        
        if len(body_positions) != obs_count:
            print(f"[drone] Warning: Body position count mismatch: {len(body_positions)} vs {obs_count}")
            return
        
        # Send each observation frame to pipeline with absolute displacement calculation
        self._send_observations_to_pipeline(leg_id, trial_info)
    
    def _send_observations_to_pipeline(self, leg_id, trial_info):
        """Calculate absolute foot displacement and send frames to pipeline."""
        observations = trial_info["observations"]
        body_positions = trial_info["body_positions"]
        initial_body_pos = trial_info["initial_body_pos"]
        
        if not observations or not body_positions or initial_body_pos is None:
            print(f"[drone] Error: Missing data for {leg_id}")
            return
        
        # Spot leg parameters (approximate)
        L1, L2, L3 = 0.11, 0.26, 0.26
        y_offset = -0.25 if 'L' in leg_id else 0.25
        
        # Get initial body orientation
        if self.robopose.spot_node:
            initial_orientation = self.robopose.spot_node.getOrientation()
            initial_roll = math.atan2(initial_orientation[7], initial_orientation[8])
            initial_pitch = math.asin(-initial_orientation[6])
            initial_yaw = math.atan2(initial_orientation[3], initial_orientation[0])
        else:
            initial_roll = initial_pitch = initial_yaw = 0.0
        
        # Calculate initial foot position in body-local coordinates from first observation
        # This gives us the true initial position based on actual joint angles
        first_obs = observations[0]
        first_joint_angles = first_obs.get("joint_angles", [0.0, 0.0, 0.0])
        
        theta1_init = math.radians(first_joint_angles[0])
        theta2_init = math.radians(first_joint_angles[1])
        theta3_init = math.radians(first_joint_angles[2])
        
        x_init_local = L2 * math.cos(theta2_init) + L3 * math.cos(theta2_init + theta3_init)
        y_init_local = y_offset + L1 * math.sin(theta1_init)
        z_init_local = -(L2 * math.sin(theta2_init) + L3 * math.sin(theta2_init + theta3_init))
        
        # Also calculate world position for logging
        x_init_world = initial_body_pos[0] + x_init_local
        y_init_world = initial_body_pos[1] + y_init_local
        z_init_world = initial_body_pos[2] + z_init_local
        
        print(f"[drone] {leg_id}: Initial foot position (world): ({x_init_world:.3f}, {y_init_world:.3f}, {z_init_world:.3f})")
        print(f"[drone] {leg_id}: Initial body position: {initial_body_pos}")
        print(f"[drone] {leg_id}: Initial body orientation (RPY): ({math.degrees(initial_roll):.1f}°, {math.degrees(initial_pitch):.1f}°, {math.degrees(initial_yaw):.1f}°)")
        
        # Process each observation frame
        for i, (obs, body_pos) in enumerate(zip(observations, body_positions)):
            # Simplified: assume joint angles from observation (in real RoboPose, estimate from image)
            joint_angles = obs.get("joint_angles", [0.0, 0.0, 0.0])
            
            # Get current body orientation
            if self.robopose.spot_node:
                current_orientation = self.robopose.spot_node.getOrientation()
                current_roll = math.atan2(current_orientation[7], current_orientation[8])
                current_pitch = math.asin(-current_orientation[6])
                current_yaw = math.atan2(current_orientation[3], current_orientation[0])
            else:
                current_roll = current_pitch = current_yaw = 0.0
            
            # Debug: Print joint angles for first observation
            if i == 0:
                print(f"[drone] {leg_id}: First observation joint angles: {joint_angles}")
            
            # Calculate foot position in body-local coordinates using forward kinematics
            theta1 = math.radians(joint_angles[0])
            theta2 = math.radians(joint_angles[1])
            theta3 = math.radians(joint_angles[2])
            
            x_local = L2 * math.cos(theta2) + L3 * math.cos(theta2 + theta3)
            y_local = y_offset + L1 * math.sin(theta1)
            z_local = -(L2 * math.sin(theta2) + L3 * math.sin(theta2 + theta3))
            
            # Apply rotation compensation to account for body orientation changes (e.g., falling)
            # Calculate orientation change from initial
            delta_roll = current_roll - initial_roll
            delta_pitch = current_pitch - initial_pitch
            
            # Simplified rotation compensation (assumes small angles for efficiency)
            # For large angles, full rotation matrix would be more accurate
            # But for diagnosis purposes, this approximation is sufficient
            if abs(delta_roll) > 0.01 or abs(delta_pitch) > 0.01:
                # Apply inverse rotation to get true leg-local position
                # Rotate around Y-axis (pitch) then X-axis (roll)
                cos_p, sin_p = math.cos(-delta_pitch), math.sin(-delta_pitch)
                cos_r, sin_r = math.cos(-delta_roll), math.sin(-delta_roll)
                
                # Pitch rotation (around Y)
                x_temp = x_local * cos_p + z_local * sin_p
                z_temp = -x_local * sin_p + z_local * cos_p
                
                # Roll rotation (around X)
                y_compensated = y_local * cos_r - z_temp * sin_r
                z_compensated = y_local * sin_r + z_temp * cos_r
                x_compensated = x_temp
                
                x_local = x_compensated
                y_local = y_compensated
                z_local = z_compensated
            
            # Calculate displacement in body-local frame (relative to initial body position)
            # This removes the effect of body movement, focusing only on leg movement
            # by comparing current local position to initial local position
            x_disp = x_local - x_init_local
            y_disp = y_local - y_init_local
            z_disp = z_local - z_init_local
            
            end_position = [x_disp, y_disp, z_disp]
            base_orientation = [math.degrees(current_roll), math.degrees(current_pitch), math.degrees(current_yaw)]
            
            # Send frame to pipeline
            self.pipeline.record_robo_pose_frame(
                leg_id=leg_id,
                joint_angles=joint_angles,
                end_position=end_position,
                base_orientation=base_orientation,
                base_position=body_pos
            )
        
        final_disp = math.sqrt(x_disp**2 + y_disp**2 + z_disp**2)
        print(f"[drone] {leg_id}: Sent {len(observations)} frames to pipeline, final displacement: {final_disp:.6f}m")
        print(f"[drone] {leg_id}: Body moved: ({body_positions[-1][0] - initial_body_pos[0]:.3f}, "
              f"{body_positions[-1][1] - initial_body_pos[1]:.3f}, "
              f"{body_positions[-1][2] - initial_body_pos[2]:.3f})m")
        
        # Get self-diagnosis data from Spot
        self_diag = trial_info.get("self_diag_data")
        if self_diag:
            # Complete trial in pipeline with Spot's self-diagnosis data
            self.pipeline.complete_trial(
                leg_id=leg_id,
                theta_cmd=[],  # Drone doesn't record commanded angles
                theta_meas=[],  # Drone doesn't record measured angles (uses visual estimation)
                omega_meas=[],  # Drone doesn't record velocities
                tau_meas=[],  # Drone doesn't record torques
                tau_nominal=self_diag["tau_nominal"],
                safety_level=self_diag["safety_level"],
                end_time=trial_info["end_time"],
                spot_can_raw=self_diag.get("self_can_raw")  # Pass Spot's self_can_raw
            )
            print(f"[drone] {leg_id}: Trial completed in pipeline with self-diagnosis data")
            print(f"[drone] {leg_id}: tau_nominal={self_diag['tau_nominal']:.3f}, safety={self_diag['safety_level']}, spot_can={self_diag.get('self_can_raw', 'N/A'):.3f}")
        else:
            # Complete trial without self-diagnosis data (shouldn't happen)
            print(f"[drone] Warning: No self-diagnosis data for {leg_id}, completing with defaults")
            self.pipeline.complete_trial(
                leg_id=leg_id,
                theta_cmd=[],
                theta_meas=[],
                omega_meas=[],
                tau_meas=[],
                tau_nominal=0.0,
                safety_level="UNKNOWN",
                end_time=trial_info["end_time"]
            )
        
        # Track completion
        self.trials_completed[leg_id] = self.trials_completed.get(leg_id, 0) + 1
        print(f"[drone] {leg_id}: Completed {self.trials_completed[leg_id]}/{self.total_trials_per_leg} trials")
        
        # Check if all legs are done
        if all(count >= self.total_trials_per_leg for count in self.trials_completed.values()):
            print("\n[drone] All trials completed! Finalizing diagnosis...")
            self.finalize_diagnosis()
    
    def update_position(self):
        """Update drone position to track Spot."""
        if self.center_node is None:
            center_position = [0.0, 0.0, 0.0]
        else:
            center_position = list(self.center_node.getPosition())
        
        target_x = float(center_position[0] + self.offset_x)
        target_y = float(center_position[1] + self.offset_y)
        target_z = float(center_position[2] + self.offset_z)
        
        if self.translation_field is not None:
            try:
                self.translation_field.setSFVec3f([target_x, target_y, target_z])
            except RuntimeError as exc:
                print(f"[drone] translation error: {exc}")
        
        # Set velocity to zero (hovering)
        try:
            self.drone_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        except RuntimeError:
            pass
        
        # Point camera toward Spot
        dx = center_position[0] - target_x
        dy = center_position[1] - target_y
        heading = math.atan2(dy, dx)
        
        if self.rotation_field is not None:
            try:
                self.rotation_field.setSFRotation([0.0, 0.0, 1.0, float(heading)])
            except RuntimeError as exc:
                print(f"[drone] rotation error: {exc}")
    
    def run(self):
        """Main control loop."""
        print("[drone] Starting observation mode")
        print(f"[drone] Battery: {self.battery_current_wh:.2f}Wh / {self.battery_capacity_wh:.2f}Wh")
        
        while self.supervisor.step(self.time_step) != -1:
            current_time = self.supervisor.getTime()
            
            # Update battery consumption
            self.update_battery(current_time)
            
            # Check battery level
            if self.battery_current_wh <= 0:
                print("\n" + "="*80)
                print("[drone] BATTERY DEPLETED - Landing required")
                print("="*80)
                break
            
            # Process incoming triggers
            self.process_triggers()
            
            # Make RoboPose observation if needed
            self.make_observation()
            
            # Update position to track Spot
            self.update_position()
    
    def update_battery(self, current_time):
        """Update battery consumption based on flight state."""
        if self.battery_last_update == 0.0:
            self.battery_last_update = current_time
            return
        
        # Calculate time elapsed (in hours)
        time_elapsed_s = current_time - self.battery_last_update
        time_elapsed_h = time_elapsed_s / 3600.0
        
        # Determine power consumption based on state
        # For simplicity, assume hovering most of the time
        # In real scenario, you'd check velocity to determine if moving
        velocity = self.drone_node.getVelocity()
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2) if velocity else 0.0
        
        if speed > 0.1:  # Moving
            power_w = self.battery_move_power_w
        else:  # Hovering
            power_w = self.battery_hover_power_w
        
        # Calculate energy consumed
        energy_consumed_wh = power_w * time_elapsed_h
        self.battery_current_wh -= energy_consumed_wh
        
        # Clamp to zero
        if self.battery_current_wh < 0:
            self.battery_current_wh = 0.0
        
        # Log battery status periodically (every 10 seconds)
        if current_time - self.battery_last_log_time >= self.battery_log_interval:
            battery_percent = (self.battery_current_wh / self.battery_capacity_wh) * 100.0
            flight_time_min = current_time / 60.0
            estimated_remaining_min = (self.battery_current_wh / power_w) * 60.0 if power_w > 0 else 0.0
            
            print(f"[drone] Battery: {battery_percent:.1f}% ({self.battery_current_wh:.2f}Wh) | "
                  f"Flight time: {flight_time_min:.1f}min | "
                  f"Estimated remaining: {estimated_remaining_min:.1f}min")
            
            self.battery_last_log_time = current_time
        
        self.battery_last_update = current_time
    
    def finalize_diagnosis(self):
        """Finalize pipeline and output integrated diagnosis results."""
        print("\n" + "="*80)
        print("DRONE: Finalizing Integrated Diagnosis")
        print("="*80)
        
        # Finalize pipeline to compute all probabilities
        session_record = self.pipeline.finalize()
        
        # Import logger for file output
        from diagnostics_pipeline.logger import DiagnosticsLogger
        logger = DiagnosticsLogger()
        
        # Save session results to file
        # Use pipeline's session object (SessionState)
        logger.log_session(self.pipeline.session)
        
        print("\n" + "="*80)
        print("INTEGRATED DIAGNOSTIC RESULTS (from Drone)")
        print("="*80)
        
        # Display results for each leg
        for leg_id in diag_config.LEG_IDS:
            leg_state = session_record.legs.get(leg_id)
            if not leg_state:
                print(f"\n[{leg_id}] No data available")
                continue
            
            print(f"\n[{leg_id}] Diagnosis Summary:")
            
            # Spot self-diagnosis
            print(f"  Spot self-diagnosis:")
            print(f"    spot_can (Can-move): {leg_state.spot_can:.3f}")
            
            # Drone observation results
            print(f"  Drone observation:")
            print(f"    drone_can (Can-move): {leg_state.drone_can:.3f}")
            
            # Final diagnosis
            print(f"  Final diagnosis:")
            print(f"    Movement: {leg_state.movement_result}")
            print(f"    Cause: {leg_state.cause_final}")
            print(f"    p_can: {leg_state.p_can:.3f}")
            
            # Display LLM probability distribution
            print(f"  LLM probability distribution:")
            for cause, prob in leg_state.p_llm.items():
                bar = "█" * int(prob * 40)
                print(f"    {cause:12s}: {prob:.3f} {bar}")
        
        print("\n" + "="*80)
        print(f"Session ID: {session_record.session_id}")
        print(f"Fallen: {session_record.fallen} (probability: {session_record.fallen_probability:.1%})")
        print(f"Log saved to: controllers/spot_self_diagnosis/logs/")
        print("="*80)


def main():
    controller = DroneCircularController()
    controller.run()


if __name__ == "__main__":
    main()
