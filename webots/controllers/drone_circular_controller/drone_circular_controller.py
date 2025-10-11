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


class DroneController:
    """Drone controller with RoboPose and communication."""
    
    def __init__(self):
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())
        
        # Parse command line arguments
        arguments = sys.argv[1:]
        self.offset_x, self.offset_y, self.offset_z, self.center_def = self.parse_arguments(arguments)
        
        # Initialize RoboPose simulator
        self.robopose = RoboPoseSimulator(self.supervisor, self.center_def)
        
        # Initialize receiver for triggers from Spot
        self.receiver = self.supervisor.getDevice("receiver")
        if self.receiver is None:
            print("[drone] Warning: receiver not found")
        else:
            self.receiver.enable(self.time_step)
            self.receiver.setChannel(1)
        
        # Initialize emitter for sending observations
        self.emitter = self.supervisor.getDevice("emitter")
        if self.emitter is None:
            print("[drone] Warning: emitter not found")
        else:
            self.emitter.setChannel(2)
        
        # Get drone node and fields for positioning
        self.drone_node = self.supervisor.getSelf()
        self.translation_field = self.drone_node.getField("translation")
        self.rotation_field = self.drone_node.getField("rotation")
        
        # Get center (Spot) node
        self.center_node = self.supervisor.getFromDef(self.center_def)
        
        # State tracking
        self.active_trials = {}  # leg_id -> trial_info
        self.observation_mode = "IDLE"  # IDLE or ACTIVE
        self.fps_current = diag_config.ROBOPOSE_FPS_IDLE
        self.last_observation_time = 0
        self.observation_interval = 1.0 / self.fps_current
        
        print("[drone] Controller initialized")
        print(f"[drone] Offset: ({self.offset_x:.2f}, {self.offset_y:.2f}, {self.offset_z:.2f})")
        print(f"[drone] Center: {self.center_def}")
        print(f"[drone] FPS: idle={diag_config.ROBOPOSE_FPS_IDLE}, trigger={diag_config.ROBOPOSE_FPS_TRIGGER}")
    
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
        """Process incoming trigger messages from Spot."""
        if self.receiver is None:
            return
        
        while self.receiver.getQueueLength() > 0:
            message = self.receiver.getString()
            self.receiver.nextPacket()
            
            try:
                parts = message.split("|")
                if parts[0] == "TRIGGER":
                    leg_id = parts[1]
                    trial_index = int(parts[2])
                    direction = parts[3]
                    start_time = float(parts[4])
                    duration_ms = int(parts[5])
                    
                    end_time = start_time + (duration_ms / 1000.0)
                    
                    self.active_trials[leg_id] = {
                        "trial_index": trial_index,
                        "direction": direction,
                        "start_time": start_time,
                        "end_time": end_time,
                        "observations": []
                    }
                    
                    # Switch to high-frequency observation mode
                    self.observation_mode = "ACTIVE"
                    self.fps_current = diag_config.ROBOPOSE_FPS_TRIGGER
                    self.observation_interval = 1.0 / self.fps_current
                    
                    print(f"[drone] Trigger received: {leg_id} trial {trial_index} dir={direction}")
            
            except Exception as e:
                print(f"[drone] Error processing trigger: {e}")
    
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
                trial_info["observations"].append(observation)
            elif current_time > trial_info["end_time"]:
                # Trial complete - send data
                self.send_trial_data(leg_id, trial_info)
                del self.active_trials[leg_id]
        
        # Switch back to idle mode if no active trials
        if len(self.active_trials) == 0 and self.observation_mode == "ACTIVE":
            self.observation_mode = "IDLE"
            self.fps_current = diag_config.ROBOPOSE_FPS_IDLE
            self.observation_interval = 1.0 / self.fps_current
            print("[drone] Switched to idle observation mode")
    
    def send_trial_data(self, leg_id, trial_info):
        """Send collected observation data."""
        if self.emitter is None:
            return
        
        obs_count = len(trial_info["observations"])
        print(f"[drone] Sending {obs_count} observations for {leg_id} trial {trial_info['trial_index']}")
        
        # In a real system, this would send the actual observations
        # For now, we'll send a summary message
        message = f"DATA|{leg_id}|{trial_info['trial_index']}|{obs_count}"
        self.emitter.send(message.encode('utf-8'))
    
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
        
        while self.supervisor.step(self.time_step) != -1:
            # Process incoming triggers
            self.process_triggers()
            
            # Make RoboPose observation if needed
            self.make_observation()
            
            # Update position to track Spot
            self.update_position()


def main():
    controller = DroneController()
    controller.run()


if __name__ == "__main__":
    main()
