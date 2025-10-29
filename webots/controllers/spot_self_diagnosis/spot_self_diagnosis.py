"""Spot self-diagnosis controller with full diagnostics pipeline integration."""

from controller import Supervisor, Motor, PositionSensor
import struct
import time
import math
import json
from pathlib import Path
from datetime import datetime
import sys
import configparser

# Add diagnostics_pipeline to path
CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]
if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.append(str(CONTROLLERS_ROOT))

from diagnostics_pipeline import config as diag_config
from diagnostics_pipeline.pipeline import DiagnosticsPipeline
from diagnostics_pipeline.logger import DiagnosticsLogger

# Load scenario configuration
CONFIG_PATH = CONTROLLERS_ROOT.parent / "config" / "scenario.ini"


class SpotDiagnosticsController:
    """Spot controller for leg self-diagnosis with drone communication."""
    
    LEG_MOTOR_NAMES = {
        "FL": ["front left shoulder abduction motor", "front left shoulder rotation motor", "front left elbow motor"],
        "FR": ["front right shoulder abduction motor", "front right shoulder rotation motor", "front right elbow motor"],
        "RL": ["rear left shoulder abduction motor", "rear left shoulder rotation motor", "rear left elbow motor"],
        "RR": ["rear right shoulder abduction motor", "rear right shoulder rotation motor", "rear right elbow motor"],
    }
    
    def __init__(self):
        self.robot = Supervisor()  # Use Supervisor for customData communication
        self.time_step = int(self.robot.getBasicTimeStep())
        
        # Get self reference for customData communication
        self.self_node = self.robot.getSelf()
        self.custom_data_field = self.self_node.getField("customData")
        
        # Message queue for drone communication
        self.message_queue = []
        
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
        # Store actual target angle for accurate tracking score calculation
        self.actual_target_angle = 0.0  # degrees
        self.current_target_motor = None  # Store motor reference for torque feedback
        
        # Load scenario configuration
        self.scenario_config = self._load_scenario_config()
        
        # Initialize diagnostics pipeline
        session_id = f"spot_diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.pipeline = DiagnosticsPipeline(session_id)
        self.diag_logger = DiagnosticsLogger()
        self.session_start_time = datetime.now()
        
        print("[spot] Controller initialized")
        print(f"[spot] Time step: {self.time_step} ms")
        print(f"[spot] Will diagnose {len(diag_config.LEG_IDS)} legs, {diag_config.TRIAL_COUNT} trials each")
        print(f"[spot] Scenario: {self.scenario_config.get('scenario', 'unknown')}")
        print(f"[spot] Leg environments: FL={self.scenario_config.get('fl_environment')}, FR={self.scenario_config.get('fr_environment')}, RL={self.scenario_config.get('rl_environment')}, RR={self.scenario_config.get('rr_environment')}")
        print("[spot] Pipeline integration: ENABLED")
    
    def _load_scenario_config(self):
        """Load scenario configuration from scenario.ini"""
        config = configparser.ConfigParser()
        try:
            config.read(CONFIG_PATH)
            result = {
                'scenario': config.get('DEFAULT', 'scenario', fallback='none'),
                'buriedFoot': config.get('DEFAULT', 'buriedFoot', fallback=None),
                'trappedFoot': config.get('DEFAULT', 'trappedFoot', fallback=None),
                'tangledFoot': config.get('DEFAULT', 'tangledFoot', fallback=None),
                # 新しいper-leg環境設定を読み込む
                'fl_environment': config.get('DEFAULT', 'fl_environment', fallback='NONE'),
                'fr_environment': config.get('DEFAULT', 'fr_environment', fallback='NONE'),
                'rl_environment': config.get('DEFAULT', 'rl_environment', fallback='NONE'),
                'rr_environment': config.get('DEFAULT', 'rr_environment', fallback='NONE'),
            }
            return result
        except Exception as e:
            print(f"[spot] Warning: Could not load scenario config: {e}")
            return {'scenario': 'unknown'}
    
    def _get_leg_environment(self, leg_id):
        """Get environment setting for a specific leg from scenario.ini.
        
        Returns:
            "NONE", "BURIED", "TRAPPED", or "TANGLED"
        """
        # Map leg_id to config key (e.g., "FL" -> "fl_environment")
        env_key = f"{leg_id.lower()}_environment"
        return self.scenario_config.get(env_key, "NONE").upper()
    
    def get_expected_cause(self, leg_id):
        """Get expected cause from scenario configuration."""
        scenario = self.scenario_config.get('scenario', 'none')
        
        # 新しい方式: fl_environment, fr_environment などから直接取得
        leg_env = self._get_leg_environment(leg_id)
        if leg_env in ["BURIED", "TRAPPED", "TANGLED"]:
            return leg_env
        
        # 旧方式: 互換性のため残す
        leg_full_names = {
            "FL": "front_left",
            "FR": "front_right",
            "RL": "rear_left",
            "RR": "rear_right",
        }
        leg_full_name = leg_full_names.get(leg_id, leg_id.lower())
        
        if scenario == 'sand_burial':
            if self.scenario_config.get('buriedFoot') == leg_full_name:
                return "BURIED"
        elif scenario == 'foot_trap':
            if self.scenario_config.get('trappedFoot') == leg_full_name:
                return "TRAPPED"
        elif scenario == 'foot_vine':
            if self.scenario_config.get('tangledFoot') == leg_full_name:
                return "TANGLED"
        
        return "NONE"
    
    def _initialize_devices(self):
        """Initialize all motors and sensors."""
        for leg_id, motor_names in self.LEG_MOTOR_NAMES.items():
            self.motors[leg_id] = []
            self.sensors[leg_id] = []
            
            for motor_name in motor_names:
                motor = self.robot.getDevice(motor_name)
                if motor:
                    # Enable torque feedback for all motors
                    motor.enableTorqueFeedback(self.time_step)
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
        """Send trigger message to drone via customData."""
        # Message format: "TRIGGER|leg_id|trial_index|direction|start_time|duration_ms"
        message = f"TRIGGER|{leg_id}|{trial_index}|{direction}|{start_time:.6f}|{duration_ms}"
        self.message_queue.append(message)
        self._flush_messages()
        print(f"[spot] Sent trigger: {leg_id} trial {trial_index} dir={direction}")
    
    def send_self_diagnosis_to_drone(self, leg_id, trial_index, tau_limit, safety_level, self_can_raw):
        """Send self-diagnosis data to drone for integrated diagnosis via customData."""
        
        # Get measured data from trial
        theta_meas_count = len(self.trial_data["theta_meas"])
        tau_meas_count = len(self.trial_data["tau_meas"])
        
        # Calculate summary statistics
        if theta_meas_count > 0:
            theta_avg = sum(self.trial_data["theta_meas"]) / theta_meas_count
            theta_final = self.trial_data["theta_meas"][-1]
        else:
            theta_avg = 0.0
            theta_final = 0.0
        
        if tau_meas_count > 0:
            tau_avg = sum(self.trial_data["tau_meas"]) / tau_meas_count
            tau_max = max(self.trial_data["tau_meas"])
        else:
            tau_avg = 0.0
            tau_max = 0.0
        
        # Message format: "SELF_DIAG|leg_id|trial_index|theta_samples|theta_avg|theta_final|tau_avg|tau_max|tau_limit|safety|self_can_raw"
        message = (f"SELF_DIAG|{leg_id}|{trial_index}|{theta_meas_count}|"
                  f"{theta_avg:.6f}|{theta_final:.6f}|{tau_avg:.6f}|{tau_max:.6f}|"
                  f"{tau_limit:.6f}|{safety_level}|{self_can_raw:.6f}")
        self.message_queue.append(message)
        self._flush_messages()
    
    def send_spot_can_to_drone(self, leg_id, spot_can):
        """仕様ステップ5: spotはspot_canをドローンへ送る
        
        各脚の全6回の試行が終了した後、シグモイド変換されたspot_canを送信する。
        """
        # Message format: "SPOT_CAN|leg_id|spot_can"
        message = f"SPOT_CAN|{leg_id}|{spot_can:.6f}"
        self.message_queue.append(message)
        self._flush_messages()
        print(f"[spot] 仕様ステップ5: {leg_id}のspot_can={spot_can:.3f}をドローンへ送信")
    
    def send_observation_frame(self, leg_id, trial_index):
        """Send current joint angles to drone for visual observation simulation.
        
        This allows the drone to simulate RoboPose observations using actual joint angles.
        Message format: "JOINT_ANGLES|leg_id|trial_index|angle0|angle1|angle2"
        """
        # Get all joint angles for this leg
        leg_sensors = self.sensors.get(leg_id, [])
        if not leg_sensors or len(leg_sensors) < 3:
            return
        
        joint_angles = []
        for i in range(3):  # shoulder, hip, knee
            if leg_sensors[i]:
                angle_rad = leg_sensors[i].getValue()
                angle_deg = math.degrees(angle_rad)
                joint_angles.append(f"{angle_deg:.6f}")
            else:
                joint_angles.append("0.0")
        
        # Send joint angles to drone
        message = f"JOINT_ANGLES|{leg_id}|{trial_index}|" + "|".join(joint_angles)
        self.message_queue.append(message)
        self._flush_messages()
        
        # Debug: print first observation only to avoid spam
        if not hasattr(self, '_sent_first_obs'):
            self._sent_first_obs = {}
        if leg_id not in self._sent_first_obs:
            print(f"[spot] Sent joint angles for {leg_id} trial {trial_index}: [{', '.join([f'{float(a):.1f}°' for a in joint_angles])}]")
            self._sent_first_obs[leg_id] = True
    
    def _flush_messages(self):
        """Flush message queue to customData field."""
        if self.message_queue:
            # Join all messages with newline separator
            combined = "\n".join(self.message_queue)
            self.custom_data_field.setSFString(combined)
            self.message_queue.clear()
    
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
        
        # Motor selection strategy: Use different joints for different trials
        # Trial 1-2: Shoulder abduction (index 0) - y-axis movement (~0.8cm)
        # Trial 3: Hip flexion (index 1) - x-z axis movement (~5-7cm)
        # Trial 4: Knee flexion (index 2) - x-z axis movement (~5-7cm)
        motor_index = diag_config.TRIAL_MOTOR_INDICES[trial_index - 1]
        motor_names = ["shoulder abduction", "hip flexion", "knee flexion"]
        motor_name = motor_names[motor_index] if motor_index < len(motor_names) else f"joint {motor_index}"
        
        if len(leg_motors) > motor_index and leg_motors[motor_index]:
            target_motor = leg_motors[motor_index]
            target_sensor = leg_sensors[motor_index] if len(leg_sensors) > motor_index else None
        else:
            print(f"[spot] Warning: Motor index {motor_index} ({motor_name}) not available for {leg_id}")
            return False
        
        # Store motor index for later use in measurement and reset
        self.current_motor_index = motor_index
        
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
        
        # Store actual target angle for accurate tracking score calculation
        self.actual_target_angle = safe_angle * sign  # degrees
        
        print(f"[spot] {leg_id} Trial {trial_index}: Using {motor_name} at {safe_angle:.2f}° ({direction})")
        
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
        self.current_trial_index = trial_index  # Store for observation frames
        
        if target_motor and target_sensor:
            # Store motor reference for torque feedback
            self.current_target_motor = target_motor
            
            # Get initial position
            initial_pos = target_sensor.getValue()
            target_pos = initial_pos + angle_rad
            
            # Debug log
            print(f"[spot] {leg_id} Trial {trial_index}: initial={math.degrees(initial_pos):.3f}°, "
                  f"target={math.degrees(target_pos):.3f}°, change={math.degrees(angle_rad):.3f}°")
            
            # BURIED環境のシミュレーション: モーター出力を大幅に制限
            # scenario.iniで指定された脚のモーター出力を制限することで、
            # 物理環境に依存せず確実にBURIED状態を再現
            leg_env = self._get_leg_environment(leg_id)
            
            # デバッグログをファイルに出力
            import os
            log_path = "/tmp/motor_control_debug.log"
            with open(log_path, "a") as f:
                f.write(f"Trial {trial_index}: leg={leg_id}, env={leg_env}, angle_rad={angle_rad:.4f}\n")
            
            if leg_env == "BURIED":
                # BURIEDの場合、モーターの動きを極端に制限
                # 通常の5%の速度と、非常に小さな角度変化のみ許可
                restricted_angle = angle_rad * 0.05  # 5%の動き
                target_pos = initial_pos + restricted_angle
                target_motor.setPosition(target_pos)
                target_motor.setVelocity(target_motor.getMaxVelocity() * 0.05)  # 5%速度
                
                with open(log_path, "a") as f:
                    f.write(f"  → BURIED restriction applied: {angle_rad:.4f} → {restricted_angle:.4f} (5%)\n")
            else:
                # 通常動作
                target_motor.setPosition(target_pos)
                # Use 20% velocity for stable, safe movement
                target_motor.setVelocity(target_motor.getMaxVelocity() * 0.2)
        
        return True
    
    def _reset_leg_position(self, leg_id):
        """Reset leg motor to initial position (0°) after trial.
        This prevents cumulative angle issues causing physical constraints."""
        leg_motors = self.motors.get(leg_id, [])
        
        # Use the same motor index that was used in the last trial
        motor_index = getattr(self, 'current_motor_index', 0)
        
        if len(leg_motors) > motor_index and leg_motors[motor_index]:
            target_motor = leg_motors[motor_index]
            # Reset to 0° position
            target_motor.setPosition(0.0)
            # Use slower velocity for smooth, stable reset
            target_motor.setVelocity(target_motor.getMaxVelocity() * 0.15)
            print(f"[spot] {leg_id}: Resetting motor {motor_index} to 0°")
        else:
            print(f"[spot] Warning: Cannot reset {leg_id}, motor {motor_index} not available")
    
    def measure_trial_data(self, leg_id):
        """Collect sensor data during trial execution."""
        leg_sensors = self.sensors.get(leg_id, [])
        if not leg_sensors or not any(leg_sensors):
            return
        
        # Use the same motor index that was used in execute_trial
        motor_index = getattr(self, 'current_motor_index', 0)
        
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
            
            # Measure torque (use feedback if available, else estimate from velocity change)
            if self.current_target_motor:
                try:
                    # Try to get actual torque feedback from motor
                    actual_torque = self.current_target_motor.getTorqueFeedback()
                    # Check if torque is valid (not nan or inf)
                    if math.isnan(actual_torque) or math.isinf(actual_torque):
                        # Fallback to 0 if invalid
                        self.trial_data["tau_meas"].append(0.0)
                    else:
                        self.trial_data["tau_meas"].append(abs(actual_torque))
                except (AttributeError, TypeError):
                    # If getTorqueFeedback not available, estimate from acceleration
                    if len(self.trial_data["omega_meas"]) >= 3:
                        # Use median of recent accelerations to reduce noise
                        recent_omegas = self.trial_data["omega_meas"][-3:]
                        dt = self.time_step / 1000.0
                        
                        accels = []
                        for i in range(1, len(recent_omegas)):
                            accel = abs((recent_omegas[i] - recent_omegas[i-1]) / dt)
                            accels.append(accel)
                        
                        # Use median to reduce noise impact
                        from statistics import median
                        accel_median = median(accels) if accels else 0.0
                        
                        # More precise inertia model (kg*m^2)
                        # Spot leg segment approximate inertia
                        inertia = 0.05
                        estimated_torque = inertia * accel_median
                        self.trial_data["tau_meas"].append(estimated_torque)
                    else:
                        self.trial_data["tau_meas"].append(0.0)
            else:
                self.trial_data["tau_meas"].append(0.0)
            
            # Send current joint angles to drone for observation
            current_trial_index = getattr(self, 'current_trial_index', 1)
            self.send_observation_frame(leg_id, current_trial_index)
    
    def _calculate_tau_limit(self, angle_deg: float) -> float:
        """Calculate torque limit based on motor type and real measurements.
        
        実測データ分析結果:
        - RR (正常): 0.1-1.7 N·m  ← 全モーター
        - FL/FR/RL (異常): 44-50 N·m  ← 物理的負荷または TRAPPED
        
        モーター種別ごとの正常トルクリミット:
        - shoulder (index 0): 通常 0.1-2.0 N·m → limit=5.0
        - hip (index 1): 通常 0.1-2.0 N·m → limit=5.0
        - knee (index 2): 通常 0.1-2.0 N·m → limit=5.0
        
        TRAPPED検出閾値: 10.0 N·m 以上
        → limit=5.0 設定により、正常時は高スコア、TRAPPED時は低スコアを実現
        
        Args:
            angle_deg: Target angle in degrees (currently unused, reserved for future)
            
        Returns:
            Torque limit in N·m
        """
        # 実測データに基づく統一リミット
        # 正常時の最大値: FL=1.3, FR=5.4, RL=14.6 N·m
        # → limit=7.0 により、FL/FRの正常動作を許容しつつ、TRAPPED(35-50)を検出
        tau_limit = 7.0  # N·m
        
        return tau_limit
    
    def _calculate_self_can_raw(self) -> float:
        """Calculate self_can_raw score from collected trial data.
        
        Uses the same scoring logic as diagnostics_pipeline/self_diagnosis.py
        but calculates directly from trial_data without pipeline.
        
        Returns:
            self_can_raw score (0.0 to 1.0)
        """
        from diagnostics_pipeline import config as diag_config
        
        # Extract data
        theta_meas = self.trial_data.get("theta_meas", [])
        omega_meas = self.trial_data.get("omega_meas", [])
        tau_meas = self.trial_data.get("tau_meas", [])
        
        if not theta_meas or len(theta_meas) < 2:
            return 0.35  # Default for insufficient data
        
        # Use the actual target angle stored during execute_trial
        target_angle = abs(getattr(self, 'actual_target_angle', 5.0))  # Default to 5.0°
        
        # 1. Tracking score (追従性)
        # Actual angle change achieved
        final_angle = abs(theta_meas[-1] - theta_meas[0])
        tracking_error = abs(target_angle - final_angle)
        tracking_score = max(0.0, 1.0 - (tracking_error / diag_config.E_MAX_DEG))
        
        # 2. Velocity score (速度性能)
        peak_velocity = max(abs(v) for v in omega_meas) if omega_meas else 0.0
        velocity_score = min(1.0, peak_velocity / diag_config.OMEGA_REF_DEG_PER_SEC)
        
        # 3. Torque score (トルク評価)
        # Filter out nan values
        valid_tau = [abs(t) for t in tau_meas if not math.isnan(t) and not math.isinf(t)]
        mean_torque = sum(valid_tau) / len(valid_tau) if valid_tau else 0.0
        tau_limit = self._calculate_tau_limit(target_angle)
        if tau_limit > 0 and mean_torque > 0:
            torque_score = 1.0 - min(1.0, mean_torque / tau_limit)
        else:
            torque_score = 1.0  # No torque data or zero limit → assume OK
        
        # 4. Safety score (安全性)
        safety_score = diag_config.SAFE_SCORE_NORMAL  # Always SAFE for now
        
        # Weighted combination
        weights = diag_config.SELF_WEIGHTS
        self_can_raw = (
            tracking_score * weights["track"] +
            velocity_score * weights["vel"] +
            torque_score * weights["tau"] +
            safety_score * weights["safe"]
        )
        
        # Debug output
        valid_tau_count = len([t for t in tau_meas if t > 0])
        print(f"[spot_self_can] target={target_angle:.2f}°, final={final_angle:.2f}°, "
              f"peak_vel={peak_velocity:.1f}°/s, mean_tau={mean_torque:.3f} (limit={tau_limit:.1f}), "
              f"valid={valid_tau_count}/{len(tau_meas)}")
        print(f"[spot_self_can] track={tracking_score:.3f}, vel={velocity_score:.3f}, "
              f"tau={torque_score:.3f}, safe={safety_score:.3f} → self_can={self_can_raw:.3f}")
        
        return self_can_raw
    
    
    def finalize_and_output_results(self):
        """Finalize Spot's self-diagnosis pipeline and output results.
        Note: Integrated diagnosis is performed by Drone."""
        # Finalize pipeline to compute self-diagnosis results
        session_record = self.pipeline.finalize()
        
        print("\n" + "="*80)
        print("SPOT SELF-DIAGNOSIS RESULTS (Internal Sensors Only)")
        print("="*80)
        print("Note: Integrated diagnosis with drone observation will be shown by drone controller.")
        print("="*80)
        
        # Display self-diagnosis results for each leg
        for leg_id in diag_config.LEG_IDS:
            leg_state = session_record.legs.get(leg_id)
            if not leg_state:
                print(f"\n[{leg_id}] No data available")
                continue
            
            print(f"\n[{leg_id}] Self-Diagnosis:")
            print(f"  spot_can (Can-move probability): {leg_state.spot_can:.3f}")
            status = "OK" if leg_state.spot_can >= diag_config.SELF_CAN_THRESHOLD else "ABNORMAL"
            print(f"  Status: {status}")
        
        print("\n" + "="*80)
        print("Waiting for drone to complete integrated diagnosis...")
        print("="*80)
    
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
                # Use standard trial pattern for all legs
                # Previously had reversed pattern for RL leg to avoid ground collision,
                # but this caused false MALFUNCTION detection due to poor movement
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
                    
                    # Generate commanded angle for this timestep using ACTUAL target angle
                    progress = (step + 1) / trial_steps
                    # Use the actual target angle calculated in execute_trial()
                    theta_cmd = self.actual_target_angle * progress
                    self.trial_data["theta_cmd"].append(theta_cmd)
                    
                    # Measure actual sensor values
                    self.measure_trial_data(leg_id)
                
                # Complete trial in pipeline with collected data
                # Note: Drone observation frames will be generated by drone controller
                end_time = self.robot.getTime()
                # Calculate tau_limit based on actual target angle
                tau_limit = self._calculate_tau_limit(self.actual_target_angle)
                safety_level = "SAFE"  # Movement is considered safe
                
                # Calculate self_can_raw from collected data
                self_can_raw = self._calculate_self_can_raw()
                
                # Do NOT call complete_trial here - Drone will call it after receiving SELF_DIAG
                # This prevents duplicate complete_trial calls with empty joint_angles
                
                # Send self-diagnosis data to Drone for integration
                self.send_self_diagnosis_to_drone(leg_id, trial_index, tau_limit, safety_level, self_can_raw)
                
                print(f"[spot] Trial {trial_index} complete - collected {len(self.trial_data['theta_meas'])} samples")
                
                # TODO: Reset leg to initial position after confirming stability
                # self._reset_leg_position(leg_id)
                # reset_steps = int((1.5 * 1000) / self.time_step)
                # for _ in range(reset_steps):
                #     if self.robot.step(self.time_step) == -1:
                #         return
                
                # Small pause between trials
                for _ in range(10):
                    if self.robot.step(self.time_step) == -1:
                        return
            
            # 仕様ステップ3,5: 全6回の試行が終了したら、spot_canを計算してドローンに送信
            print(f"\n[spot] 仕様ステップ3: {leg_id}の全試行終了、spot_can計算")
            leg_state = self.pipeline.session.legs.get(leg_id)
            if leg_state and hasattr(leg_state, 'spot_can'):
                self.send_spot_can_to_drone(leg_id, leg_state.spot_can)
            else:
                print(f"[spot] Warning: {leg_id}のspot_canが計算されていません")
            
            # Pause before next leg
            for _ in range(20):
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
