#!/usr/bin/env python3
"""Spot 自己診断コントローラ（webots_new 簡潔版）

役割:
- 各脚について 6 試行の小さな関節動作を実行
- Drone に customData で以下を送る
  - TRIGGER: 試行開始
  - JOINT_ANGLES: 観測フレーム（関節角）
  - SELF_DIAG: Spot の自己診断（試行の要約 + self_can_raw）

メッセージ形式（既存互換）:
- TRIGGER|leg_id|trial_index|direction|start_time|duration_ms
- JOINT_ANGLES|leg_id|trial_index|a0|a1|a2
- SELF_DIAG|leg_id|trial_index|theta_samples|theta_avg|theta_final|tau_avg|tau_max|tau_nominal|safety|self_can_raw
"""

from __future__ import annotations

import configparser
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from controller import Supervisor


CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CONTROLLERS_ROOT.parent / "config" / "scenario.ini"

if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.append(str(CONTROLLERS_ROOT))

from diagnostics_pipeline import config as diag_config  # noqa: E402


LEG_MOTOR_NAMES: Dict[str, List[str]] = {
    "FL": [
        "front left shoulder abduction motor",
        "front left shoulder rotation motor",
        "front left elbow motor",
    ],
    "FR": [
        "front right shoulder abduction motor",
        "front right shoulder rotation motor",
        "front right elbow motor",
    ],
    "RL": [
        "rear left shoulder abduction motor",
        "rear left shoulder rotation motor",
        "rear left elbow motor",
    ],
    "RR": [
        "rear right shoulder abduction motor",
        "rear right shoulder rotation motor",
        "rear right elbow motor",
    ],
}


def _deg(rad: float) -> float:
    return math.degrees(rad)


def _rad(deg: float) -> float:
    return math.radians(deg)


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


class SpotSelfDiagnosis:
    def __init__(self) -> None:
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.self_node = self.robot.getSelf()
        self.custom_data_field = self.self_node.getField("customData")

        self.motors: Dict[str, List] = {}
        self.sensors: Dict[str, List] = {}
        self._init_devices()

        self.scenario = self._load_scenario()

        self._queue: List[str] = []

        # 起動直後はセンサ値が NaN のことがあるので、少し step して安定させる
        for _ in range(2):
            if self.robot.step(self.time_step) == -1:
                break

        session_id = f"spot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"[spot_new] init session={session_id} timestep={self.time_step}ms")
        print(
            f"[spot_new] env FL={self._leg_env('FL')} FR={self._leg_env('FR')} RL={self._leg_env('RL')} RR={self._leg_env('RR')}"
        )

    # ---- config / devices ----

    def _load_scenario(self) -> Dict[str, str]:
        cfg = configparser.ConfigParser()
        try:
            cfg.read(CONFIG_PATH)
            return {
                "fl_environment": cfg.get("DEFAULT", "fl_environment", fallback="NONE"),
                "fr_environment": cfg.get("DEFAULT", "fr_environment", fallback="NONE"),
                "rl_environment": cfg.get("DEFAULT", "rl_environment", fallback="NONE"),
                "rr_environment": cfg.get("DEFAULT", "rr_environment", fallback="NONE"),
            }
        except Exception as exc:
            print(f"[spot_new] warn: scenario.ini read failed: {exc}")
            return {}

    def _leg_env(self, leg_id: str) -> str:
        return self.scenario.get(f"{leg_id.lower()}_environment", "NONE").strip().upper()

    def _init_devices(self) -> None:
        for leg_id, motor_names in LEG_MOTOR_NAMES.items():
            self.motors[leg_id] = []
            self.sensors[leg_id] = []
            for motor_name in motor_names:
                motor = self.robot.getDevice(motor_name)
                sensor = self.robot.getDevice(motor_name.replace("motor", "sensor"))

                if motor:
                    try:
                        motor.enableTorqueFeedback(self.time_step)
                    except Exception:
                        pass
                if sensor:
                    try:
                        sensor.enable(self.time_step)
                    except Exception:
                        pass

                self.motors[leg_id].append(motor)
                self.sensors[leg_id].append(sensor)

    # ---- messaging ----

    def _send(self, message: str) -> None:
        self._queue.append(message)
        combined = "\n".join(self._queue)
        self.custom_data_field.setSFString(combined)
        self._queue.clear()

    def send_trigger(self, leg_id: str, trial_index: int, direction: str, start_time: float, duration_ms: int) -> None:
        self._send(f"TRIGGER|{leg_id}|{trial_index}|{direction}|{start_time:.6f}|{duration_ms}")

    def send_joint_angles(self, leg_id: str, trial_index: int) -> None:
        sensors = self.sensors.get(leg_id, [])
        if len(sensors) < 3:
            return

        angles = []
        for i in range(3):
            s = sensors[i]
            if s:
                v = s.getValue()
                if not _is_finite(v):
                    v = 0.0
                angles.append(f"{_deg(v):.6f}")
            else:
                angles.append("0.0")
        self._send(f"JOINT_ANGLES|{leg_id}|{trial_index}|" + "|".join(angles))

    def _read_sensor(self, sensor, retries: int = 3) -> Optional[float]:
        """センサ値を読み、NaN/inf の場合は少し待って再試行する。"""
        for _ in range(max(1, retries)):
            v = sensor.getValue()
            if _is_finite(v):
                return float(v)
            if self.robot.step(self.time_step) == -1:
                return None
        return None

    def send_self_diag(
        self,
        leg_id: str,
        trial_index: int,
        theta_meas: List[float],
        tau_meas: List[float],
        tau_nominal: float,
        safety_level: str,
        self_can_raw: float,
    ) -> None:
        theta_n = len(theta_meas)
        theta_avg = (sum(theta_meas) / theta_n) if theta_n else 0.0
        theta_final = theta_meas[-1] if theta_n else 0.0

        tau_n = len(tau_meas)
        tau_avg = (sum(tau_meas) / tau_n) if tau_n else 0.0
        tau_max = max(tau_meas) if tau_n else 0.0

        msg = (
            f"SELF_DIAG|{leg_id}|{trial_index}|{theta_n}|"
            f"{theta_avg:.6f}|{theta_final:.6f}|{tau_avg:.6f}|{tau_max:.6f}|"
            f"{tau_nominal:.6f}|{safety_level}|{self_can_raw:.6f}"
        )
        self._send(msg)

    # ---- trial execution ----

    def _safe_angle(self, motor, sensor) -> Tuple[float, float]:
        if not motor or not sensor:
            return 0.0, 0.0

        current = self._read_sensor(sensor)
        if current is None:
            return 0.0, 0.0

        current_deg = _deg(current)

        min_pos = motor.getMinPosition()
        max_pos = motor.getMaxPosition()
        if min_pos == float("-inf"):
            min_pos = current - _rad(30)
        if max_pos == float("inf"):
            max_pos = current + _rad(30)

        min_deg = _deg(min_pos)
        max_deg = _deg(max_pos)
        margin = 5.0

        safe_pos = max(0.0, min(30.0, (max_deg - current_deg - margin)))
        safe_neg = max(0.0, min(30.0, (current_deg - min_deg - margin)))
        return safe_pos, safe_neg

    def _calculate_tau_limit(self) -> float:
        return 7.0

    def _score_self_can_raw(self, theta_cmd: List[float], theta_meas: List[float], omega_meas: List[float], tau_meas: List[float], tau_limit: float) -> float:
        # tracking
        if theta_cmd and theta_meas and len(theta_cmd) == len(theta_meas):
            errors = [abs(c - m) for c, m in zip(theta_cmd, theta_meas)]
            track = max(0.0, min(1.0, 1.0 - (sum(errors) / len(errors)) / 3.0))
        else:
            track = 0.0

        # velocity
        if omega_meas:
            peak = max(abs(v) for v in omega_meas)
            vel = max(0.0, min(1.0, peak / 27.0))
        else:
            vel = 0.0

        # torque
        if tau_meas and tau_limit > 1e-6:
            mean_abs = sum(abs(x) for x in tau_meas) / len(tau_meas)
            tau = max(0.0, min(1.0, 1.0 - (mean_abs / tau_limit)))
        else:
            tau = 1.0

        safe = 1.0

        w = diag_config.SELF_WEIGHTS
        raw = w["track"] * track + w["vel"] * vel + w["tau"] * tau + w["safe"] * safe
        return max(0.0, min(1.0, raw))

    def run(self) -> None:
        for leg_id in diag_config.LEG_IDS:
            print(f"[spot_new] ===== {leg_id} start =====")
            for trial_index in range(1, diag_config.TRIAL_COUNT + 1):
                direction = diag_config.TRIAL_PATTERN[trial_index - 1]
                motor_index = diag_config.TRIAL_MOTOR_INDICES[trial_index - 1]

                motors = self.motors.get(leg_id, [])
                sensors = self.sensors.get(leg_id, [])
                if len(motors) <= motor_index or not motors[motor_index]:
                    print(f"[spot_new] warn: motor missing {leg_id} idx={motor_index}")
                    continue

                motor = motors[motor_index]
                sensor = sensors[motor_index] if len(sensors) > motor_index else None

                safe_pos, safe_neg = self._safe_angle(motor, sensor)
                requested = diag_config.TRIAL_ANGLE_DEG

                if direction == "+":
                    if safe_pos < 0.5:
                        continue
                    angle = min(requested, safe_pos)
                    sign = 1.0
                else:
                    if safe_neg < 0.5:
                        continue
                    angle = min(requested, safe_neg)
                    sign = -1.0

                env = self._leg_env(leg_id)
                angle_rad = _rad(angle * sign)

                joint_names = ["shoulder", "hip", "knee"]
                joint_name = joint_names[motor_index] if 0 <= motor_index < len(joint_names) else str(motor_index)
                print(
                    f"[spot_new] {leg_id} trial {trial_index}/{diag_config.TRIAL_COUNT} dir={direction} joint={joint_name} env={env}"
                )

                # BURIED は動き・速度を極端に制限（旧実装の再現）
                if env == "BURIED":
                    angle_rad *= 0.05
                    vel_scale = 0.05
                # TANGLED は接触が激しくなりやすいので、NaN/転倒を避けるため適度にマイルドにする
                elif env == "TANGLED":
                    angle_rad *= 0.85
                    vel_scale = 0.15
                else:
                    vel_scale = 0.2

                start_time = self.robot.getTime()
                duration_ms = int(diag_config.TRIAL_DURATION_S * 1000)
                self.send_trigger(leg_id, trial_index, direction, start_time, duration_ms)

                theta_cmd: List[float] = []
                theta_meas: List[float] = []
                omega_meas: List[float] = []
                tau_meas: List[float] = []

                # command
                if sensor:
                    initial = self._read_sensor(sensor)
                    if initial is None:
                        # センサが不安定なら、この試行はスキップ
                        continue
                    target = initial + angle_rad
                else:
                    initial = 0.0
                    target = angle_rad

                if not _is_finite(target):
                    # 念のため NaN/inf を弾く
                    continue

                try:
                    motor.setPosition(target)
                    motor.setVelocity(motor.getMaxVelocity() * vel_scale)
                except Exception:
                    pass

                # run for duration
                steps = max(1, int((diag_config.TRIAL_DURATION_S * 1000) / max(1, self.time_step)))
                prev_theta = None

                for _ in range(steps):
                    if self.robot.step(self.time_step) == -1:
                        return

                    # observation frame to drone
                    self.send_joint_angles(leg_id, trial_index)

                    if not sensor:
                        continue

                    v = sensor.getValue()
                    if not _is_finite(v):
                        # NaN のときは 0 扱い（ログ/スコアが壊れないように）
                        v = 0.0
                    theta = _deg(v)
                    theta_meas.append(theta)

                    # commanded angle (deg) for tracking: target position (deg)
                    theta_cmd.append(_deg(target))

                    if prev_theta is None:
                        omega_meas.append(0.0)
                    else:
                        dt = self.time_step / 1000.0
                        omega_meas.append((theta - prev_theta) / dt)
                    prev_theta = theta

                    # torque
                    try:
                        t = motor.getTorqueFeedback()
                        if math.isnan(t) or math.isinf(t):
                            t = 0.0
                        tau_meas.append(abs(float(t)))
                    except Exception:
                        tau_meas.append(0.0)

                # reset
                try:
                    motor.setPosition(0.0)
                    motor.setVelocity(motor.getMaxVelocity() * 0.15)
                except Exception:
                    pass

                tau_limit = self._calculate_tau_limit()
                self_can_raw = self._score_self_can_raw(theta_cmd, theta_meas, omega_meas, tau_meas, tau_limit)
                self.send_self_diag(leg_id, trial_index, theta_meas, tau_meas, tau_limit, "NORMAL", self_can_raw)
                print(f"[spot_new] {leg_id} trial {trial_index}/{diag_config.TRIAL_COUNT} done self_can_raw={self_can_raw:.3f}")

                # 重要: customData は同一ステップ内の最後の setSFString が勝つ。
                # ここで step せず次の TRIGGER を送ると SELF_DIAG が上書きされ、
                # Drone 側に一度も観測されないことがある。
                if self.robot.step(self.time_step) == -1:
                    return

        # keep stepping a little so drone can consume last messages
        for _ in range(10):
            if self.robot.step(self.time_step) == -1:
                return

        print("[spot_new] diagnosis complete")


def main() -> None:
    SpotSelfDiagnosis().run()


if __name__ == "__main__":
    main()
