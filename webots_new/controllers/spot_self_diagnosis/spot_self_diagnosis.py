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
import os
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


def _sanitize_sensor_rad(v: float) -> float:
    """センサの異常値を丸める（NaN/inf/極端な値を0扱いにする）。"""
    try:
        fv = float(v)
    except Exception:
        return 0.0
    if not _is_finite(fv):
        return 0.0
    # 初期化不良などで桁違いの値が出ることがあるため、汎用的に上限を設ける
    # Spotの関節は通常 ±数rad 程度なので、それを大きく超える値は異常扱いにする
    if abs(fv) > 3.0:  # rad（約172deg）を超えるなら異常扱い
        return 0.0
    return fv


class SpotSelfDiagnosis:
    def __init__(self) -> None:
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self._start_time = float(self.robot.getTime())
        self._max_runtime_s = float(os.getenv("SPOT_MAX_RUNTIME_S", "180"))
        self._auto_quit = os.getenv("SPOT_AUTO_QUIT", "1").strip() == "1"

        self.self_node = self.robot.getSelf()
        self.custom_data_field = self.self_node.getField("customData")

        self.motors: Dict[str, List] = {}
        self.sensors: Dict[str, List] = {}
        self._init_devices()

        self.scenario = self._load_scenario()

        self._queue: List[str] = []

        # 起動直後はセンサ値が NaN のことがあるので、少し step して安定させる
        for _ in range(50):
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
        motors = self.motors.get(leg_id, [])
        sensors = self.sensors.get(leg_id, [])
        if len(motors) < 3 or len(sensors) < 3:
            return

        angles = []
        for i in range(3):
            motor = motors[i] if i < len(motors) else None
            sensor = sensors[i] if i < len(sensors) else None
            v = self._read_joint_rad(motor, sensor)
            angles.append(f"{_deg(v):.6f}")
        self._send(f"JOINT_ANGLES|{leg_id}|{trial_index}|" + "|".join(angles))

    def _read_joint_rad(self, motor, sensor) -> float:
        """関節角(rad)を取得する。

        - PositionSensor が NaN/inf のままのことがあるため、少し待って再試行する
        - それでもダメなら motor の target position を代替値として使う
        """
        # まずはセンサを優先（ここではstepしない：送信周期が崩れるため）
        if sensor is not None:
            try:
                raw = float(sensor.getValue())
            except Exception:
                raw = None
            if raw is not None and _is_finite(raw) and abs(raw) <= 3.0:
                return float(raw)

        # フォールバック: motor の target position（取得できる環境のみ）
        if motor is not None:
            try:
                v = _sanitize_sensor_rad(motor.getTargetPosition())
                if _is_finite(v) and abs(v) <= 3.0:
                    return float(v)
            except Exception:
                pass

        return 0.0

    def send_self_diag(
        self,
        leg_id: str,
        trial_index: int,
        theta_meas: List[float],
        tau_meas: List[float],
        tau_nominal: float,
        safety_level: str,
        self_can_raw: float,
        malfunction_flag: int = 0,
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
            f"{tau_nominal:.6f}|{safety_level}|{self_can_raw:.6f}|{int(malfunction_flag)}"
        )
        self._send(msg)

    # ---- trial execution ----

    def _safe_angle(self, motor, sensor) -> Tuple[float, float]:
        if motor is None or sensor is None:
            return 0.0, 0.0

        current = self._read_joint_rad(motor, sensor)

        current_deg = _deg(current)

        min_pos = motor.getMinPosition()
        max_pos = motor.getMaxPosition()
        if min_pos == float("-inf"):
            min_pos = current - _rad(30)
        if max_pos == float("inf"):
            max_pos = current + _rad(30)

        min_deg = _deg(min_pos)
        max_deg = _deg(max_pos)
        margin = 1.0

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
                # フェイルセーフ: 何らかの理由で無限ループ/停止した場合は強制終了
                now = float(self.robot.getTime())
                if (now - self._start_time) > self._max_runtime_s:
                    print(f"[spot_new] watchdog: runtime>{self._max_runtime_s:.0f}s -> quit")
                    try:
                        self.robot.simulationQuit(1)
                    except Exception:
                        pass
                    return

                direction = diag_config.TRIAL_PATTERN[trial_index - 1]
                motor_index = diag_config.TRIAL_MOTOR_INDICES[trial_index - 1]

                motors = self.motors.get(leg_id, [])
                sensors = self.sensors.get(leg_id, [])
                if len(motors) <= motor_index or not motors[motor_index]:
                    print(f"[spot_new] warn: motor missing {leg_id} idx={motor_index}")
                    continue

                motor = motors[motor_index]
                sensor = sensors[motor_index] if len(sensors) > motor_index else None

                if sensor is None:
                    try:
                        motor_name = LEG_MOTOR_NAMES.get(leg_id, [])[motor_index]
                        sensor_name = motor_name.replace("motor", "sensor")
                    except Exception:
                        motor_name = ""
                        sensor_name = ""
                    print(f"[spot_new] warn: sensor missing {leg_id} idx={motor_index} motor='{motor_name}' sensor='{sensor_name}'")

                safe_pos, safe_neg = self._safe_angle(motor, sensor)
                requested = diag_config.TRIAL_ANGLE_DEG

                if direction == "+":
                    # 角度が小さくても試行自体は実行してデータを送る（セッション未出力を防ぐ）
                    angle = min(requested, max(0.0, safe_pos))
                    sign = 1.0
                else:
                    angle = min(requested, max(0.0, safe_neg))
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
                # MALFUNCTION は「指示を出しても関節が動かない」状態を再現する。
                # 実装上は「指示（意図）は出すが、アクチュエータが無視して動かない」を再現する。
                elif env == "MALFUNCTION":
                    vel_scale = 0.2
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
                if sensor is not None:
                    initial = self._read_joint_rad(motor, sensor)
                    desired_target = initial + angle_rad
                    # MALFUNCTION: 意図したtargetは記録するが、実際のmotor指令は初期角度のままにする
                    target = initial if env == "MALFUNCTION" else desired_target
                else:
                    initial = 0.0
                    desired_target = angle_rad
                    target = 0.0 if env == "MALFUNCTION" else desired_target

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

                    if sensor is None:
                        continue

                    v = sensor.getValue()
                    v = _sanitize_sensor_rad(v)
                    theta = _deg(v)
                    theta_meas.append(theta)

                    # commanded angle (deg) for tracking: target position (deg)
                    theta_cmd.append(_deg(desired_target))

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
                    # MALFUNCTION のときも確実に止める
                    reset_vel = 0.0 if env == "MALFUNCTION" else (motor.getMaxVelocity() * 0.15)
                    motor.setVelocity(reset_vel)
                except Exception:
                    pass

                tau_limit = self._calculate_tau_limit()
                self_can_raw = self._score_self_can_raw(theta_cmd, theta_meas, omega_meas, tau_meas, tau_limit)
                if env == "MALFUNCTION":
                    # 「指示しても動かない」ので、自己診断は低くする
                    self_can_raw = 0.0
                self.send_self_diag(
                    leg_id,
                    trial_index,
                    theta_meas,
                    tau_meas,
                    tau_limit,
                    "NORMAL",
                    self_can_raw,
                    malfunction_flag=1 if env == "MALFUNCTION" else 0,
                )
                print(f"[spot_new] {leg_id} trial {trial_index}/{diag_config.TRIAL_COUNT} done self_can_raw={self_can_raw:.3f}")

                # MALFUNCTION の確認用ログ（指示Δと実測Δ）
                if env == "MALFUNCTION":
                    try:
                        cmd_delta_deg = float(angle * sign)
                    except Exception:
                        cmd_delta_deg = 0.0
                    if theta_meas:
                        actual_delta_deg = float(theta_meas[-1] - theta_meas[0])
                        print(
                            f"[spot_new] MALFUNCTION_CHECK leg={leg_id} trial={trial_index} joint={joint_name} "
                            f"cmd_delta_deg={cmd_delta_deg:.3f} actual_delta_deg={actual_delta_deg:.3f}"
                        )
                    else:
                        print(
                            f"[spot_new] MALFUNCTION_CHECK leg={leg_id} trial={trial_index} joint={joint_name} "
                            f"cmd_delta_deg={cmd_delta_deg:.3f} actual_delta_deg=NaN (no sensor samples)"
                        )

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

        # Drone側が落ちた/quitできないと Webots が開きっぱなしになることがあるので、
        # Spot側でもフェイルセーフとして終了する（必要なら SPOT_AUTO_QUIT=0 で無効化）。
        if self._auto_quit:
            # drone finalize の猶予
            for _ in range(int(5.0 / max(0.001, self.time_step / 1000.0))):
                if self.robot.step(self.time_step) == -1:
                    return
            print("[spot_new] auto quit (failsafe)")
            try:
                self.robot.simulationQuit(0)
            except Exception:
                pass


def main() -> None:
    SpotSelfDiagnosis().run()


if __name__ == "__main__":
    main()
