#!/usr/bin/env python3
"""Spot 自己診断コントローラ（webots_new 簡潔版）

役割:
- 各脚について 6 試行の小さな関節動作を実行
- Drone に customData で以下を送る
  - TRIGGER: 試行開始
  - JOINT_ANGLES: 観測フレーム（関節角）
  - SELF_DIAG: Spot の自己診断（試行の要約 + self_can_raw）

メッセージ形式（既存互換）:
- TRIGGER|leg_id|trial_index|direction|start_time|duration_ms[|trial_angle_deg_effective]
- JOINT_ANGLES|leg_id|trial_index|a0|a1|a2
- SELF_DIAG|leg_id|trial_index|theta_samples|theta_avg|theta_final|tau_avg|tau_max|tau_nominal|safety|self_can_raw
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from controller import Supervisor


CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]

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


def _clamp_to_motor_limits(motor, v_rad: float) -> float:
    """motor の min/maxPosition に収まるように角度(rad)をクランプする。

    センサ異常や初期化不良で、現在角が物理的な可動域を超えて見えることがある。
    その値をそのまま setPosition すると Webots が警告を出し、試行が破綻しやすい。
    """
    try:
        fv = float(v_rad)
    except Exception:
        return 0.0
    try:
        if motor is None:
            return fv
        min_pos = float(motor.getMinPosition())
        max_pos = float(motor.getMaxPosition())
        if min_pos != float("-inf") and fv < min_pos:
            return min_pos
        if max_pos != float("inf") and fv > max_pos:
            return max_pos
    except Exception:
        return fv
    return fv


def _rpy_rad_from_orientation_matrix(o: List[float]) -> Tuple[float, float, float]:
    """Webotsのorientation(3x3)から roll/pitch/yaw を求める（rad）。"""
    # Drone側の実装と同形にして、姿勢の定義ずれを避ける
    roll = math.atan2(o[7], o[8])
    pitch = math.asin(-o[6])
    yaw = math.atan2(o[3], o[0])
    return roll, pitch, yaw


def _wrap_to_pi(x: float) -> float:
    """角度差を [-pi, pi] に折り返す。"""
    x = float(x)
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


class SpotSelfDiagnosis:
    def __init__(self) -> None:
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self._start_time = float(self.robot.getTime())
        self._max_runtime_s = float(os.getenv("SPOT_MAX_RUNTIME_S", "180"))
        # Spot側で simulationQuit してしまうと、Drone側のfinalize（LLM推論など）が長い場合に
        # Droneプロセスが取り残されて「Forced termination」になりやすい。
        # そのため既定は無効化し、必要な場合のみ SPOT_AUTO_QUIT=1 で有効化する。
        self._auto_quit = os.getenv("SPOT_AUTO_QUIT", "0").strip() == "1"

        self.self_node = self.robot.getSelf()
        self.custom_data_field = self.self_node.getField("customData")

        # 姿勢安定化（簡易）
        self._stab_enable = os.getenv("SPOT_STAB_ENABLE", "1").strip() == "1"
        self._stab_kp_roll = float(os.getenv("SPOT_STAB_KP_ROLL", "0.9"))
        self._stab_kp_pitch = float(os.getenv("SPOT_STAB_KP_PITCH", "0.9"))
        self._stab_max_rad = float(os.getenv("SPOT_STAB_MAX_RAD", "0.14"))  # 約8deg
        self._stab_filter_alpha = float(os.getenv("SPOT_STAB_FILTER_ALPHA", "0.25"))
        # 姿勢補正の配分（既定は knee のみ。hip/shoulder は必要なら環境変数で有効化）
        self._stab_gain_shoulder = float(os.getenv("SPOT_STAB_GAIN_SHOULDER", "0.0"))
        self._stab_gain_hip = float(os.getenv("SPOT_STAB_GAIN_HIP", "0.0"))
        self._stab_gain_knee = float(os.getenv("SPOT_STAB_GAIN_KNEE", "1.0"))
        self._stance_vel = float(os.getenv("SPOT_STANCE_VEL", "0.25"))
        self._stab_roll_err_f = 0.0
        self._stab_pitch_err_f = 0.0
        self._init_roll_pitch: Optional[Tuple[float, float]] = None

        # 転倒しそうな傾きが出たら、試行の速度を落とす/中断する（転倒後リセット依存を減らす）
        self._tilt_guard_enable = os.getenv("SPOT_TILT_GUARD_ENABLE", "1").strip() == "1"
        # 既定は「早めに減速・かなり倒れたら中断」寄り（過度な中断でデータが欠けないように）
        self._tilt_guard_slow_rad = float(os.getenv("SPOT_TILT_GUARD_SLOW_RAD", "0.35"))  # 約20deg
        self._tilt_guard_abort_rad = float(os.getenv("SPOT_TILT_GUARD_ABORT_RAD", "1.20"))  # 約69deg
        self._tilt_guard_min_vel_scale = float(os.getenv("SPOT_TILT_GUARD_MIN_VEL_SCALE", "0.05"))
        self._last_tilt_abs = 0.0

        # レッグ間での姿勢崩れ（他脚の拘束の影響）を抑えるため、各脚の試行開始前に
        # 可能な範囲で初期姿勢へ戻す（Supervisorなので軽量に実施できる）。
        self._reset_between_legs = os.getenv("SPOT_RESET_BETWEEN_LEGS", "1").strip() == "1"
        self._reset_settle_steps = int(os.getenv("SPOT_RESET_SETTLE_STEPS", "12"))
        self._init_translation: Optional[List[float]] = None
        self._init_rotation: Optional[List[float]] = None
        self._init_joint_rad: Dict[str, List[float]] = {leg_id: [0.0, 0.0, 0.0] for leg_id in diag_config.LEG_IDS}

        # Drone側はTRIGGER後に最初の数フレームで足先ベースライン(中央値)を確定してから end_disp を記録する。
        # ここで即座に関節を動かすと、ベースラインが「動作中の姿勢」になり、NONEでも end_disp が小さく見える。
        # そのため TRIGGER 直後に短いベースライン期間を入れてから動作を開始する。
        self._baseline_steps = int(os.getenv("SPOT_BASELINE_STEPS", "6"))

        self.motors: Dict[str, List] = {}
        self.sensors: Dict[str, List] = {}
        self._init_devices()

        self._queue: List[str] = []

        # JOINT_ANGLES送信用：センサが一時的にNaN/欠損になることがあるため、
        # 最後に取得できた有効角(rad)を保持してフォールバックに使う。
        self._last_joint_rad: Dict[Tuple[str, int], float] = {}

        self._malfunction_legs = self._parse_malfunction_legs(sys.argv[1:])

        # 環境オブジェクト（TRAPPED/BURIED等）は「ワールド初期配置」を前提に置かれているため、
        # ここでは physics で動いた後の実位置(getPosition)ではなく、フィールド値（初期配置）を保持する。
        try:
            self._init_translation = list(self.self_node.getField("translation").getSFVec3f())
        except Exception:
            self._init_translation = None
        try:
            self._init_rotation = list(self.self_node.getField("rotation").getSFRotation())
        except Exception:
            self._init_rotation = None

        # 起動直後はセンサ値が NaN のことがあるので、少し step して安定させる
        for _ in range(50):
            if self.robot.step(self.time_step) == -1:
                break

        # 初期関節角（rad）を保持
        for lid in diag_config.LEG_IDS:
            lm = self.motors.get(lid, [])
            ls = self.sensors.get(lid, [])
            vals: List[float] = []
            for j in range(3):
                m = lm[j] if j < len(lm) else None
                s = ls[j] if j < len(ls) else None
                v0 = self._read_joint_rad(lid, j, m, s)
                if m is not None:
                    v0 = _clamp_to_motor_limits(m, v0)
                vals.append(float(v0))
            if len(vals) == 3:
                self._init_joint_rad[lid] = vals

        # 初期姿勢（roll/pitch）を記録
        try:
            ori = list(self.self_node.getOrientation())
            roll, pitch, _ = _rpy_rad_from_orientation_matrix(ori)
            self._init_roll_pitch = (roll, pitch)
        except Exception:
            self._init_roll_pitch = None

        session_id = f"spot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"[spot_new] init session={session_id} timestep={self.time_step}ms")

    def _reset_pose_for_leg_start(self) -> None:
        if not self._reset_between_legs:
            return

        try:
            if self._init_translation is not None:
                # Supervisor.getPosition は field ではないため、Field経由で set
                self.self_node.getField("translation").setSFVec3f(list(self._init_translation))
            if self._init_rotation is not None:
                self.self_node.getField("rotation").setSFRotation(list(self._init_rotation))
            try:
                self.self_node.resetPhysics()
            except Exception:
                pass
        except Exception:
            # 姿勢復元に失敗しても試行は継続
            pass

        # 関節を初期角へ戻してから少しsettle
        for lid in diag_config.LEG_IDS:
            motors = self.motors.get(lid, [])
            targets = self._init_joint_rad.get(lid) or [0.0, 0.0, 0.0]
            for j in range(min(3, len(motors))):
                motor = motors[j]
                if motor is None:
                    continue
                t = _clamp_to_motor_limits(motor, float(targets[j]))
                try:
                    motor.setPosition(t)
                    motor.setVelocity(motor.getMaxVelocity() * 0.20)
                except Exception:
                    pass

        settle = max(0, int(self._reset_settle_steps))
        for _ in range(settle):
            if self.robot.step(self.time_step) == -1:
                return

    # ---- posture stabilization ----

    def _update_tilt_filter(self) -> Tuple[float, float]:
        """現在姿勢から roll/pitch 誤差（フィルタ済み）を更新し返す。"""
        # 姿勢制御を切っていても、傾きガード用に tilt 自体は測る
        if self._init_roll_pitch is None:
            self._stab_roll_err_f = 0.0
            self._stab_pitch_err_f = 0.0
            self._last_tilt_abs = 0.0
            return 0.0, 0.0

        try:
            ori = list(self.self_node.getOrientation())
            roll, pitch, _ = _rpy_rad_from_orientation_matrix(ori)
        except Exception:
            self._last_tilt_abs = 0.0
            return self._stab_roll_err_f, self._stab_pitch_err_f

        init_roll, init_pitch = self._init_roll_pitch
        roll_err = _wrap_to_pi(roll - init_roll)
        pitch_err = _wrap_to_pi(pitch - init_pitch)

        a = max(0.0, min(1.0, self._stab_filter_alpha))
        self._stab_roll_err_f = (1.0 - a) * self._stab_roll_err_f + a * roll_err
        self._stab_pitch_err_f = (1.0 - a) * self._stab_pitch_err_f + a * pitch_err
        self._last_tilt_abs = max(abs(self._stab_roll_err_f), abs(self._stab_pitch_err_f))
        return self._stab_roll_err_f, self._stab_pitch_err_f

    def _tilt_abs_raw(self) -> float:
        """現在姿勢の傾き(rad)を返す（yaw非依存）。

        Webotsのorientation(3x3)を local->world の回転行列Rとみなし、
        local上方向(0,1,0)のworld成分 up_world = R*(0,1,0) = (R01,R11,R21) を用いる。
        world上方向(0,1,0)とのなす角が tilt。
        """
        if self._init_roll_pitch is None:
            return 0.0
        try:
            ori = list(self.self_node.getOrientation())
        except Exception:
            return 0.0

        # up_world · world_up = up_world_y = R11 = ori[4]
        c = float(ori[4])
        if math.isnan(c) or math.isinf(c):
            return 0.0
        c = max(-1.0, min(1.0, c))
        return float(math.acos(c))

    def _compute_posture_offsets(self) -> Dict[str, Dict[int, float]]:
        """ロール/ピッチ誤差から、各脚・各関節(0..2)への補正量(rad)を返す。"""
        roll_err_f, pitch_err_f = self._update_tilt_filter()
        if not self._stab_enable:
            return {leg_id: {0: 0.0, 1: 0.0, 2: 0.0} for leg_id in diag_config.LEG_IDS}

        u_roll = -self._stab_kp_roll * roll_err_f
        u_pitch = -self._stab_kp_pitch * pitch_err_f

        def clamp(x: float) -> float:
            lim = abs(self._stab_max_rad)
            if lim <= 0.0:
                return 0.0
            return max(-lim, min(lim, x))

        g_shoulder = float(self._stab_gain_shoulder)
        g_hip = float(self._stab_gain_hip)
        g_knee = float(self._stab_gain_knee)

        offsets: Dict[str, Dict[int, float]] = {}
        for leg_id in diag_config.LEG_IDS:
            side = 1.0 if leg_id in ("FL", "RL") else -1.0
            front = 1.0 if leg_id in ("FL", "FR") else -1.0
            u = clamp(u_roll * side + u_pitch * front)
            offsets[leg_id] = {
                0: float(u * g_shoulder),
                1: float(u * g_hip),
                2: float(u * g_knee),
            }
        return offsets

    def _apply_stance_targets(
        self,
        base_targets: Dict[str, List[float]],
        active_leg: str,
        active_motor_index: int,
    ) -> None:
        """非アクティブ脚を初期姿勢へ維持しつつ、必要なら姿勢補正を加える。"""
        offsets = self._compute_posture_offsets()
        for leg_id in diag_config.LEG_IDS:
            # 故障脚は「動かない」を再現するため、姿勢制御の対象外
            if leg_id in self._malfunction_legs:
                continue

            motors = self.motors.get(leg_id, [])
            if not motors or len(motors) < 3:
                continue

            targets = base_targets.get(leg_id)
            if not targets or len(targets) < 3:
                continue

            for j in range(3):
                motor = motors[j]
                if motor is None:
                    continue
                if leg_id == active_leg and j == active_motor_index:
                    # この関節は試行で明示的に動かす
                    continue

                t = float(targets[j])
                if self._stab_enable:
                    t = float(t + offsets.get(leg_id, {}).get(j, 0.0))

                # 姿勢補正で可動域外へ出ないようにクランプ
                t = _clamp_to_motor_limits(motor, t)

                try:
                    motor.setPosition(t)
                    # 低速で「踏ん張る」程度にして、過大な反力を作りにくくする
                    v = max(0.0, min(1.0, float(self._stance_vel)))
                    motor.setVelocity(motor.getMaxVelocity() * v)
                except Exception:
                    pass

    def _parse_malfunction_legs(self, argv: List[str]) -> set[str]:
        legs: set[str] = set()
        for a in argv:
            if not isinstance(a, str):
                continue
            if a.startswith("--malfunction_legs="):
                raw = a.split("=", 1)[1].strip()
                if not raw:
                    continue
                for part in raw.split(","):
                    leg = part.strip().upper()
                    if leg in diag_config.LEG_IDS:
                        legs.add(leg)
        if legs:
            print(f"[spot_new] malfunction_legs={sorted(legs)}")
        return legs

    # ---- config / devices ----

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

    def send_trigger(
        self,
        leg_id: str,
        trial_index: int,
        direction: str,
        start_time: float,
        duration_ms: int,
        trial_angle_deg_effective: Optional[float] = None,
    ) -> None:
        # 既存互換のため、追加フィールドは末尾に付ける（Drone側はあれば利用、無ければ無視）
        if trial_angle_deg_effective is None:
            self._send(f"TRIGGER|{leg_id}|{trial_index}|{direction}|{start_time:.6f}|{duration_ms}")
            return
        try:
            ang = float(trial_angle_deg_effective)
        except Exception:
            ang = None
        if ang is None or (not _is_finite(ang)):
            self._send(f"TRIGGER|{leg_id}|{trial_index}|{direction}|{start_time:.6f}|{duration_ms}")
            return
        self._send(f"TRIGGER|{leg_id}|{trial_index}|{direction}|{start_time:.6f}|{duration_ms}|{ang:.6f}")

    def send_joint_angles(self, leg_id: str, trial_index: int) -> None:
        motors = self.motors.get(leg_id, [])
        sensors = self.sensors.get(leg_id, [])
        if len(motors) < 3 or len(sensors) < 3:
            return

        angles = []
        for i in range(3):
            motor = motors[i] if i < len(motors) else None
            sensor = sensors[i] if i < len(sensors) else None
            v = self._read_joint_rad(leg_id, i, motor, sensor)
            angles.append(f"{_deg(v):.6f}")
        self._send(f"JOINT_ANGLES|{leg_id}|{trial_index}|" + "|".join(angles))

    def _read_joint_rad(self, leg_id: str, joint_i: int, motor, sensor) -> float:
        """関節角(rad)を取得する。

        - PositionSensor が NaN/inf のままのことがあるため、少し待って再試行する
        - それでもダメなら motor の target position を代替値として使う
        """
        key = (str(leg_id), int(joint_i))

        # まずはセンサを優先（ここではstepしない：送信周期が崩れるため）
        if sensor is not None:
            try:
                raw = float(sensor.getValue())
            except Exception:
                raw = None
            if raw is not None and _is_finite(raw) and abs(raw) <= 3.0:
                v = float(raw)
                self._last_joint_rad[key] = v
                return v

        # フォールバック1: 直近の有効値
        v_last = self._last_joint_rad.get(key)
        if v_last is not None and _is_finite(v_last) and abs(v_last) <= 3.0:
            return float(v_last)

        # フォールバック: motor の target position（取得できる環境のみ）
        if motor is not None:
            try:
                v = _sanitize_sensor_rad(motor.getTargetPosition())
                if _is_finite(v) and abs(v) <= 3.0:
                    self._last_joint_rad[key] = float(v)
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

        # leg_id/joint_i がここでは分からないため、保持フォールバックは使わずに取得する
        # （"unknown" キーのキャッシュが別関節に汚染されるのを避ける）
        current = 0.0
        try:
            raw = float(sensor.getValue())
            if _is_finite(raw) and abs(raw) <= 3.0:
                current = float(raw)
            else:
                raise ValueError("sensor value not finite")
        except Exception:
            try:
                current = float(_sanitize_sensor_rad(motor.getTargetPosition()))
            except Exception:
                current = 0.0

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
            denom = float(getattr(diag_config, "E_MAX_DEG", 3.0) or 3.0)
            track = max(0.0, min(1.0, 1.0 - (sum(errors) / len(errors)) / max(1e-6, denom)))
        else:
            track = 0.0

        # velocity
        if omega_meas:
            peak = max(abs(v) for v in omega_meas)
            denom = float(getattr(diag_config, "OMEGA_REF_DEG_PER_SEC", 27.0) or 27.0)
            vel = max(0.0, min(1.0, peak / max(1e-6, denom)))
        else:
            vel = 0.0

        # torque: 強い抵抗が出ている場合は「指令通りに動けていない」可能性が高い。
        # （環境ラベルは参照せず、センサ値のみから間接的に拘束を反映する）
        valid_tau = [abs(float(t)) for t in (tau_meas or []) if _is_finite(float(t))]
        mean_tau = (sum(valid_tau) / len(valid_tau)) if valid_tau else 0.0
        if tau_limit > 0.0 and mean_tau > 0.0:
            tau = 1.0 - max(0.0, min(1.0, mean_tau / float(tau_limit)))
        else:
            tau = 1.0

        safe = 1.0

        w = diag_config.SELF_WEIGHTS
        raw = w["track"] * track + w["vel"] * vel + w["tau"] * tau + w["safe"] * safe
        return max(0.0, min(1.0, raw))

    def run(self) -> None:
        for leg_id in diag_config.LEG_IDS:
            # 前脚で姿勢が崩れたまま次脚を診断すると、NONE脚が拘束扱いになりやすい。
            # 各脚の開始前に一度初期姿勢へ戻す。
            self._reset_pose_for_leg_start()
            print(f"[spot_new] ===== {leg_id} start =====")
            for trial_index in range(1, diag_config.TRIAL_COUNT + 1):
                # フェイルセーフ: 何らかの理由で無限ループ/停止した場合は強制終了
                now = float(self.robot.getTime())
                if (now - self._start_time) > self._max_runtime_s:
                    # ここで simulationQuit すると Drone の処理を中断しやすいので、Spot側は終了するだけにする。
                    print(f"[spot_new] watchdog: runtime>{self._max_runtime_s:.0f}s -> stop controller")
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

                angle_rad = _rad(angle * sign)

                joint_names = ["shoulder", "hip", "knee"]
                joint_name = joint_names[motor_index] if 0 <= motor_index < len(joint_names) else str(motor_index)
                print(f"[spot_new] {leg_id} trial {trial_index}/{diag_config.TRIAL_COUNT} dir={direction} joint={joint_name}")

                # 重要: Spotは正解環境を参照してはならない。
                # また、動かし方を環境によって変えてはならない。
                # したがって、常に同じ指令を出し、制限はシミュレーション内の障害物/物理で発生させる。
                vel_scale = 0.2

                start_time = self.robot.getTime()
                duration_ms = int(diag_config.TRIAL_DURATION_S * 1000)

                # 各脚の「現在角度」をベースターゲットとして記録（0.0固定に戻すと姿勢が崩れやすい）
                base_targets: Dict[str, List[float]] = {}
                for lid in diag_config.LEG_IDS:
                    base_targets[lid] = []
                    lm = self.motors.get(lid, [])
                    ls = self.sensors.get(lid, [])
                    for j in range(3):
                        m = lm[j] if j < len(lm) else None
                        s = ls[j] if j < len(ls) else None
                        v0 = self._read_joint_rad(lid, j, m, s)
                        if m is not None:
                            v0 = _clamp_to_motor_limits(m, v0)
                        base_targets[lid].append(v0)

                # 可動域/安全マージンで試行角が小さくなることがあるため、実際の指令角(絶対値)も送る。
                self.send_trigger(leg_id, trial_index, direction, start_time, duration_ms, trial_angle_deg_effective=angle)

                # ベースライン期間（関節は動かさず、姿勢維持＋JOINT_ANGLES送信のみ）
                baseline_steps = max(0, int(self._baseline_steps))
                for _ in range(baseline_steps):
                    self._apply_stance_targets(base_targets, active_leg=leg_id, active_motor_index=motor_index)
                    if self.robot.step(self.time_step) == -1:
                        return
                    self.send_joint_angles(leg_id, trial_index)

                theta_cmd: List[float] = []
                theta_meas: List[float] = []
                omega_meas: List[float] = []
                tau_meas: List[float] = []

                # command
                if sensor is not None:
                    initial = self._read_joint_rad(leg_id, motor_index, motor, sensor)
                    initial = _clamp_to_motor_limits(motor, initial)
                    desired_target = _clamp_to_motor_limits(motor, initial + angle_rad)
                    target = desired_target
                else:
                    initial = 0.0
                    desired_target = _clamp_to_motor_limits(motor, angle_rad)
                    target = desired_target

                if not _is_finite(target):
                    # 念のため NaN/inf を弾く
                    continue

                try:
                    if leg_id in self._malfunction_legs:
                        # 故障: 動作指令を出しても脚が動かない状況を、制御入力の無効化で再現する。
                        # これにより Drone(RoboPose) だけでも MALFUNCTION を推定できる。
                        if sensor is not None:
                            v = _sanitize_sensor_rad(sensor.getValue())
                            v = _clamp_to_motor_limits(motor, v)
                            motor.setPosition(v)
                        motor.setVelocity(0.0)
                    else:
                        motor.setPosition(target)
                        motor.setVelocity(motor.getMaxVelocity() * vel_scale)
                except Exception:
                    pass

                # run for duration
                steps = max(1, int((diag_config.TRIAL_DURATION_S * 1000) / max(1, self.time_step)))
                prev_theta = None

                for _ in range(steps):
                    # 非アクティブ脚を支え姿勢を維持（+必要ならロール/ピッチ補正）
                    self._apply_stance_targets(base_targets, active_leg=leg_id, active_motor_index=motor_index)

                    # 転倒しそうな傾きが出たら、試行の進行を抑える（倒れてからのリセット依存を減らす）
                    if self._tilt_guard_enable and leg_id not in self._malfunction_legs:
                        tilt = float(self._tilt_abs_raw())
                        abort = abs(float(self._tilt_guard_abort_rad))
                        slow = abs(float(self._tilt_guard_slow_rad))
                        if abort > 0.0 and slow > 0.0 and tilt >= slow:
                            # slow..abort の間で線形に速度を落とす
                            denom = max(1e-6, (abort - slow))
                            r = max(0.0, min(1.0, (tilt - slow) / denom))
                            scale = max(float(self._tilt_guard_min_vel_scale), 1.0 - r)
                            try:
                                motor.setVelocity(motor.getMaxVelocity() * vel_scale * scale)
                            except Exception:
                                pass

                    if self.robot.step(self.time_step) == -1:
                        return

                    # step後の姿勢で危険域なら試行を打ち切る（0フレームを避けるため step 後に判定）
                    if self._tilt_guard_enable and leg_id not in self._malfunction_legs:
                        abort = abs(float(self._tilt_guard_abort_rad))
                        tilt_post = float(self._tilt_abs_raw())
                        if abort > 0.0 and tilt_post >= abort:
                            print(f"[spot_new] tilt_guard: abort tilt={tilt_post:.3f}rad leg={leg_id} trial={trial_index}")
                            break

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
                    if leg_id in self._malfunction_legs:
                        if sensor is not None:
                            v = _sanitize_sensor_rad(sensor.getValue())
                            v = _clamp_to_motor_limits(motor, v)
                            motor.setPosition(v)
                        motor.setVelocity(0.0)
                    else:
                        # 試行開始時の角度へ戻す（絶対0.0へ戻すと姿勢が崩れやすい）
                        motor.setPosition(_clamp_to_motor_limits(motor, initial))
                        motor.setVelocity(motor.getMaxVelocity() * 0.15)
                except Exception:
                    pass

                tau_limit = self._calculate_tau_limit()
                self_can_raw = self._score_self_can_raw(theta_cmd, theta_meas, omega_meas, tau_meas, tau_limit)
                if leg_id in self._malfunction_legs:
                    # 仕様.txt: 故障は「spotが動く・droneが動かない」またはその逆で判定する。
                    # ここでは Spot 側の自己診断だけを "動く" と主張させ、Drone 観測は（指令無効化により）動かない → 不一致を再現する。
                    self_can_raw = 1.0
                self.send_self_diag(
                    leg_id,
                    trial_index,
                    theta_meas,
                    tau_meas,
                    tau_limit,
                    "NORMAL",
                    self_can_raw,
                    malfunction_flag=(1 if leg_id in self._malfunction_legs else 0),
                )
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

        # Spot側は基本的に Drone の simulationQuit を待つ。
        # （Spotが先にquitすると Drone が取り残されてハング/残プロセスの原因になりやすい）
        if self._auto_quit:
            # 明示的に有効化された場合のみ、一定時間待ってから終了する。
            wait_s = float(os.getenv("SPOT_AUTO_QUIT_WAIT_S", "30"))
            steps = int(wait_s / max(0.001, self.time_step / 1000.0))
            for _ in range(max(1, steps)):
                if self.robot.step(self.time_step) == -1:
                    return
            print("[spot_new] auto quit (enabled)")
            try:
                self.robot.simulationQuit(0)
            except Exception:
                pass
        else:
            # 終了要求が来るまでアイドルし、Drone側の完了を待つ。
            while self.robot.step(self.time_step) != -1:
                pass


def main() -> None:
    SpotSelfDiagnosis().run()


if __name__ == "__main__":
    main()
