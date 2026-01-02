#!/usr/bin/env python3
"""Drone コントローラ（webots_new 簡潔版）

役割:
- Spot の customData を監視し、TRIGGER/JOINT_ANGLES/SELF_DIAG を処理
- 試行の間に pipeline.record_robo_pose_frame() で観測フレームを蓄積
- 試行終了後に pipeline.complete_trial() を呼び、最終的に pipeline.finalize() で JSONL を出力

既存互換:
- Spot からのメッセージ形式は既存版と同じ
- ログは controllers/drone_circular_controller/logs/ に出力される（view_result.py が読む）
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


HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # logs をこのフォルダ配下に作るため

CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CONTROLLERS_ROOT.parent / "config" / "scenario.ini"

if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.append(str(CONTROLLERS_ROOT))

from diagnostics_pipeline import config as diag_config  # noqa: E402
from diagnostics_pipeline.pipeline import DiagnosticsPipeline  # noqa: E402


def _rpy_rad_from_orientation_matrix(o: List[float]) -> Tuple[float, float, float]:
    """Webotsのorientation(3x3)から roll/pitch/yaw を求める（rad）"""
    roll = math.atan2(o[7], o[8])
    pitch = math.asin(-o[6])
    yaw = math.atan2(o[3], o[0])
    return roll, pitch, yaw


def _rpy_deg_from_orientation_matrix(o: List[float]) -> Tuple[float, float, float]:
    r, p, y = _rpy_rad_from_orientation_matrix(o)
    return math.degrees(r), math.degrees(p), math.degrees(y)


def _fk_foot_local(leg_id: str, joint_angles_deg: List[float]) -> Tuple[float, float, float]:
    # 既存版と同じ近似パラメータ
    L1, L2, L3 = 0.11, 0.26, 0.26
    y_offset = -0.25 if "L" in leg_id else 0.25

    a0, a1, a2 = (joint_angles_deg + [0.0, 0.0, 0.0])[:3]
    t1 = math.radians(a0)
    t2 = math.radians(a1)
    t3 = math.radians(a2)

    x = L2 * math.cos(t2) + L3 * math.cos(t2 + t3)
    y = y_offset + L1 * math.sin(t1)
    z = -(L2 * math.sin(t2) + L3 * math.sin(t2 + t3))
    return x, y, z


def _compensate_for_body_tilt(x: float, y: float, z: float, delta_roll: float, delta_pitch: float) -> Tuple[float, float, float]:
    """胴体の傾き（roll/pitch）変化を簡易補正する（元実装の近似に合わせる）"""
    if abs(delta_roll) <= 0.01 and abs(delta_pitch) <= 0.01:
        return x, y, z

    # 逆回転で「脚ローカルに戻す」
    cos_p, sin_p = math.cos(-delta_pitch), math.sin(-delta_pitch)
    cos_r, sin_r = math.cos(-delta_roll), math.sin(-delta_roll)

    # pitch（Y軸回り）
    x_tmp = x * cos_p + z * sin_p
    z_tmp = -x * sin_p + z * cos_p

    # roll（X軸回り）
    y_out = y * cos_r - z_tmp * sin_r
    z_out = y * sin_r + z_tmp * cos_r
    x_out = x_tmp
    return x_out, y_out, z_out


class DroneCircularController:
    def __init__(self) -> None:
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())

        session_id = f"drone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.pipeline = DiagnosticsPipeline(session_id)

        self.expected_causes = self._load_expected_causes()
        self.pipeline.set_expected_causes(self.expected_causes)

        self.offset_x, self.offset_y, self.offset_z, self.center_def = self._parse_args(sys.argv[1:])

        self.spot_node = self.supervisor.getFromDef("SPOT")
        self.spot_custom = self.spot_node.getField("customData") if self.spot_node else None

        self.drone_node = self.supervisor.getSelf()
        self.translation_field = self.drone_node.getField("translation")
        self.rotation_field = self.drone_node.getField("rotation")

        self.center_node = self.supervisor.getFromDef(self.center_def)

        self.last_custom_data = ""

        # (leg_id, trial_index) -> active state
        self.active: Dict[Tuple[str, int], Dict] = {}
        self.trials_completed: Dict[str, int] = {leg: 0 for leg in diag_config.LEG_IDS}
        self._finalized = False
        self._quit_time: float | None = None

        self._last_center_pos = [0.0, 0.0, 0.0]

        print(f"[drone_new] init session={session_id} timestep={self.time_step}ms")
        print(f"[drone_new] expected={self.expected_causes}")

    def _load_expected_causes(self) -> Dict[str, str]:
        expected: Dict[str, str] = {"FL": "NONE", "FR": "NONE", "RL": "NONE", "RR": "NONE"}
        if not CONFIG_PATH.exists():
            return expected
        cfg = configparser.ConfigParser()
        cfg.read(CONFIG_PATH)
        for leg_id in expected:
            key = f"{leg_id.lower()}_environment"
            expected[leg_id] = cfg.get("DEFAULT", key, fallback="NONE").strip().upper()
        return expected

    def _parse_args(self, args: List[str]) -> Tuple[float, float, float, str]:
        offset_x = 0.0
        offset_y = -2.0
        offset_z = 3.0
        center_def = "SPOT"

        for arg in args:
            if arg.startswith("--offset-x="):
                offset_x = float(arg.split("=", 1)[1])
            elif arg.startswith("--offset-y="):
                offset_y = float(arg.split("=", 1)[1])
            elif arg.startswith("--offset-z="):
                offset_z = float(arg.split("=", 1)[1])
            elif arg.startswith("--center-def="):
                center_def = arg.split("=", 1)[1]
            elif arg.startswith("--radius="):
                r = float(arg.split("=", 1)[1])
                offset_y = -abs(r)
            elif arg.startswith("--height="):
                offset_z = float(arg.split("=", 1)[1])
            # --period は既存 world にあるが、簡潔版では未使用

        return offset_x, offset_y, offset_z, center_def

    # ---- Spot message processing ----

    def process_messages(self) -> None:
        if not self.spot_custom:
            return

        raw = self.spot_custom.getSFString()
        if not raw or raw == self.last_custom_data:
            return
        self.last_custom_data = raw

        for msg in raw.strip().split("\n"):
            if not msg:
                continue
            parts = msg.split("|")
            tag = parts[0]

            try:
                if tag == "TRIGGER":
                    leg_id = parts[1]
                    trial_index = int(parts[2])
                    direction = parts[3]
                    start_time = float(parts[4])
                    duration_ms = int(parts[5])

                    end_time = start_time + duration_ms / 1000.0

                    self.pipeline.start_trial(leg_id, trial_index, direction, start_time, duration_ms / 1000.0)
                    self.active[(leg_id, trial_index)] = {
                        "trial_index": trial_index,
                        "direction": direction,
                        "start_time": start_time,
                        "end_time": end_time,
                        "frames": 0,
                        "init_foot": None,
                        "init_rpy": None,
                        "self_diag": None,
                    }

                    print(
                        f"[drone_new] start {leg_id} trial {trial_index}/{diag_config.TRIAL_COUNT} dir={direction}"
                    )

                elif tag == "JOINT_ANGLES":
                    leg_id = parts[1]
                    trial_index = int(parts[2])
                    angles = [float(parts[3]), float(parts[4]), float(parts[5])]

                    state = self.active.get((leg_id, trial_index))
                    if not state:
                        continue

                    if not self.spot_node:
                        continue

                    base_pos = list(self.spot_node.getPosition())

                    # 姿勢（初期と現在）
                    ori = list(self.spot_node.getOrientation())
                    roll, pitch, yaw = _rpy_rad_from_orientation_matrix(ori)
                    rpy_deg = (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

                    if state.get("init_rpy") is None:
                        state["init_rpy"] = (roll, pitch, yaw)

                    init_roll, init_pitch, _ = state["init_rpy"]
                    delta_roll = roll - init_roll
                    delta_pitch = pitch - init_pitch

                    # 足先位置（胴体ローカル）
                    x_local, y_local, z_local = _fk_foot_local(leg_id, angles)

                    # 胴体の傾き変化を補正（転倒/傾きで見かけの変位が変わるのを抑える）
                    x_local, y_local, z_local = _compensate_for_body_tilt(
                        x_local, y_local, z_local, delta_roll, delta_pitch
                    )

                    if state["init_foot"] is None:
                        state["init_foot"] = (x_local, y_local, z_local)

                    init_x, init_y, init_z = state["init_foot"]
                    end_disp = [x_local - init_x, y_local - init_y, z_local - init_z]

                    self.pipeline.record_robo_pose_frame(
                        leg_id=leg_id,
                        trial_index=trial_index,
                        joint_angles=angles,
                        end_position=end_disp,
                        base_orientation=list(rpy_deg),
                        base_position=base_pos,
                    )
                    state["frames"] += 1

                elif tag == "SELF_DIAG":
                    # SELF_DIAG|leg_id|trial_index|theta_samples|theta_avg|theta_final|tau_avg|tau_max|tau_nominal|safety|self_can_raw
                    leg_id = parts[1]
                    trial_index = int(parts[2])
                    tau_nominal = float(parts[8])
                    safety = parts[9]
                    self_can_raw = float(parts[10]) if len(parts) > 10 else 0.35

                    state = self.active.get((leg_id, trial_index))
                    if not state:
                        continue

                    state["self_diag"] = {
                        "tau_nominal": tau_nominal,
                        "safety": safety,
                        "self_can_raw": self_can_raw,
                    }

            except Exception as exc:
                print(f"[drone_new] warn: parse failed: {exc} msg={msg}")

    def _maybe_complete_trials(self) -> None:
        if self._finalized:
            return
        now = self.supervisor.getTime()
        for (leg_id, trial_index), state in list(self.active.items()):
            if now < state["end_time"]:
                continue

            # SELF_DIAG が来ていない場合は少し待つ（最大 5 秒）
            if state.get("self_diag") is None and (now - state["end_time"]) < 5.0:
                continue

            self_diag = state.get("self_diag") or {"tau_nominal": 0.0, "safety": "UNKNOWN", "self_can_raw": None}

            self.pipeline.complete_trial(
                leg_id=leg_id,
                trial_index=trial_index,
                theta_cmd=[],
                theta_meas=[],
                omega_meas=[],
                tau_meas=[],
                tau_nominal=float(self_diag["tau_nominal"]),
                safety_level=str(self_diag["safety"]),
                end_time=state["end_time"],
                spot_can_raw=self_diag.get("self_can_raw"),
            )

            self.trials_completed[leg_id] = self.trials_completed.get(leg_id, 0) + 1
            del self.active[(leg_id, trial_index)]

            frames = int(state.get("frames", 0))
            raw = self_diag.get("self_can_raw")
            raw_text = f"{float(raw):.3f}" if raw is not None else "None"
            print(
                f"[drone_new] done  {leg_id} trial {trial_index}/{diag_config.TRIAL_COUNT} frames={frames} spot_raw={raw_text}"
            )
            print(
                f"[drone_new] progress {leg_id} {self.trials_completed[leg_id]}/{diag_config.TRIAL_COUNT}"
            )

            if all(self.trials_completed[l] >= diag_config.TRIAL_COUNT for l in diag_config.LEG_IDS):
                print("[drone_new] all trials done -> finalize")
                self.pipeline.finalize()
                self._finalized = True
                # finalize 後もしばらく動かしてから終了（バッチ実行用）
                self._quit_time = now + 1.0

    # ---- drone motion ----

    def update_position(self) -> None:
        center_pos = list(self.center_node.getPosition()) if self.center_node else [0.0, 0.0, 0.0]

        # 物理が不安定になった場合に NaN が混入して Webots API エラーになることがあるのでガードする。
        if any(not math.isfinite(float(v)) for v in center_pos):
            center_pos = list(self._last_center_pos)
        else:
            self._last_center_pos = list(center_pos)

        target = [
            float(center_pos[0] + self.offset_x),
            float(center_pos[1] + self.offset_y),
            float(center_pos[2] + self.offset_z),
        ]

        try:
            self.translation_field.setSFVec3f(target)
        except Exception:
            pass

        try:
            self.drone_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        except Exception:
            pass

        dx = center_pos[0] - target[0]
        dy = center_pos[1] - target[1]
        heading = math.atan2(dy, dx)
        if not math.isfinite(float(heading)):
            return
        try:
            self.rotation_field.setSFRotation([0.0, 0.0, 1.0, float(heading)])
        except Exception:
            pass

    def run(self) -> None:
        while self.supervisor.step(self.time_step) != -1:
            self.process_messages()
            self._maybe_complete_trials()
            self.update_position()

            if self._finalized and self._quit_time is not None:
                if self.supervisor.getTime() >= self._quit_time:
                    print("[drone_new] quit simulation")
                    try:
                        self.supervisor.simulationQuit(0)
                    except Exception:
                        pass


def main() -> None:
    DroneCircularController().run()


if __name__ == "__main__":
    main()
