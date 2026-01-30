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
import signal
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


# 以前は関節角からFKで足先変位を近似していたが、外部拘束（TRAPPED/TANGLED/BURIED）下でも
# 関節角自体は動いてしまうため、足先が「動いた」と誤認しやすかった。
# 現行は Webots のシミュレーション内にある脚リンク（forearm Solid）の実測ワールド位置を使う。
# 取得に失敗した場合のみFKへフォールバックする。


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

    # NOTE: Spot から送られる JOINT_ANGLES は「度」。
    # FKは三角関数を使うので、ここでラジアンに変換して計算する。
    a1_deg, a2_deg, a3_deg = (joint_angles_deg + [0.0, 0.0, 0.0])[:3]
    t1, t2, t3 = math.radians(a1_deg), math.radians(a2_deg), math.radians(a3_deg)

    x = L2 * math.cos(t2) + L3 * math.cos(t2 + t3)
    y = y_offset + L1 * math.sin(t1)
    z = -(L2 * math.sin(t2) + L3 * math.sin(t2 + t3))
    return x, y, z


def _iter_child_nodes(node) -> List[object]:
    """Node配下の参照ノード(SFNode/MFNode)を雑に列挙する。

    PROTO内の構造は field 名が一定とは限らないため、getNumberOfFields() を使って
    すべてのFieldから SFNode/MFNode を試行的に取り出す。
    """
    out: List[object] = []
    if node is None:
        return out
    try:
        n = int(node.getNumberOfFields())
    except Exception:
        return out

    # Webots field type enum (WbFieldType) の整数値を利用。
    # 参照: SFNODE=7, MFNODE=16 （Webots標準）
    SFNODE = 7
    MFNODE = 16

    for i in range(n):
        try:
            f = node.getFieldByIndex(i)
        except Exception:
            continue

        try:
            ftype = int(f.getType())
        except Exception:
            continue

        if ftype == MFNODE:
            try:
                cnt = int(f.getCount())
            except Exception:
                cnt = 0
            for j in range(cnt):
                try:
                    child = f.getMFNode(j)
                except Exception:
                    child = None
                if child is not None:
                    out.append(child)
            continue

        if ftype == SFNODE:
            try:
                child = f.getSFNode()
                if child is not None:
                    out.append(child)
            except Exception:
                pass

    return out


def _find_solid_by_name(root, solid_name: str):
    """root配下から name が一致する Solid を探索して返す。見つからなければ None。"""
    if root is None or not solid_name:
        return None
    target = str(solid_name)
    stack = [root]
    seen_ids: set[int] = set()
    while stack:
        node = stack.pop()
        try:
            nid = int(node.getId())  # type: ignore[attr-defined]
        except Exception:
            nid = id(node)
        if nid in seen_ids:
            continue
        seen_ids.add(nid)

        # Solid.name を見る
        try:
            name_field = node.getField("name")
        except Exception:
            name_field = None
        if name_field is not None:
            try:
                if str(name_field.getSFString()) == target:
                    return node
            except Exception:
                pass

        stack.extend(_iter_child_nodes(node))
    return None


def _find_named_nodes_containing(root, needle: str, *, limit: int = 30) -> List[str]:
    """デバッグ用: name フィールドを持つノード名を収集する。"""
    if root is None or not needle:
        return []
    needle_l = str(needle).lower()

    stack = [root]
    seen_ids: set[int] = set()
    out: List[str] = []
    while stack and len(out) < limit:
        node = stack.pop()
        try:
            nid = int(node.getId())  # type: ignore[attr-defined]
        except Exception:
            nid = id(node)
        if nid in seen_ids:
            continue
        seen_ids.add(nid)

        try:
            name_field = node.getField("name")
        except Exception:
            name_field = None
        if name_field is not None:
            try:
                nm = str(name_field.getSFString())
                if needle_l in nm.lower():
                    out.append(nm)
            except Exception:
                pass

        stack.extend(_iter_child_nodes(node))
    return out


class DroneCircularController:
    def __init__(self) -> None:
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())
        self._start_time = float(self.supervisor.getTime())
        self._max_runtime_s = float(os.getenv("DRONE_MAX_RUNTIME_S", "180"))

        session_id = f"drone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_id = session_id
        self.pipeline = DiagnosticsPipeline(session_id)

        self.expected_causes = self._load_expected_causes()
        self.pipeline.set_expected_causes(self.expected_causes)

        self.offset_x, self.offset_y, self.offset_z, self.center_def = self._parse_args(sys.argv[1:])

        self.spot_node = self.supervisor.getFromDef("SPOT")
        self.spot_custom = self.spot_node.getField("customData") if self.spot_node else None

        # 各脚の forearm(Solid) を解決（足先近傍リンクの実測位置を得るため）
        self._foot_solid_by_leg: Dict[str, object] = {}
        if self.spot_node is not None:
            try:
                self._foot_solid_by_leg = self._resolve_foot_solids(self.spot_node)
            except Exception as exc:
                try:
                    print(f"[drone_new] warn: resolve_foot_solids failed: {exc}")
                except Exception:
                    pass
                self._foot_solid_by_leg = {}

        self.drone_node = self.supervisor.getSelf()
        self.translation_field = self.drone_node.getField("translation")
        self.rotation_field = self.drone_node.getField("rotation")

        self.center_node = self.supervisor.getFromDef(self.center_def)

        # ワールド上の環境物体（TRAP/VINE/BURIED）から、各脚の環境ヒントを推定する。
        # NOTE: ベンチ環境では対象物が z=-100 に退避している（=非アクティブ）ため、位置で判定できる。
        self._world_env_hint_by_leg: Dict[str, str] = self._infer_world_env_hints()

        self.last_custom_data = ""

        # 重複TRIGGER/SELF_DIAGの抑制（同一文字列が続く場合でも JOINT_ANGLES は処理したい）
        self._seen_trigger: set[tuple[str, int, float]] = set()
        self._seen_self_diag: set[tuple[str, int]] = set()

        # (leg_id, trial_index) -> active state
        self.active: Dict[Tuple[str, int], Dict] = {}
        self.trials_completed: Dict[str, int] = {leg: 0 for leg in diag_config.LEG_IDS}
        self._finalized = False
        self._quit_time: float | None = None
        self._quit_message_printed = False
        self._quit_error_printed = False
        self._snapshot_saved = False
        self._snapshot_path: str | None = None

        self._terminate_requested = False

        self._save_snapshot_enabled = os.getenv("DRONE_SAVE_SNAPSHOT", "0") in {"1", "true", "TRUE", "yes", "YES"}

        self._camera = self._init_camera() if self._save_snapshot_enabled else None

        self._last_center_pos = [0.0, 0.0, 0.0]

        # trial内の初期数フレームは関節角が安定しない/欠損することがあるため、
        # 足先ローカル位置のベースラインを中央値で確定してから end_disp を記録する。
        self._baseline_frames: int = int(os.getenv("DRONE_BASELINE_FRAMES", "5"))
        # JOINT_ANGLES は「度」なので、ジャンプ閾値も度で比較する
        self._max_angle_jump_deg: float = float(os.getenv("DRONE_MAX_ANGLE_JUMP_DEG", "35"))

        # Webotsが終了する際、コントローラにSIGTERMが送られる。
        # ここで重い処理（LLM推論など）を走らせると 1 秒で終わらず Forced termination になりやすい。
        # SIGTERM受信時は「Qwenなしで高速finalize→即終了」を行う。
        try:
            signal.signal(signal.SIGTERM, self._on_sigterm)
        except Exception:
            pass

        print(f"[drone_new] init session={session_id} timestep={self.time_step}ms")
        print(f"[drone_new] expected={self.expected_causes}")
        if self._world_env_hint_by_leg:
            try:
                hints = ", ".join(f"{k}:{v}" for k, v in self._world_env_hint_by_leg.items())
                print(f"[drone_new] world env hints: {hints}")
            except Exception:
                pass
        if self._foot_solid_by_leg:
            resolved = ", ".join(f"{k}:{'ok' if v is not None else 'ng'}" for k, v in self._foot_solid_by_leg.items())
            print(f"[drone_new] foot solids: {resolved}")
            if all(v is None for v in self._foot_solid_by_leg.values()) and self.spot_node is not None:
                # 探索失敗時の手掛かり（ログ量を抑えるため少数のみ）
                hints = _find_named_nodes_containing(self.spot_node, "forearm", limit=12)
                if hints:
                    print(f"[drone_new] debug: found name contains 'forearm': {hints}")
        else:
            print("[drone_new] foot solids: not resolved (fallback to FK)")

        # 位置合わせデバッグ（環境物体が脚に掛かっているかの確認用）
        if os.getenv("DRONE_DEBUG_ENV_POS", "1") in {"1", "true", "TRUE", "yes", "YES"}:
            try:
                self._debug_print_initial_positions()
            except Exception:
                pass


    def _debug_print_initial_positions(self) -> None:
        if self.spot_node is None:
            return
        try:
            spot_pos = list(self.spot_node.getPosition())
        except Exception:
            spot_pos = None

        if spot_pos is not None:
            print(f"[drone_new] debug: spot base pos={spot_pos}")

        # foot solids
        for leg_id in diag_config.LEG_IDS:
            node = self._foot_solid_by_leg.get(leg_id)
            if node is None:
                continue
            try:
                pos = list(node.getPosition())
            except Exception:
                continue
            print(f"[drone_new] debug: foot[{leg_id}] pos={pos}")

        # env objects
        env_defs: Dict[str, Dict[str, str | List[str]]] = {
            "TRAP": {"FL": "FOOT_TRAP_FL", "FR": "FOOT_TRAP_FR", "RL": "FOOT_TRAP_RL", "RR": "FOOT_TRAP_RR"},
            "VINE": {"FL": "FOOT_VINE_FL", "FR": "FOOT_VINE_FR", "RL": "FOOT_VINE_RL", "RR": "FOOT_VINE_RR"},
            "BURIED": {
                "FL": ["BURIED_TOP_FL", "BURIED_LEFT_FL", "BURIED_RIGHT_FL", "BURIED_FRONT_FL", "BURIED_BACK_FL"],
                "FR": ["BURIED_TOP_FR", "BURIED_LEFT_FR", "BURIED_RIGHT_FR", "BURIED_FRONT_FR", "BURIED_BACK_FR"],
                "RL": ["BURIED_TOP_RL", "BURIED_LEFT_RL", "BURIED_RIGHT_RL", "BURIED_FRONT_RL", "BURIED_BACK_RL"],
                "RR": ["BURIED_TOP_RR", "BURIED_LEFT_RR", "BURIED_RIGHT_RR", "BURIED_FRONT_RR", "BURIED_BACK_RR"],
            },
        }

        def _pos(def_name: str) -> Optional[List[float]]:
            try:
                node = self.supervisor.getFromDef(def_name)
            except Exception:
                node = None
            if node is None:
                return None
            try:
                return list(node.getPosition())
            except Exception:
                return None

        for kind, mapping in env_defs.items():
            for leg_id in diag_config.LEG_IDS:
                d = mapping.get(leg_id)
                if isinstance(d, list):
                    # BURIED は代表として TOP を出す（他も必要なら増やす）
                    name = d[0] if d else ""
                    if not name:
                        continue
                    p = _pos(name)
                    if p is None:
                        continue
                    print(f"[drone_new] debug: {kind}[{leg_id}] {name} pos={p}")
                else:
                    name = str(d or "")
                    if not name:
                        continue
                    p = _pos(name)
                    if p is None:
                        continue
                    print(f"[drone_new] debug: {kind}[{leg_id}] {name} pos={p}")

    def _resolve_foot_solids(self, spot_root) -> Dict[str, object]:
        """Spot PROTO内部から、各脚の FOREARM Solid を解決する。

        NOTE:
        - Supervisor API で PROTO インスタンスの内部ノードへは、一般のフィールド走査では辿れない。
          そのため Spot.proto / SpotLeg.proto 内の DEF を getFromProtoDef() で参照する。
        """

        leg_proto_def = {
            "FL": "FRONT_LEFT_LEG",
            "FR": "FRONT_RIGHT_LEG",
            "RL": "REAR_LEFT_LEG",
            "RR": "REAR_RIGHT_LEG",
        }

        out: Dict[str, object] = {}
        for leg_id, leg_def in leg_proto_def.items():
            leg_node = None
            try:
                leg_node = spot_root.getFromProtoDef(leg_def)
            except Exception:
                leg_node = None

            if leg_node is None:
                # 互換: 一部環境では内部 DEF がグローバルに見えることがある
                try:
                    leg_node = self.supervisor.getFromDef(leg_def)
                except Exception:
                    leg_node = None

            forearm_node = None
            if leg_node is not None:
                try:
                    forearm_node = leg_node.getFromProtoDef("FOREARM")
                except Exception:
                    forearm_node = None

            out[leg_id] = forearm_node

        return out

    def _infer_world_env_hints(self) -> Dict[str, str]:
        """ワールド上の環境物体の位置から、各脚の環境ヒントを推定する。

        優先順位: TRAP -> VINE -> BURIED -> NONE
        """

        def _is_active(def_name: str) -> bool:
            if not def_name:
                return False
            try:
                node = self.supervisor.getFromDef(def_name)
            except Exception:
                node = None
            if node is None:
                return False
            try:
                pos = list(node.getPosition())
            except Exception:
                return False
            # 非アクティブ時は z=-100 に退避
            try:
                return float(pos[2]) > -50.0
            except Exception:
                return False

        def _any_active(def_names: list[str]) -> bool:
            for name in def_names:
                if _is_active(name):
                    return True
            return False

        trap_def = {"FL": "FOOT_TRAP_FL", "FR": "FOOT_TRAP_FR", "RL": "FOOT_TRAP_RL", "RR": "FOOT_TRAP_RR"}
        vine_def = {"FL": "FOOT_VINE_FL", "FR": "FOOT_VINE_FR", "RL": "FOOT_VINE_RL", "RR": "FOOT_VINE_RR"}
        # NOTE: set_environment は BURIED_BOTTOM を常に退避させる場合があるため、
        # BURIED_TOP/側面/前後などのいずれかが有効なら BURIED とみなす。
        buried_defs = {
            "FL": ["BURIED_TOP_FL", "BURIED_LEFT_FL", "BURIED_RIGHT_FL", "BURIED_FRONT_FL", "BURIED_BACK_FL"],
            "FR": ["BURIED_TOP_FR", "BURIED_LEFT_FR", "BURIED_RIGHT_FR", "BURIED_FRONT_FR", "BURIED_BACK_FR"],
            "RL": ["BURIED_TOP_RL", "BURIED_LEFT_RL", "BURIED_RIGHT_RL", "BURIED_FRONT_RL", "BURIED_BACK_RL"],
            "RR": ["BURIED_TOP_RR", "BURIED_LEFT_RR", "BURIED_RIGHT_RR", "BURIED_FRONT_RR", "BURIED_BACK_RR"],
        }

        out: Dict[str, str] = {}
        for leg_id in diag_config.LEG_IDS:
            if _is_active(trap_def.get(leg_id, "")):
                out[leg_id] = "TRAPPED"
            elif _is_active(vine_def.get(leg_id, "")):
                out[leg_id] = "TANGLED"
            elif _any_active(buried_defs.get(leg_id, [])):
                out[leg_id] = "BURIED"
            else:
                out[leg_id] = "NONE"
        return out

    def _on_sigterm(self, signum, frame) -> None:  # type: ignore[no-untyped-def]
        if self._terminate_requested:
            return
        self._terminate_requested = True
        # 終了時はQwen推論を無効化して素早くfinalizeできるようにする
        os.environ["QWEN_ENABLE"] = "0"
        # ログはなるべく残す（Webotsが即killする可能性があるので軽量に）
        try:
            print("[drone_new] SIGTERM received -> fast finalize (Qwen disabled)")
        except Exception:
            pass

    def _init_camera(self):
        # Mavic2Pro.proto の cameraSlot にはデフォルトで Camera が入っている。
        # name 未指定のため、多くの環境では "camera" で取得できる。
        candidate_names = ["camera", "Camera", "front camera", "camera0"]
        for name in candidate_names:
            try:
                cam = self.supervisor.getDevice(name)
                if cam is None:
                    continue
                try:
                    cam.enable(self.time_step)
                except Exception:
                    pass
                print(f"[drone_new] camera device found: {name}")
                return cam
            except Exception:
                continue

        print("[drone_new] camera device not found (snapshot will be skipped)")
        return None

    def _save_snapshot_once(self) -> str | None:
        if not self._save_snapshot_enabled:
            return None
        if self._snapshot_saved:
            return self._snapshot_path
        self._snapshot_saved = True

        if self._camera is None:
            return None

        # 画像は controller 配下の logs/images に保存し、webots_new ルートから辿れる相対パスを記録する。
        root_dir = HERE.parent.parent
        images_dir = HERE / "logs" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        abs_path = images_dir / f"{self.session_id}.png"
        rel_path = abs_path.relative_to(root_dir)
        try:
            # quality は JPEG 用だが PNG でも引数が必要。
            self._camera.saveImage(str(abs_path), 100)
            self._snapshot_path = str(rel_path)
            print(f"[drone_new] saved snapshot: {self._snapshot_path}")
            return self._snapshot_path
        except Exception as exc:
            print(f"[drone_new] warn: snapshot failed: {exc}")
            return None

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
        if not raw:
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

                    trial_angle_deg_effective = None
                    if len(parts) > 6:
                        try:
                            trial_angle_deg_effective = float(parts[6])
                        except Exception:
                            trial_angle_deg_effective = None

                    key = (leg_id, trial_index, start_time)
                    if key in self._seen_trigger:
                        continue
                    self._seen_trigger.add(key)

                    end_time = start_time + duration_ms / 1000.0

                    self.pipeline.start_trial(
                        leg_id,
                        trial_index,
                        direction,
                        start_time,
                        duration_ms / 1000.0,
                        trial_angle_deg_effective=trial_angle_deg_effective,
                    )
                    self.active[(leg_id, trial_index)] = {
                        "trial_index": trial_index,
                        "direction": direction,
                        "start_time": start_time,
                        "end_time": end_time,
                        "trial_angle_deg_effective": trial_angle_deg_effective,
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

                    # 角度の急激なジャンプを除外（センサ欠損/初期化揺れ対策）
                    last_angles = state.get("last_angles")
                    if isinstance(last_angles, list) and len(last_angles) == 3:
                        try:
                            max_jump = max(abs(float(angles[i]) - float(last_angles[i])) for i in range(3))
                        except Exception:
                            max_jump = 0.0
                        if max_jump > self._max_angle_jump_deg:
                            # このフレームは破棄（軌跡長の破綻を防ぐ）
                            continue
                    state["last_angles"] = list(angles)

                    if not self.spot_node:
                        continue

                    base_pos = list(self.spot_node.getPosition())

                    def _mat3_t_mul_vec3(m: list[float], v: list[float]) -> list[float]:
                        # Webots getOrientation() は 3x3 行列（row-major）。local->world を与えるので、
                        # world->local は転置を用いる。
                        return [
                            m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
                            m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
                            m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
                        ]

                    # 姿勢（ログ用）
                    ori = list(self.spot_node.getOrientation())
                    roll, pitch, yaw = _rpy_rad_from_orientation_matrix(ori)
                    rpy_deg = (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

                    # 足先近傍リンク（forearm Solid）のワールド位置を優先
                    foot_world = None
                    foot_is_world = False
                    try:
                        foot_node = self._foot_solid_by_leg.get(leg_id)
                        if foot_node is not None:
                            foot_world = list(foot_node.getPosition())
                            foot_is_world = True
                    except Exception:
                        foot_world = None
                        foot_is_world = False

                    # フォールバック（FK）: 既存の近似（胴体ローカル）
                    if foot_world is None or len(foot_world) < 3:
                        x_local, y_local, z_local = _fk_foot_local(leg_id, angles)
                        foot_world = [float(x_local), float(y_local), float(z_local)]
                        foot_is_world = False

                    # end_positions は「初期足先位置からの変位ベクトル」系列。
                    # 以前は base の平行移動・回転を毎フレーム除去して胴体ローカル差分を作っていたが、
                    # base姿勢が大きく変わる（転倒/傾き）と座標系が揺れて end_disp が過大になり、
                    # TRAPPED/BURIED でも drone_can が高止まりする原因になった。
                    # ここでは Foot Solid が取れている場合は foot(world) の変位をそのまま使い、
                    # 拘束時に「足先が世界座標で動いていない」ことが can に反映されるようにする。

                    if foot_is_world:
                        foot_disp_src = [float(foot_world[0]), float(foot_world[1]), float(foot_world[2])]
                    else:
                        # FKフォールバック時は従来通り（胴体ローカル近似）
                        foot_disp_src = [float(foot_world[0]), float(foot_world[1]), float(foot_world[2])]

                    # ベースライン（中央値）を確定してから end_disp を記録する
                    if state.get("init_foot") is None:
                        samples = state.get("foot_samples")
                        if not isinstance(samples, list):
                            samples = []
                            state["foot_samples"] = samples
                        samples.append((float(foot_disp_src[0]), float(foot_disp_src[1]), float(foot_disp_src[2])))
                        if len(samples) < max(1, self._baseline_frames):
                            continue

                        xs = sorted(p[0] for p in samples)
                        ys = sorted(p[1] for p in samples)
                        zs = sorted(p[2] for p in samples)
                        mid = len(samples) // 2
                        state["init_foot"] = (xs[mid], ys[mid], zs[mid])
                        try:
                            del state["foot_samples"]
                        except Exception:
                            pass

                    init_foot = state.get("init_foot")
                    if not isinstance(init_foot, tuple) or len(init_foot) != 3:
                        continue
                    init_x, init_y, init_z = init_foot
                    end_disp = [float(foot_disp_src[0]) - init_x, float(foot_disp_src[1]) - init_y, float(foot_disp_src[2]) - init_z]

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

                    # 1試行につき1回だけ受け付ける（同一customDataが複数ステップ続いても二重計上しない）
                    key = (leg_id, trial_index)
                    if key in self._seen_self_diag:
                        continue
                    self._seen_self_diag.add(key)

                    # tau / malfunction_flag は MALFUNCTION の切り分けに使う
                    tau_avg = float(parts[6]) if len(parts) > 6 else 0.0
                    tau_max = float(parts[7]) if len(parts) > 7 else 0.0
                    tau_nominal = float(parts[8]) if len(parts) > 8 else 0.0
                    safety = parts[9] if len(parts) > 9 else "UNKNOWN"
                    self_can_raw = float(parts[10]) if len(parts) > 10 else 0.35
                    malfunction_flag = int(parts[11]) if len(parts) > 11 else 0

                    state = self.active.get((leg_id, trial_index))
                    if not state:
                        continue

                    state["self_diag"] = {
                        "tau_nominal": tau_nominal,
                        "tau_avg": tau_avg,
                        "tau_max": tau_max,
                        "safety": safety,
                        "self_can_raw": self_can_raw,
                        "malfunction_flag": malfunction_flag,
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

            self_diag = state.get("self_diag") or {
                "tau_nominal": 0.0,
                "tau_avg": 0.0,
                "tau_max": 0.0,
                "safety": "UNKNOWN",
                "self_can_raw": None,
                "malfunction_flag": 0,
            }

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
                spot_tau_avg=float(self_diag.get("tau_avg") or 0.0),
                spot_tau_max=float(self_diag.get("tau_max") or 0.0),
                spot_malfunction_flag=int(self_diag.get("malfunction_flag") or 0),
                trial_angle_deg_effective=state.get("trial_angle_deg_effective"),
                world_env_hint=self._world_env_hint_by_leg.get(leg_id),
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
                snap = self._save_snapshot_once()
                if snap:
                    try:
                        # セッションログに載せる（外部処理が参照できるように保持）
                        self.pipeline.session.image_path = snap
                    except Exception:
                        pass
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
            # 終了要求が来たら、できるだけ早くログを残して終了する
            if self._terminate_requested and not self._finalized:
                try:
                    self.pipeline.finalize()
                except Exception:
                    pass
                self._finalized = True
                break

            self.process_messages()
            self._maybe_complete_trials()
            self.update_position()

            # フェイルセーフ: 何らかの理由で試行が完了しない/quitできない場合でも終了させる
            now = float(self.supervisor.getTime())
            if (now - self._start_time) > self._max_runtime_s and not self._finalized:
                print(f"[drone_new] watchdog: runtime>{self._max_runtime_s:.0f}s -> finalize+quit")
                snap = self._save_snapshot_once()
                if snap:
                    try:
                        self.pipeline.session.image_path = snap
                    except Exception:
                        pass
                try:
                    self.pipeline.finalize()
                except Exception:
                    pass
                self._finalized = True
                self._quit_time = now + 0.5

            if self._finalized and self._quit_time is not None:
                if self.supervisor.getTime() >= self._quit_time:
                    if not self._quit_message_printed:
                        print("[drone_new] quit simulation")
                        self._quit_message_printed = True
                    try:
                        self.supervisor.simulationQuit(0)
                    except Exception as exc:
                        if not self._quit_error_printed:
                            print(f"[drone_new] warn: simulationQuit failed: {exc}")
                            self._quit_error_printed = True

        # step() が -1 で抜けるケースでも、ログ未出力を避けるため軽量finalizeを試みる
        if not self._finalized:
            try:
                os.environ.setdefault("QWEN_ENABLE", "0")
                self.pipeline.finalize()
            except Exception:
                pass


def main() -> None:
    DroneCircularController().run()


if __name__ == "__main__":
    main()
