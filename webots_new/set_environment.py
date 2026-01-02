#!/usr/bin/env python3
"""環境設定スクリプト（webots_new 簡潔版）

やること:
- webots_new/config/scenario.ini を更新する（脚ごとの環境）
- webots_new/worlds/sotuken_world.wbt の環境オブジェクトを表示/非表示にする

使い方:
  対話モード:
    python set_environment.py

  コマンドライン（脚の順序: FL FR RL RR）:
    python set_environment.py NONE BURIED TRAPPED NONE

設定できる環境:
  NONE, BURIED, TRAPPED, TANGLED, MALFUNCTION
"""

import configparser
import shutil
import sys
from datetime import datetime
from pathlib import Path

VALID_ENV = ["NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION"]
LEG_IDS = ["FL", "FR", "RL", "RR"]
LEG_NAMES = {
    "FL": "front_left",
    "FR": "front_right",
    "RL": "rear_left",
    "RR": "rear_right",
}

# ギミックの設置中心座標（ワールド座標）
# 既存の FL 用座標 (0.45, 0.17) を基準に、左右は Y 反転・後脚は X を負側へ。
# ※ワールド/Spotの向きが変わる場合はここを調整してください。
LEG_GIMMICK_CENTER_XY: dict[str, tuple[float, float]] = {
    "FL": (0.45, 0.17),
    "FR": (0.45, -0.17),
    "RL": (-0.35, 0.17),
    "RR": (-0.35, -0.17),
}

# TRAPPED(FOOT_TRAP) は形状の都合で微妙に中心からズレることがあるため、脚ごとに微調整できるようにする。
TRAP_OFFSET_X: dict[str, float] = {
    "FL": 0.00,
    "FR": 0.03,
    "RL": 0.03,
    "RR": 0.03,
}

# TANGLED(FOOT_VINE) は「脚の真下」を外すとほぼ効かないため、まずは脚中心と同じ基準に揃える。
# ここから微調整する場合は VINE_OFFSET_* を触る。
VINE_CENTER_XY: dict[str, tuple[float, float]] = {
    "FL": LEG_GIMMICK_CENTER_XY["FL"],
    "FR": LEG_GIMMICK_CENTER_XY["FR"],
    "RL": LEG_GIMMICK_CENTER_XY["RL"],
    "RR": LEG_GIMMICK_CENTER_XY["RR"],
}

VINE_OFFSET_X: dict[str, float] = {
    "FL": 0.00,
    "FR": 0.00,
    "RL": 0.05,
    "RR": 0.05,
}
VINE_OFFSET_Y: dict[str, float] = {
    "FL": 0.00,
    "FR": 0.00,
    "RL": 0.01,
    "RR": -0.01,
}
# VINEは 0.0 だと地面に埋まって接触が弱いことがあるので、少し浮かせる。
VINE_Z = 0.08

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "scenario.ini"
WORLD_PATH = ROOT / "worlds" / "sotuken_world.wbt"


def _normalize_env(text: str) -> str:
    """文字列を環境ラベルに正規化（不正なら None）。"""
    env = (text or "").strip().upper()
    if env in VALID_ENV:
        return env
    return ""


def _ask_env_for_leg(leg_id: str) -> str:
    """対話モードで環境を入力してもらう。"""
    while True:
        raw = input(f"{leg_id} ({LEG_NAMES[leg_id]}) の環境 (デフォルト: NONE): ").strip()
        if raw == "":
            raw = "NONE"
        env = _normalize_env(raw)
        if env:
            return env
        print(f"エラー: '{raw}' は無効です。{', '.join(VALID_ENV)} から選んでください。")


def read_args() -> list[str]:
    """引数 or 対話モードで FL/FR/RL/RR の環境を返す。"""
    if len(sys.argv) == 1:
        print("=" * 60)
        print("環境設定スクリプト（対話モード）")
        print("=" * 60)
        print(f"有効な環境: {', '.join(VALID_ENV)}")
        envs = []
        for leg_id in LEG_IDS:
            envs.append(_ask_env_for_leg(leg_id))
        return envs

    if len(sys.argv) != 5:
        print("使い方:")
        print("  対話モード:     python set_environment.py")
        print("  コマンドライン: python set_environment.py FL FR RL RR")
        print("例:")
        print("  python set_environment.py NONE BURIED TRAPPED NONE")
        print(f"有効な環境: {', '.join(VALID_ENV)}")
        print(f"脚の順序: {' '.join(LEG_IDS)}")
        sys.exit(1)

    envs = []
    for raw in sys.argv[1:]:
        env = _normalize_env(raw)
        if not env:
            print(f"エラー: '{raw}' は無効です。{', '.join(VALID_ENV)} から選んでください。")
            sys.exit(1)
        envs.append(env)
    return envs


def update_scenario_ini(envs: list[str]) -> None:
    """scenario.ini を更新する（物理パラメータは残す）。"""
    old = configparser.ConfigParser()
    if CONFIG_PATH.exists():
        old.read(CONFIG_PATH)

    cfg = configparser.ConfigParser()
    cfg["DEFAULT"] = {}

    # 既存の物理パラメータはなるべく引き継ぐ（無いものはデフォルト）
    keep_keys = [
        "toplevel",
        "friction",
        "bounce",
        "material",
        "sand.radius",
        "sand.height",
        "sand.color",
        "trap.offsetx",
        "trap.offsety",
        "trap.offsetz",
        "trap.friction",
        "trap.bounce",
        "trap.material",
        "vine.offsetx",
        "vine.offsety",
        "vine.offsetz",
        "vine.rotation",
        "vine.friction",
        "vine.bounce",
        "vine.material",
    ]

    if "DEFAULT" in old:
        for key in keep_keys:
            if key in old["DEFAULT"]:
                cfg["DEFAULT"][key] = old["DEFAULT"][key]

    # 最低限のデフォルト
    cfg["DEFAULT"].setdefault("scenario", "none")
    cfg["DEFAULT"].setdefault("buriedFoot", "front_left")

    # 脚ごとの環境（新方式）
    for leg_id, env in zip(LEG_IDS, envs):
        cfg["DEFAULT"][f"{leg_id.lower()}_environment"] = env

    # 旧方式（後方互換）: 最初に見つかった 1つだけを scenario として書く
    scenario = "none"
    for leg_id, env in zip(LEG_IDS, envs):
        leg_name = LEG_NAMES[leg_id]
        if env == "BURIED" and scenario == "none":
            scenario = "sand_burial"
            cfg["DEFAULT"]["buriedFoot"] = leg_name
        if env == "TRAPPED" and scenario == "none":
            scenario = "foot_trap"
            cfg["DEFAULT"]["trappedFoot"] = leg_name
        if env == "TANGLED" and scenario == "none":
            scenario = "foot_vine"
            cfg["DEFAULT"]["tangledFoot"] = leg_name
    cfg["DEFAULT"]["scenario"] = scenario

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        cfg.write(f)

    print(f"scenario.ini 更新: {CONFIG_PATH}")
    for leg_id, env in zip(LEG_IDS, envs):
        print(f"  {leg_id} ({LEG_NAMES[leg_id]}): {env}")


def _set_translation(lines: list[str], def_name: str, new_xyz: tuple[float, float, float]) -> bool:
    """.wbt の DEF ノード配下にある translation 行を探して置換する（簡単な行ベース）。"""
    x, y, z = new_xyz
    for i, line in enumerate(lines):
        if f"DEF {def_name} " not in line:
            continue
        # 近い範囲で translation 行を探す
        for j in range(i, min(i + 25, len(lines))):
            if "translation" in lines[j]:
                indent = lines[j].split("translation")[0]
                lines[j] = f"{indent}translation {x} {y} {z}\n"
                return True
        return False
    return False


def update_world_wbt(envs: list[str]) -> bool:
    """worlds/sotuken_world.wbt の環境オブジェクトを表示/非表示にする。"""
    if not WORLD_PATH.exists():
        print(f"警告: ワールドが見つかりません: {WORLD_PATH}")
        return False

    # バックアップ
    backup = WORLD_PATH.parent / f"{WORLD_PATH.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wbt"
    shutil.copy2(WORLD_PATH, backup)
    print(f"バックアップ作成: {backup.name}")

    lines = WORLD_PATH.read_text(encoding="utf-8").splitlines(True)

    # 仕様: envs は脚ごとに完全に適用する
    # ワールド側に脚ごとのギミックDEF（例: FOOT_TRAP_FL）がある前提。

    # 注意: BURIED_BOX は Group ノードで translation を持たない。
    # ここでは親を動かさず、子 Solid の translation だけを切り替える。

    # BURIED_BOX の子（脚ごとに表示/非表示）
    # 既存配置を「中心 (cx, cy) とオフセット」で表現
    def _buried_child_positions(
        center_xy: tuple[float, float],
        leg_id: str,
    ) -> dict[str, tuple[float, float, float]]:
        cx, cy = center_xy
        side = 0.09
        suffix = f"_{leg_id}"
        return {
            f"BURIED_BOTTOM{suffix}": (cx, cy, 0.0),
            f"BURIED_TOP{suffix}": (cx, cy, 0.12),
            f"BURIED_LEFT{suffix}": (cx - side, cy, 0.08),
            f"BURIED_RIGHT{suffix}": (cx + side, cy, 0.08),
            f"BURIED_FRONT{suffix}": (cx, cy + side, 0.08),
            f"BURIED_BACK{suffix}": (cx, cy - side, 0.08),
        }

    for leg_id, env in zip(LEG_IDS, envs):
        buried_children = _buried_child_positions(LEG_GIMMICK_CENTER_XY[leg_id], leg_id)
        if env == "BURIED":
            for name, pos in buried_children.items():
                _set_translation(lines, name, pos)
            print(f"BURIED_BOX: 表示 ({leg_id})")
        else:
            for name in buried_children:
                _set_translation(lines, name, (0.0, 0.0, -100.0))

    # FOOT_TRAP（脚ごと）
    for leg_id, env in zip(LEG_IDS, envs):
        def_name = f"FOOT_TRAP_{leg_id}"
        if env == "TRAPPED":
            cx, cy = LEG_GIMMICK_CENTER_XY[leg_id]
            _set_translation(lines, def_name, (cx + TRAP_OFFSET_X.get(leg_id, 0.03), cy, 0.0))
            print(f"FOOT_TRAP: 表示 ({leg_id})")
        else:
            _set_translation(lines, def_name, (0.0, 0.0, -100.0))

    # FOOT_VINE（脚ごと）
    for leg_id, env in zip(LEG_IDS, envs):
        def_name = f"FOOT_VINE_{leg_id}"
        if env == "TANGLED":
            cx, cy = VINE_CENTER_XY.get(leg_id, LEG_GIMMICK_CENTER_XY[leg_id])
            _set_translation(
                lines,
                def_name,
                (
                    cx + VINE_OFFSET_X.get(leg_id, 0.0),
                    cy + VINE_OFFSET_Y.get(leg_id, 0.0),
                    VINE_Z,
                ),
            )
            print(f"FOOT_VINE: 表示 ({leg_id})")
        else:
            _set_translation(lines, def_name, (0.0, 0.0, -100.0))

    WORLD_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"ワールド更新: {WORLD_PATH}")
    return True


def main() -> None:
    envs = read_args()

    print("\n" + "=" * 70)
    print("環境設定の適用")
    print("=" * 70)

    print("\n[1/2] scenario.ini を更新中...")
    update_scenario_ini(envs)

    print("\n[2/2] ワールドファイルを更新中...")
    ok = update_world_wbt(envs)

    if ok:
        print("\n" + "=" * 70)
        print("環境設定が完了しました")
        print("=" * 70)
        print("Webotsが起動中なら、いったん終了→再起動してください。")
    else:
        print("\nワールド更新に失敗しました（scenario.ini は更新済みです）")


if __name__ == "__main__":
    main()
