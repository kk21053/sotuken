#!/usr/bin/env python3
"""診断結果表示スクリプト（webots_new 簡潔版）

- drone コントローラが出力する `logs/leg_diagnostics_sessions.jsonl` の最新行を読み、見やすく表示します。
"""

import json
import sys
from pathlib import Path

# drone controller のログを読む
LOG_DIR = Path(__file__).resolve().parent / "controllers" / "drone_circular_controller" / "logs"
SESSION_LOG = LOG_DIR / "leg_diagnostics_sessions.jsonl"


def load_last_json(path: Path):
    if not path.exists():
        print(f"エラー: ログファイルが見つかりません: {path}")
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        print("エラー: ログファイルが空です")
        return None

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        print(f"エラー: JSONの解析に失敗しました: {e}")
        return None


def jp_cause(label: str) -> str:
    table = {
        "NONE": "正常",
        "BURIED": "埋まる",
        "TRAPPED": "挟まる",
        "TANGLED": "絡まる",
        "MALFUNCTION": "故障",
        "FALLEN": "転倒",
    }
    return table.get(label, str(label))


def show(result: dict) -> None:
    print("\n" + "=" * 80)
    print("診断結果")
    print("=" * 80)
    print(f"セッションID: {result.get('session_id', 'N/A')}")
    print(f"転倒:       {result.get('fallen', False)} (確率: {float(result.get('fallen_probability', 0.0)):.3f})")
    print("-")

    legs = result.get("legs", {})

    print("-" * 80)
    print(f"{'脚':^4} | {'判定':^10} | {'p_can':^6} | {'原因':^8} | {'期待値':^8}")
    print("-" * 80)

    correct = 0
    total = 0

    for leg_id in ["FL", "FR", "RL", "RR"]:
        if leg_id not in legs:
            print(f"{leg_id:^4} | {'データなし':^10} | {'-':^6} | {'-':^8} | {'-':^8}")
            continue

        leg = legs[leg_id]
        movement = leg.get("movement_result", "-")
        p_can = float(leg.get("p_can", 0.0))
        cause_final = leg.get("cause_final", "-")
        expected = leg.get("expected_cause", "-")
        fallen = bool(leg.get("fallen", False))

        # 旧view_result.pyと同じ表示ルール（転倒で expected=NONE のとき）
        expected_text = "正常/転倒" if fallen and expected == "NONE" else jp_cause(expected)

        print(
            f"{leg_id:^4} | {movement:^10} | {p_can:^6.3f} | {jp_cause(cause_final):^8} | {expected_text:^8}"
        )

        # 正解率（旧ロジック準拠）
        if cause_final == expected:
            correct += 1
            total += 1
        elif fallen and expected == "NONE" and cause_final == "FALLEN":
            correct += 1
            total += 1
        else:
            total += 1

    print("-" * 80)

    if total > 0:
        acc = correct / total * 100
        print(f"診断精度: {correct}/{total} ({acc:.1f}%)")

    print("=" * 80)


def main() -> None:
    result = load_last_json(SESSION_LOG)
    if result is None:
        sys.exit(1)
    show(result)


if __name__ == "__main__":
    main()
