# Webots起動エラー修正報告 (第3弾)

## エラー: LegState に conf_final 属性が存在しない

### エラーメッセージ
```
AttributeError: 'LegState' object has no attribute 'conf_final'. Did you mean: 'cause_final'?
```

### 発生箇所
`drone_circular_controller.py` line 559

### 原因
`finalize_diagnosis()`が削除済みの古いフィールドを参照していた：
- `conf_final` - 削除済み（信頼度は不要）
- `p_final` - 削除済み（`p_llm`に統一）
- `leg_state.trials` - `LegStatus`には存在しない（`LegState`のみ）

### LegStatus の実際のフィールド
```python
@dataclass
class LegStatus:
    """結果表示用の脚の状態スナップショット"""
    leg_id: str
    spot_can: float          # 仕様ステップ3
    drone_can: float         # 仕様ステップ4
    p_drone: Dict[str, float]  # ドローンの確率分布
    p_llm: Dict[str, float]    # LLMの確率分布（仕様ステップ7）
    movement_result: str      # "動く" | "動かない" | "一部動く"
    cause_final: str         # 最終拘束原因
    p_can: float            # 最終動作確率
```

### 修正内容

**drone_circular_controller.py** (lines 547-567)

```python
# 修正前
print(f"  Trials completed: {len(leg_state.trials)}/{diag_config.TRIAL_COUNT}")

print(f"  Drone observation:")
print(f"    Can-move probability: {leg_state.drone_can:.3f}")
print(f"    Cause: {leg_state.cause_final} ({leg_state.conf_final:.1%} confidence)")

print(f"  Cause distribution:")
for cause, prob in leg_state.p_final.items():
    bar = "█" * int(prob * 40)
    print(f"    {cause:12s}: {prob:.3f} {bar}")

# 修正後
# 試行数表示は削除（LegStatusに trials フィールドなし）

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
```

### 修正理由

1. **conf_final 削除**
   - 仕様に「信頼度」の要求なし
   - `cause_final`（拘束原因）と`p_can`（動作確率）があれば十分

2. **p_final → p_llm に統一**
   - 仕様ステップ7でLLMが返す確率分布は`p_llm`
   - `p_final`は削除済み

3. **trials フィールド削除**
   - `LegStatus`はスナップショット（結果表示用）
   - `trials`は`LegState`（内部状態）にのみ存在
   - 試行数表示は仕様に記載なし

4. **表示内容を仕様準拠に改善**
   - `spot_can` - 仕様ステップ3
   - `drone_can` - 仕様ステップ4
   - `movement_result` - 仕様ステップ7,9
   - `cause_final` - 仕様ステップ7,9
   - `p_can` - 仕様ステップ7
   - `p_llm` - 仕様ステップ7（LLM判定結果）

### 新しい出力形式（仕様準拠）

```
[FL] Diagnosis Summary:
  Spot self-diagnosis:
    spot_can (Can-move): 0.850
  Drone observation:
    drone_can (Can-move): 0.999
  Final diagnosis:
    Movement: 動く
    Cause: NONE
    p_can: 0.924
  LLM probability distribution:
    NONE        : 0.950 ████████████████████████████████████████
    BURIED      : 0.010 
    TRAPPED     : 0.010 
    TANGLED     : 0.010 
    MALFUNCTION : 0.020 █
```

---

## テスト結果
✅ **test_system.py**: 4/4テストケース合格

---

## まとめ

### 削除した古いフィールド参照
- ❌ `conf_final` - 存在しない
- ❌ `p_final` - 存在しない
- ❌ `leg_state.trials` - LegStatusには存在しない

### 正しいフィールド
- ✅ `spot_can` - Spot自己診断結果
- ✅ `drone_can` - ドローン観測結果
- ✅ `p_llm` - LLM確率分布
- ✅ `movement_result` - 動作判定
- ✅ `cause_final` - 最終拘束原因
- ✅ `p_can` - 最終動作確率

---

## 次のステップ
🚀 Webotsを再起動して最終診断結果の表示を確認
