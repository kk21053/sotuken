"""LLM-based analysis combining self-diagnosis and drone observation using Ollama."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from . import config
from .models import LegState

# Ollamaクライアントをインポート
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[llm] Warning: ollama package not found. Install with: uv pip install ollama")


CAUSE_DEFINITIONS = {
    "NONE": "脚は正常に動作しており障害がない状態",
    "BURIED": "脚が地面や砂に埋まっており大きく持ち上げられない状態",
    "TRAPPED": "関節は動くのに末端が障害物等に固定され前進できない状態",
    "TANGLED": "ツタなどに絡まり小さい往復でしか動けない状態",
    "MALFUNCTION": "センサー故障または測定エラー（物理的に矛盾する状態）",
    "FALLEN": "ロボットが転倒しており正常な診断が困難な状態",
}


class LLMAnalyzer:
    """
    LLM-based analyzer using Ollama (Llama 3.2).
    
    ローカルLLMを使用した診断分析器:
    - Ollama経由でLlama 3.2 1Bモデルを使用
    - 仕様のルールをプロンプトで指示
    - フォールバック: LLM利用不可時はルールベースに切り替え
    
    Processing time: ~100-500ms per leg (GPU使用時)
    Memory usage: ~2GB (モデルロード時)
    """

    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        use_llm: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        self._model_name = model_name
        self._use_llm = use_llm and OLLAMA_AVAILABLE
        self._max_tokens = max_new_tokens
        
        if self._use_llm:
            print(f"[llm] Using Ollama with model: {self._model_name}")
            # Ollamaサーバーが起動しているか確認
            try:
                ollama.list()
                print("[llm] Ollama server is running")
            except Exception as e:
                print(f"[llm] Warning: Ollama server not accessible: {e}")
                print("[llm] Falling back to rule-based inference")
                self._use_llm = False
        else:
            print("[llm] Using rule-based inference (fallback mode)")

    def infer(self, leg: LegState, all_legs=None, trial_direction=None) -> Dict[str, float]:
        """
        LLM-based or rule-based inference for leg constraint diagnosis.
        
        仕様のルール:
        ①両方が0.7以上 → 動く、拘束原因=正常
        ②両方が0.3以下 → 動かない、拘束原因=確率分布の最大値
        ③片方が0.7以上、もう片方が0.3以下 → 動かない、拘束原因=故障
        ④片方が中間値 → 一部動く、拘束原因=確率分布の最大値
        
        Args:
            leg: LegState object with spot_can, drone_can, and p_drone
            all_legs: Not used (kept for API compatibility)
            trial_direction: Not used (kept for API compatibility)
        
        Returns:
            Dictionary of probabilities for each cause label
        """
        if self._use_llm:
            try:
                return self._infer_with_llm(leg)
            except Exception as e:
                print(f"[llm] LLM inference failed: {e}, falling back to rules")
                return self._infer_with_rules(leg)
        else:
            return self._infer_with_rules(leg)
    
    def _infer_with_llm(self, leg: LegState) -> Dict[str, float]:
        """LLMを使用した推論（Few-shot learning版）"""
        spot_can = leg.spot_can
        drone_can = leg.drone_can
        p_drone = dict(leg.p_drone)
        
        # プロンプトを構築
        user_prompt = self._build_prompt(leg.leg_id, spot_can, drone_can, p_drone)
        
        # Few-shot examples（診断例を提示して学習させる）
        examples = [
            # 例1: ルール1（両方高い）
            {
                'role': 'user',
                'content': self._build_example_prompt("FL", 0.85, 0.90, "NONE", "BURIED")
            },
            {
                'role': 'assistant',
                'content': '{"movement_result": "動く", "cause_final": "NONE", "reasoning": "ルール1: Spot評価0.850≥0.7 かつ ドローン評価0.900≥0.7のため正常動作"}'
            },
            # 例2: ルール2（両方低い）
            {
                'role': 'user',
                'content': self._build_example_prompt("FR", 0.15, 0.20, "BURIED", "BURIED")
            },
            {
                'role': 'assistant',
                'content': '{"movement_result": "動かない", "cause_final": "BURIED", "reasoning": "ルール2: Spot評価0.150≤0.3 かつ ドローン評価0.200≤0.3のため動かない。ドローン推定の最大値はBURIED"}'
            },
            # 例3: ルール3（矛盾 - Spot高・Drone低）
            {
                'role': 'user',
                'content': self._build_example_prompt("RL", 0.75, 0.25, "TRAPPED", "TRAPPED")
            },
            {
                'role': 'assistant',
                'content': '{"movement_result": "動かない", "cause_final": "MALFUNCTION", "reasoning": "ルール3: Spot評価0.750≥0.7 かつ ドローン評価0.250≤0.3でセンサー矛盾。故障の可能性"}'
            },
            # 例4: ルール4（中間）
            {
                'role': 'user',
                'content': self._build_example_prompt("RR", 0.50, 0.45, "TRAPPED", "TRAPPED")
            },
            {
                'role': 'assistant',
                'content': '{"movement_result": "一部動く", "cause_final": "TRAPPED", "reasoning": "ルール4: Spot評価0.500とドローン評価0.450は中間値。一部動く状態。ドローン推定の最大値はTRAPPED"}'
            },
        ]
        
        # メッセージを構築（system + examples + 実際の質問）
        messages = [
            {
                'role': 'system',
                'content': 'あなたは4足歩行ロボットの故障診断AIです。与えられたルールに厳密に従って診断してください。以下の例を参考にしてください。'
            }
        ] + examples + [
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
        
        # LLMに問い合わせ
        try:
            response = ollama.chat(
                model=self._model_name,
                messages=messages,
                format='json',  # JSON出力を強制
                options={
                    'temperature': 0.0,  # 決定論的な出力
                    'num_predict': 128,
                }
            )
        except Exception as e:
            print(f"[llm] Ollama API error: {e}")
            raise
        
        # レスポンスをパース
        result = self._parse_llm_response(response['message']['content'], leg, p_drone)
        
        print(f"[llm] LLM診断: {leg.leg_id} → {leg.movement_result}, 原因={leg.cause_final}")
        
        return result
    
    def _build_example_prompt(self, leg_id: str, spot_can: float, drone_can: float, max_cause: str, max_cause_no_none: str = None) -> str:
        """Few-shot用のプロンプト（本番と同じ形式）"""
        if max_cause_no_none is None:
            max_cause_no_none = max_cause if max_cause != "NONE" else "BURIED"
        
        # 簡易版の確率分布（例示用）
        p_drone_text = f"   {max_cause}: 0.850\n   NONE: 0.100\n   その他: 0.050"
        
        return f"""あなたは4足歩行ロボットの故障診断AIです。以下のデータを分析し、診断ルールに従って判定してください。

【診断対象】
脚ID: {leg_id}

【測定データ】
• Spot内部センサー評価: {spot_can:.3f} (範囲: 0.0～1.0、1.0が完全正常)
• ドローン外部観測評価: {drone_can:.3f} (範囲: 0.0～1.0、1.0が完全正常)

【ドローンの拘束原因推定】(確率分布の上位)
{p_drone_text}

【診断ルール - 以下に厳密に従うこと】
ルール1: Spot評価 ≥ 0.7 かつ ドローン評価 ≥ 0.7
  → 判定: "動く", 原因: "NONE"
  
ルール2: Spot評価 ≤ 0.3 かつ ドローン評価 ≤ 0.3
  → 判定: "動かない", 原因: ドローン推定の最大値 (NONE以外から選択)
  → この場合の候補: {max_cause_no_none}
  
ルール3: (Spot評価 ≥ 0.7 かつ ドローン評価 ≤ 0.3) または (Spot評価 ≤ 0.3 かつ ドローン評価 ≥ 0.7)
  → 判定: "動かない", 原因: "MALFUNCTION" (センサー矛盾)
  
ルール4: 上記以外の中間値
  → 判定: "一部動く", 原因: ドローン推定の最大値
  → この場合の候補: {max_cause}

【タスク】
1. 測定データから適用すべきルールを判定
2. そのルールに従って movement_result と cause_final を決定
3. 判定理由を reasoning に記述

【出力形式】
以下のJSON形式で出力してください（他の説明は不要、JSONのみ）:
{{
  "movement_result": "動く" または "動かない" または "一部動く",
  "cause_final": "NONE" または "BURIED" または "TRAPPED" または "TANGLED" または "MALFUNCTION" または "FALLEN",
  "reasoning": "適用したルールと判定理由（簡潔に）"
}}"""
    
    def _build_prompt(self, leg_id: str, spot_can: float, drone_can: float, p_drone: Dict[str, float]) -> str:
        """診断プロンプトを構築（真のLLM推論版 - 汎用性重視）"""
        
        # ドローンの確率分布を整形（参考情報として提示）
        p_drone_sorted = sorted(p_drone.items(), key=lambda x: -x[1])
        p_drone_text = "\n".join([f"   {cause}: {prob:.3f}" for cause, prob in p_drone_sorted[:3]])  # 上位3つ
        
        # NONE以外で最大の原因を特定（ルール2,4で必要）
        max_cause_no_none = max((v, k) for k, v in p_drone.items() if k != "NONE")[1]
        max_cause_all = max(p_drone.items(), key=lambda x: x[1])[0]
        
        prompt = f"""あなたは4足歩行ロボットの故障診断AIです。以下のデータを分析し、診断ルールに従って判定してください。

【診断対象】
脚ID: {leg_id}

【測定データ】
• Spot内部センサー評価: {spot_can:.3f} (範囲: 0.0～1.0、1.0が完全正常)
• ドローン外部観測評価: {drone_can:.3f} (範囲: 0.0～1.0、1.0が完全正常)

【ドローンの拘束原因推定】(確率分布の上位)
{p_drone_text}

【診断ルール - 以下に厳密に従うこと】
ルール1: Spot評価 ≥ 0.7 かつ ドローン評価 ≥ 0.7
  → 判定: "動く", 原因: "NONE"
  
ルール2: Spot評価 ≤ 0.3 かつ ドローン評価 ≤ 0.3
  → 判定: "動かない", 原因: ドローン推定の最大値 (NONE以外から選択)
  → この場合の候補: {max_cause_no_none}
  
ルール3: (Spot評価 ≥ 0.7 かつ ドローン評価 ≤ 0.3) または (Spot評価 ≤ 0.3 かつ ドローン評価 ≥ 0.7)
  → 判定: "動かない", 原因: "MALFUNCTION" (センサー矛盾)
  
ルール4: 上記以外の中間値
  → 判定: "一部動く", 原因: ドローン推定の最大値
  → この場合の候補: {max_cause_all}

【タスク】
1. 測定データから適用すべきルールを判定
2. そのルールに従って movement_result と cause_final を決定
3. 判定理由を reasoning に記述

【出力形式】
以下のJSON形式で出力してください（他の説明は不要、JSONのみ）:
{{
  "movement_result": "動く" または "動かない" または "一部動く",
  "cause_final": "NONE" または "BURIED" または "TRAPPED" または "TANGLED" または "MALFUNCTION" または "FALLEN",
  "reasoning": "適用したルールと判定理由（簡潔に）"
}}"""
        return prompt
    
    def _parse_llm_response(self, response_text: str, leg: LegState, p_drone: Dict[str, float]) -> Dict[str, float]:
        """LLMレスポンスをパースして確率分布を生成"""
        try:
            # JSONブロックを抽出（```json ... ``` または { ... } ）
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("JSON not found in response")
            
            result = json.loads(json_str)
            
            movement_result = result.get('movement_result', '一部動く')
            cause_final = result.get('cause_final', 'NONE')
            reasoning = result.get('reasoning', '')
            
            # LegStateに設定
            leg.movement_result = movement_result
            leg.cause_final = cause_final
            
            # spot_canとdrone_canの平均をp_canとする
            leg.p_can = (leg.spot_can + leg.drone_can) / 2
            
            # 確率分布を生成（原因に基づく）
            distribution = self._generate_distribution(cause_final)
            leg.p_llm = distribution
            
            print(f"[llm] {leg.leg_id}: {movement_result}, {cause_final} - {reasoning}")
            
            return distribution
            
        except Exception as e:
            print(f"[llm] Failed to parse LLM response: {e}")
            print(f"[llm] Response: {response_text[:200]}")
            # フォールバック: ルールベースで判定
            return self._infer_with_rules(leg)
    
    def _generate_distribution(self, cause_final: str) -> Dict[str, float]:
        """原因に基づいて確率分布を生成"""
        distribution = {
            "NONE": 0.01,
            "BURIED": 0.01,
            "TRAPPED": 0.01,
            "TANGLED": 0.01,
            "MALFUNCTION": 0.01,
            "FALLEN": 0.01,
        }
        
        # 最終原因に高い確率を割り当て
        if cause_final in distribution:
            distribution[cause_final] = 0.94
        
        return distribution
    
    def _infer_with_rules(self, leg: LegState) -> Dict[str, float]:
        """ルールベース推論（フォールバック）"""
        # 仕様通り、spot_can と drone_can を使用
        spot_can = leg.spot_can
        drone_can = leg.drone_can
        p_drone = dict(leg.p_drone)  # ドローンの確率分布をコピー
        
                # ルール①: 両方が0.7以上 → 動く、拘束原因=正常
        if spot_can >= 0.7 and drone_can >= 0.7:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2  # 平均を最終確率とする
            distribution = {
                "NONE": 0.94,
                "BURIED": 0.01,
                "TRAPPED": 0.01,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.02,
                "FALLEN": 0.01,
            }
            leg.p_llm = distribution
            leg.cause_final = "NONE"
            return distribution
        
        # ルール②: 両方が0.3以下 → 動かない、拘束原因=確率分布の最大値（NONE除外）
        elif spot_can <= 0.3 and drone_can <= 0.3:
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2
            
            # 確率分布から最大値を見つける（NONE以外から選択）
            # 「動かない」場合は正常(NONE)ではないため
            max_cause = max(
                (v, k) for k, v in p_drone.items() if k != "NONE"
            )[1]
            
            # 最大値の原因を強調した分布を作成
            distribution = {
                "NONE": 0.02,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.02,
                "MALFUNCTION": 0.02,
                "FALLEN": 0.01,
            }
            distribution[max_cause] = 0.89
            
            leg.p_llm = distribution
            leg.cause_final = max_cause
            return distribution
        
        # ルール③: 片方が0.7以上、もう片方が0.3以下 → 動かない、拘束原因=故障
        elif (spot_can >= 0.7 and drone_can <= 0.3) or (spot_can <= 0.3 and drone_can >= 0.7):
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2
            distribution = {
                "NONE": 0.01,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.93,  # 故障を強く示唆
                "FALLEN": 0.01,
            }
            leg.p_llm = distribution
            leg.cause_final = "MALFUNCTION"
            return distribution
        
        # ルール④: 片方が中間値 → 一部動く、拘束原因=確率分布の最大値
        else:
            leg.movement_result = "一部動く"
            leg.p_can = (spot_can + drone_can) / 2
            
            # 確率分布から最大値を見つける（仕様通り）
            max_cause = max(p_drone.items(), key=lambda x: x[1])[0]
            
            # 中間的な分布を作成（確信度は低め）
            distribution = {
                "NONE": 0.10,
                "BURIED": 0.05,
                "TRAPPED": 0.05,
                "TANGLED": 0.05,
                "MALFUNCTION": 0.05,
                "FALLEN": 0.01,
            }
            distribution[max_cause] = 0.69
            
            leg.p_llm = distribution
            leg.cause_final = max_cause
            return distribution
