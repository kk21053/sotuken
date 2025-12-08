"""Advanced LLM-based diagnosis system for Jetson Orin Nano Super.

低信頼度の場合にLLMを使用した高度な診断を実行します。
- llama.cpp による軽量推論
- RAGによるSpotマニュアル情報統合
- 生のRoboPoseデータ活用
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .models import LegState
from .rag_manual import get_manual_rag

# llama-cpp-pythonのインポート（オプション）
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("[llm_advanced] Warning: llama-cpp-python not found. Install with:")
    print("  CMAKE_ARGS=\"-DGGML_CUDA=ON\" pip install llama-cpp-python")


CAUSE_LABELS = ["NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION", "FALLEN"]


class AdvancedLLMAnalyzer:
    """
    Advanced LLM-based analyzer using RAG and llama.cpp.
    
    低信頼度の診断に対してLLMによる詳細分析を実行:
    - Spotマニュアルから関連情報を検索（RAG）
    - 生のRoboPose測定値を考慮
    - 多段階推論による高精度診断
    
    Processing time: ~3sec per leg (Jetson Orin Nano Super)
    Memory usage: ~3GB (model + embeddings)
    """
    
    def __init__(
        self,
        model_path: str = "models/llama-3.2-3b-instruct-q4_k_m.gguf",
        n_ctx: int = 2048,
        n_gpu_layers: int = 33,  # Jetson Orin Nano Super: 全層をGPUへ
        verbose: bool = False,
    ):
        """
        Initialize LLM analyzer.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            verbose: Enable verbose logging
        """
        self.enabled = LLAMA_CPP_AVAILABLE
        
        if not self.enabled:
            print("[llm_advanced] LLM diagnosis disabled (missing llama-cpp-python)")
            return
        
        # Load LLM model
        try:
            print(f"[llm_advanced] Loading model: {model_path}")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )
            print("[llm_advanced] Model loaded successfully")
        except Exception as e:
            print(f"[llm_advanced] Failed to load model: {e}")
            self.enabled = False
            return
        
        # Initialize RAG system
        self.rag = get_manual_rag()
        if self.rag is None or not self.rag.enabled:
            print("[llm_advanced] Warning: RAG system not available")
    
    def diagnose(
        self,
        leg: LegState,
        rule_based_result: Dict[str, float],
        confidence: float,
    ) -> Dict[str, float]:
        """
        Perform advanced LLM-based diagnosis.
        
        Args:
            leg: LegState with sensor data
            rule_based_result: Rule-based diagnosis result
            confidence: Confidence score of rule-based result
        
        Returns:
            Updated probability distribution
        """
        if not self.enabled:
            print("[llm_advanced] LLM disabled, returning rule-based result")
            return rule_based_result
        
        # Build diagnostic prompt
        prompt = self._build_prompt(leg, rule_based_result, confidence)
        
        # Call LLM
        print(f"[llm_advanced] Running LLM diagnosis for {leg.leg_id}...")
        response = self._call_llm(prompt)
        
        # Parse response
        distribution = self._parse_response(response, rule_based_result)
        
        return distribution
    
    def _build_prompt(
        self,
        leg: LegState,
        rule_based_result: Dict[str, float],
        confidence: float,
    ) -> str:
        """Build diagnostic prompt with context."""
        # Get manual context via RAG
        manual_context = ""
        if self.rag and self.rag.enabled:
            symptoms = []
            if leg.spot_can < 0.3:
                symptoms.append("自己診断スコアが低い")
            if leg.drone_can < 0.3:
                symptoms.append("外部観測で動作不良")
            
            sensor_data = {
                "spot_can": leg.spot_can,
                "drone_can": leg.drone_can,
            }
            
            manual_context = self.rag.get_context_for_diagnosis(
                leg.leg_id,
                symptoms,
                sensor_data,
            )
        
        # Prepare RoboPose raw data
        robopose_data = json.dumps(dict(leg.p_drone), indent=2, ensure_ascii=False)
        
        # Get top cause from rule-based
        top_cause = max(rule_based_result.items(), key=lambda x: x[1])[0]
        
        # Build prompt
        prompt = f"""あなたは4足歩行ロボット「Spot」の診断エキスパートです。
以下の情報をもとに、{leg.leg_id}の拘束原因を診断してください。

## 基本情報
- 脚ID: {leg.leg_id}
- 自己診断スコア (spot_can): {leg.spot_can:.3f}
- 外部観測スコア (drone_can): {leg.drone_can:.3f}

## ルールベース診断結果（信頼度: {confidence:.1%}）
推定原因: {top_cause}
確率分布:
{json.dumps(rule_based_result, indent=2, ensure_ascii=False)}

## RoboPose生測定値（ドローン画像分析）
{robopose_data}

## Spotマニュアルからの関連情報
{manual_context}

## 診断可能な拘束原因
- NONE: 正常（拘束なし）
- BURIED: 地面や砂に埋まっている
- TRAPPED: 末端が障害物に固定されている
- TANGLED: ツタなどに絡まっている
- MALFUNCTION: センサー故障または測定エラー
- FALLEN: ロボットが転倒している

## タスク
上記の情報を総合的に分析し、最も可能性の高い拘束原因を診断してください。
必ず以下のJSON形式で回答してください:

```json
{{
  "analysis": "診断の根拠を簡潔に説明",
  "distribution": {{
    "NONE": 0.0~1.0の確率,
    "BURIED": 0.0~1.0の確率,
    "TRAPPED": 0.0~1.0の確率,
    "TANGLED": 0.0~1.0の確率,
    "MALFUNCTION": 0.0~1.0の確率,
    "FALLEN": 0.0~1.0の確率
  }}
}}
```

注意: distribution内の確率の合計は1.0になるようにしてください。
"""
        
        return prompt
    
    def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Call LLM and get response."""
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.3,  # 低温度で安定した出力
                top_p=0.9,
                echo=False,
            )
            
            response = output["choices"][0]["text"]
            return response
        
        except Exception as e:
            print(f"[llm_advanced] LLM call failed: {e}")
            return ""
    
    def _parse_response(
        self,
        response: str,
        fallback: Dict[str, float],
    ) -> Dict[str, float]:
        """Parse LLM response and extract probability distribution."""
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                # Try without code fence
                json_match = re.search(r'\{.*?"distribution".*?\}', response, re.DOTALL)
            
            if not json_match:
                print("[llm_advanced] No JSON found in response, using fallback")
                return fallback
            
            json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
            data = json.loads(json_str)
            
            distribution = data.get("distribution", {})
            
            # Validate distribution
            if not all(label in distribution for label in CAUSE_LABELS):
                print("[llm_advanced] Incomplete distribution, using fallback")
                return fallback
            
            # Normalize to sum to 1.0
            total = sum(distribution.values())
            if total > 0:
                distribution = {k: v / total for k, v in distribution.items()}
            else:
                print("[llm_advanced] Zero total probability, using fallback")
                return fallback
            
            print(f"[llm_advanced] Parsed distribution: {distribution}")
            return distribution
        
        except Exception as e:
            print(f"[llm_advanced] Failed to parse response: {e}")
            print(f"[llm_advanced] Response: {response[:200]}...")
            return fallback


# Singleton instance
_llm_analyzer_instance: Optional[AdvancedLLMAnalyzer] = None


def get_llm_analyzer(
    model_path: str = "models/llama-3.2-3b-instruct-q4_k_m.gguf",
) -> Optional[AdvancedLLMAnalyzer]:
    """Get singleton LLM analyzer instance."""
    global _llm_analyzer_instance
    
    if _llm_analyzer_instance is None:
        try:
            _llm_analyzer_instance = AdvancedLLMAnalyzer(model_path=model_path)
        except Exception as e:
            print(f"[llm_advanced] Failed to initialize: {e}")
            return None
    
    return _llm_analyzer_instance
