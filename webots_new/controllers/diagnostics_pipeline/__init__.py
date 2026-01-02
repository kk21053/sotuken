"""diagnostics_pipeline（webots_new 簡潔版）

Spot（自己診断）と Drone（外部観測）の結果を統合して、
脚が「動く/動かない/一部動く」と拘束原因を推定します。

依存関係（大まか）:
- pipeline.DiagnosticsPipeline が全体の司令塔
- self_diagnosis.SelfDiagnosisAggregator が Spot の内部診断を集計
- drone_observer.DroneObservationAggregator が Drone 観測を集計
- llm_client.LLMAnalyzer が仕様のルールで最終判定
- logger.DiagnosticsLogger が JSONL に保存
"""

from .pipeline import DiagnosticsPipeline
