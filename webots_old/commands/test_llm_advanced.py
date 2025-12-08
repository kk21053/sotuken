#!/usr/bin/env python3
"""Test script for advanced LLM diagnosis system.

Tests:
1. RAG system (PDF parsing and search)
2. Confidence calculation
3. LLM diagnosis (with mock if model not available)
4. Integration with pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "controllers"))

from diagnostics_pipeline import config
from diagnostics_pipeline.models import LegState
from diagnostics_pipeline.llm_client import RuleBasedAnalyzer
from diagnostics_pipeline.rag_manual import get_manual_rag
from diagnostics_pipeline.llm_advanced import get_llm_analyzer


def test_rag_system():
    """Test RAG system for manual search."""
    print("\n" + "="*60)
    print("TEST 1: RAG System (PDF Manual Search)")
    print("="*60)
    
    rag = get_manual_rag(
        pdf_path=config.MANUAL_PDF_PATH,
        cache_dir=config.MANUAL_EMBEDDINGS_CACHE,
    )
    
    if not rag or not rag.enabled:
        print("❌ RAG system not available")
        print("   Install dependencies: pip install pymupdf sentence-transformers")
        return False
    
    print(f"✓ RAG initialized with {len(rag.chunks)} chunks")
    
    # Test search
    test_queries = [
        "脚が動かない原因",
        "センサー故障",
        "転倒時の対処方法",
    ]
    
    for query in test_queries:
        print(f"\n検索クエリ: '{query}'")
        results = rag.search(query, top_k=2)
        
        if results:
            for i, (chunk, score) in enumerate(results, 1):
                print(f"  結果{i} (類似度: {score:.3f}):")
                preview = chunk[:100].replace('\n', ' ')
                print(f"    {preview}...")
        else:
            print("  結果なし")
    
    print("\n✅ RAG test passed")
    return True


def test_confidence_calculation():
    """Test confidence calculation in rule-based analyzer."""
    print("\n" + "="*60)
    print("TEST 2: Confidence Calculation")
    print("="*60)
    
    analyzer = RuleBasedAnalyzer()
    
    test_cases = [
        # (spot_can, drone_can, expected_confidence_range)
        (0.8, 0.9, (0.90, 1.0)),   # High confidence (both high)
        (0.2, 0.1, (0.80, 1.0)),   # High confidence (both low, clear cause)
        (0.8, 0.2, (0.90, 1.0)),   # High confidence (malfunction)
        (0.5, 0.6, (0.60, 0.80)),  # Medium confidence
    ]
    
    for spot_can, drone_can, (min_conf, max_conf) in test_cases:
        leg = LegState(
            leg_id="FL",
            spot_can=spot_can,
            drone_can=drone_can,
            p_drone={
                "NONE": 0.1,
                "BURIED": 0.3,
                "TRAPPED": 0.2,
                "TANGLED": 0.2,
                "MALFUNCTION": 0.1,
                "FALLEN": 0.1,
            }
        )
        
        distribution, confidence = analyzer.infer_with_confidence(leg)
        
        print(f"\nspot_can={spot_can:.1f}, drone_can={drone_can:.1f}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Cause: {leg.cause_final}")
        
        if min_conf <= confidence <= max_conf:
            print(f"  ✓ Confidence in expected range [{min_conf:.2f}, {max_conf:.2f}]")
        else:
            print(f"  ❌ Confidence {confidence:.2f} outside expected range [{min_conf:.2f}, {max_conf:.2f}]")
            return False
    
    print("\n✅ Confidence calculation test passed")
    return True


def test_llm_diagnosis():
    """Test LLM diagnosis (if available)."""
    print("\n" + "="*60)
    print("TEST 3: LLM Diagnosis")
    print("="*60)
    
    llm_analyzer = get_llm_analyzer(model_path=config.LLM_MODEL_PATH)
    
    if not llm_analyzer or not llm_analyzer.enabled:
        print("⚠️  LLM not available (expected on systems without model)")
        print("   To enable: Download model and install llama-cpp-python")
        print("   This is normal for testing without Jetson hardware")
        return True  # Not a failure
    
    print("✓ LLM analyzer initialized")
    
    # Create test case: ambiguous scenario
    leg = LegState(
        leg_id="FR",
        spot_can=0.4,
        drone_can=0.5,
        p_drone={
            "NONE": 0.2,
            "BURIED": 0.3,
            "TRAPPED": 0.2,
            "TANGLED": 0.15,
            "MALFUNCTION": 0.1,
            "FALLEN": 0.05,
        }
    )
    
    # Get rule-based result first
    rule_based = RuleBasedAnalyzer()
    distribution, confidence = rule_based.infer_with_confidence(leg)
    
    print(f"\nRule-based result:")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Cause: {leg.cause_final}")
    
    # Run LLM diagnosis
    print("\nRunning LLM diagnosis...")
    llm_distribution = llm_analyzer.diagnose(leg, distribution, confidence)
    
    print(f"\nLLM result:")
    llm_cause = max(llm_distribution.items(), key=lambda x: x[1])[0]
    llm_confidence = max(llm_distribution.values())
    print(f"  Confidence: {llm_confidence:.2f}")
    print(f"  Cause: {llm_cause}")
    print(f"  Distribution: {llm_distribution}")
    
    # Validate distribution
    total = sum(llm_distribution.values())
    if abs(total - 1.0) > 0.01:
        print(f"❌ Distribution does not sum to 1.0 (sum={total})")
        return False
    
    print("\n✅ LLM diagnosis test passed")
    return True


def test_integration():
    """Test integration with pipeline."""
    print("\n" + "="*60)
    print("TEST 4: Pipeline Integration")
    print("="*60)
    
    # Test with LLM disabled (default)
    print("\nTest 4a: LLM disabled (rule-based only)")
    original_setting = config.USE_LLM_ADVANCED
    config.USE_LLM_ADVANCED = False
    
    from diagnostics_pipeline.pipeline import DiagnosticsPipeline
    
    pipeline = DiagnosticsPipeline(session_id="test_integration_1")
    
    if pipeline.llm_advanced is not None:
        print("❌ LLM should be None when disabled")
        config.USE_LLM_ADVANCED = original_setting
        return False
    
    print("✓ Pipeline initialized without LLM")
    
    # Test with LLM enabled
    print("\nTest 4b: LLM enabled (if available)")
    config.USE_LLM_ADVANCED = True
    
    pipeline2 = DiagnosticsPipeline(session_id="test_integration_2")
    
    # LLM may or may not be available, both are valid
    if pipeline2.llm_advanced:
        print("✓ Pipeline initialized with LLM")
    else:
        print("⚠️  Pipeline initialized without LLM (model not available)")
    
    # Restore original setting
    config.USE_LLM_ADVANCED = original_setting
    
    print("\n✅ Integration test passed")
    return True


def main():
    """Run all tests."""
    print("Advanced LLM Diagnosis System - Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: RAG system
    try:
        results.append(("RAG System", test_rag_system()))
    except Exception as e:
        print(f"\n❌ RAG test failed with exception: {e}")
        results.append(("RAG System", False))
    
    # Test 2: Confidence calculation
    try:
        results.append(("Confidence", test_confidence_calculation()))
    except Exception as e:
        print(f"\n❌ Confidence test failed with exception: {e}")
        results.append(("Confidence", False))
    
    # Test 3: LLM diagnosis
    try:
        results.append(("LLM Diagnosis", test_llm_diagnosis()))
    except Exception as e:
        print(f"\n❌ LLM test failed with exception: {e}")
        results.append(("LLM Diagnosis", False))
    
    # Test 4: Integration
    try:
        results.append(("Integration", test_integration()))
    except Exception as e:
        print(f"\n❌ Integration test failed with exception: {e}")
        results.append(("Integration", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
