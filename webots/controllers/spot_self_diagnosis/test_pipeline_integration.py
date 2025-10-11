"""Test pipeline integration without Webots."""

import sys
from pathlib import Path

# Add diagnostics_pipeline to path
CONTROLLERS_ROOT = Path(__file__).resolve().parents[1]
if str(CONTROLLERS_ROOT) not in sys.path:
    sys.path.append(str(CONTROLLERS_ROOT))

from diagnostics_pipeline.pipeline import DiagnosticsPipeline
from diagnostics_pipeline import config

def test_pipeline():
    """Test basic pipeline functionality."""
    print("Testing DiagnosticsPipeline...")
    
    session_id = "test_session_001"
    pipeline = DiagnosticsPipeline(session_id)
    
    # Test single leg, single trial
    leg_id = "FL"
    trial_index = 1
    direction = "+"
    start_time = 0.0
    
    print(f"\n1. Starting trial {trial_index} for leg {leg_id}")
    pipeline.start_trial(leg_id, trial_index, direction, start_time)
    
    # Create sample trial data
    print("2. Simulating trial data collection")
    theta_cmd = [0.0, 1.0, 2.0, 3.0, 3.0]
    theta_meas = [0.0, 0.8, 1.9, 2.7, 2.8]
    omega_meas = [0.0, 0.5, 0.4, 0.3, 0.1]
    tau_meas = [0.0, 2.0, 1.5, 1.0, 0.5]
    
    # Simulate drone observations
    print("3. Recording simulated RoboPose frames")
    for i in range(9):  # 0.5s * 18 fps = 9 frames
        joint_angles = [0.0, 0.0, 0.0]
        end_position = [0.3, 0.0, 0.0]
        base_orientation = [0.0, 0.0, 0.0]
        base_position = [0.0, 0.0, 0.3]
        
        pipeline.record_robo_pose_frame(
            leg_id=leg_id,
            joint_angles=joint_angles,
            end_position=end_position,
            base_orientation=base_orientation,
            base_position=base_position
        )
    
    # Complete trial
    print("4. Completing trial")
    end_time = start_time + 0.5
    pipeline.complete_trial(
        leg_id=leg_id,
        theta_cmd=theta_cmd,
        theta_meas=theta_meas,
        omega_meas=omega_meas,
        tau_meas=tau_meas,
        tau_nominal=10.0,
        safety_level="SAFE",
        end_time=end_time
    )
    
    # Finalize
    print("5. Finalizing pipeline")
    session_record = pipeline.finalize()
    
    # Check results
    print("\n=== RESULTS ===")
    leg_state = session_record.legs.get(leg_id)
    if leg_state:
        print(f"Leg {leg_id}:")
        print(f"  Trials: {len(leg_state.trials)}")
        print(f"  Self-diagnosis can-move: {leg_state.self_can:.3f}")
        print(f"  Drone observation can-move: {leg_state.drone_can}")
        print(f"  Drone cause distribution: {leg_state.p_drone}")
        print(f"  LLM cause distribution: {leg_state.p_llm}")
        print(f"  Fused cause distribution: {leg_state.p_final}")
        print(f"  Final cause: {leg_state.cause_final} (confidence: {leg_state.conf_final:.3f})")
        print("\n✅ Pipeline test passed!")
    else:
        print("❌ No leg state found!")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = test_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
