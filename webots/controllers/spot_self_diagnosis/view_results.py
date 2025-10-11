#!/usr/bin/env python3
"""View and analyze diagnosis results from JSONL files."""

import json
import sys
from pathlib import Path
from datetime import datetime

def view_diagnosis_results(jsonl_path):
    """Display diagnosis results in human-readable format."""
    if not Path(jsonl_path).exists():
        print(f"❌ File not found: {jsonl_path}")
        return
    
    print("=" * 80)
    print(f"DIAGNOSIS RESULTS: {Path(jsonl_path).name}")
    print("=" * 80)
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                if data.get("type") == "session_metadata":
                    print(f"\n📅 Session Info:")
                    print(f"   Start: {data.get('start_time')}")
                    print(f"   End:   {data.get('end_time')}")
                    print(f"   Legs:  {data.get('num_legs')}")
                    print(f"   Trials per leg: {data.get('trials_per_leg')}")
                    print()
                
                elif data.get("type") == "leg_result":
                    leg_id = data.get("leg_id", "?")
                    print(f"🦿 Leg {leg_id}:")
                    print(f"   Self-diagnosis:  can-move={data.get('self_can', 0):.3f}")
                    print(f"   Drone observation: can-move={data.get('drone_can', 0):.3f}")
                    
                    p_final = data.get("p_final", {})
                    if p_final:
                        print(f"   Final diagnosis:")
                        for cause, prob in sorted(p_final.items(), key=lambda x: -x[1]):
                            bar = "█" * int(prob * 30)
                            print(f"      {cause:10s} {prob:.3f} {bar}")
                    
                    cause = data.get("cause_final", "?")
                    conf = data.get("conf_final", 0)
                    print(f"   ➜ Most likely: {cause} (confidence: {conf:.3f})")
                    print()
            
            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: Invalid JSON - {e}")
    
    print("=" * 80)

def list_available_results():
    """List all available diagnosis result files."""
    logs_dir = Path(__file__).parent / "logs"
    
    if not logs_dir.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return []
    
    files = sorted(logs_dir.glob("diagnosis_*.jsonl"), reverse=True)
    
    if not files:
        print("ℹ️  No diagnosis results found yet.")
        print(f"   Files will be created in: {logs_dir}")
        return []
    
    print(f"\n📁 Available diagnosis results ({len(files)}):")
    print()
    
    for i, filepath in enumerate(files, 1):
        size = filepath.stat().st_size
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        print(f"  {i}. {filepath.name}")
        print(f"     Size: {size} bytes | Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return files

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # File path provided as argument
        view_diagnosis_results(sys.argv[1])
    else:
        # List available files
        files = list_available_results()
        
        if files:
            print("\n" + "=" * 80)
            print("Usage:")
            print(f"  python3 {Path(__file__).name} <file_path>")
            print()
            print("Example:")
            print(f"  python3 {Path(__file__).name} logs/{files[0].name}")
            print("=" * 80)
