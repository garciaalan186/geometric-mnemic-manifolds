import json
from pathlib import Path
from visualizer import BenchmarkVisualizer

def regenerate():
    # Target directory
    results_dir = Path("benchmark_results/20251207_155741")
    json_path = results_dir / "benchmark_results.json"
    
    print(f"Loading results from {json_path}...")
    with open(json_path, 'r') as f:
        results = json.load(f)
        
    print("Regenerating visualizations...")
    viz = BenchmarkVisualizer(output_dir=results_dir)
    viz.visualize_needle_results(results)
    print("Done!")

if __name__ == "__main__":
    regenerate()
