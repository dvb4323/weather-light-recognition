"""
Comprehensive dataset analysis for BDD100K weather and time-of-day classification.

Analyzes:
- Class distribution per split (train/val/test)
- Cross-tabulation (weather × time combinations)
- Class imbalance ratios
- Recommendations for class weights

Usage: python utils/analyze_dataset_distribution.py
Output: Saves detailed report to utils/dataset_analysis_report.txt
"""

import os
import json
from collections import Counter, defaultdict
import numpy as np

def analyze_split(ann_dir, split_name):
    """Analyze a single data split."""
    
    weather_counts = Counter()
    time_counts = Counter()
    cross_counts = defaultdict(lambda: defaultdict(int))
    total = 0
    missing_weather = 0
    missing_time = 0
    
    if not os.path.exists(ann_dir):
        return None
    
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
    
    for ann_file in ann_files:
        total += 1
        ann_path = os.path.join(ann_dir, ann_file)
        
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            tags = ann.get('tags', [])
            
            weather = None
            timeofday = None
            
            for tag in tags:
                if tag['name'] == 'weather':
                    weather = tag['value']
                elif tag['name'] == 'timeofday':
                    timeofday = tag['value']
            
            if weather:
                weather_counts[weather] += 1
            else:
                missing_weather += 1
                
            if timeofday:
                time_counts[timeofday] += 1
            else:
                missing_time += 1
            
            # Cross-tabulation
            if weather and timeofday:
                cross_counts[weather][timeofday] += 1
    
    return {
        'total': total,
        'weather': weather_counts,
        'time': time_counts,
        'cross': cross_counts,
        'missing_weather': missing_weather,
        'missing_time': missing_time
    }


def compute_class_weights(counts, total):
    """Compute balanced class weights."""
    weights = {}
    n_classes = len(counts)
    
    for class_name, count in counts.items():
        if count > 0:
            # Balanced weight: total / (n_classes * count)
            weights[class_name] = total / (n_classes * count)
        else:
            weights[class_name] = 0.0
    
    return weights


def format_report(all_stats):
    """Format comprehensive analysis report."""
    
    lines = []
    lines.append("="*80)
    lines.append("BDD100K DATASET ANALYSIS REPORT")
    lines.append("="*80)
    lines.append("")
    
    # Summary table
    lines.append("DATASET SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{'Split':<15} {'Total Samples':<15} {'Missing Weather':<18} {'Missing Time':<15}")
    lines.append("-" * 80)
    
    for split_name, stats in all_stats.items():
        if stats:
            lines.append(f"{split_name:<15} {stats['total']:<15} "
                        f"{stats['missing_weather']:<18} {stats['missing_time']:<15}")
    
    lines.append("")
    lines.append("")
    
    # Detailed per-split analysis
    for split_name, stats in all_stats.items():
        if not stats:
            continue
            
        lines.append("="*80)
        lines.append(f"{split_name.upper()} SET ANALYSIS")
        lines.append("="*80)
        lines.append(f"Total samples: {stats['total']}")
        lines.append("")
        
        # Weather distribution
        lines.append("WEATHER DISTRIBUTION")
        lines.append("-" * 80)
        lines.append(f"{'Class':<20} {'Count':<10} {'Percentage':<12} {'Balanced Weight':<15}")
        lines.append("-" * 80)
        
        weather_weights = compute_class_weights(stats['weather'], stats['total'])
        
        for weather, count in sorted(stats['weather'].items(), key=lambda x: -x[1]):
            pct = (count / stats['total']) * 100
            weight = weather_weights[weather]
            lines.append(f"{weather:<20} {count:<10} {pct:>6.2f}%      {weight:>6.3f}")
        
        lines.append("")
        
        # Time distribution
        lines.append("TIME OF DAY DISTRIBUTION")
        lines.append("-" * 80)
        lines.append(f"{'Class':<20} {'Count':<10} {'Percentage':<12} {'Balanced Weight':<15}")
        lines.append("-" * 80)
        
        time_weights = compute_class_weights(stats['time'], stats['total'])
        
        for time, count in sorted(stats['time'].items(), key=lambda x: -x[1]):
            pct = (count / stats['total']) * 100
            weight = time_weights[time]
            lines.append(f"{time:<20} {count:<10} {pct:>6.2f}%      {weight:>6.3f}")
        
        lines.append("")
        
        # Cross-tabulation
        lines.append("WEATHER × TIME OF DAY CROSS-TABULATION")
        lines.append("-" * 80)
        
        # Get all unique time values
        all_times = sorted(set(t for weather_dict in stats['cross'].values() for t in weather_dict.keys()))
        
        # Header
        header = f"{'Weather':<20}"
        for time in all_times:
            header += f"{time:<15}"
        header += f"{'Total':<10}"
        lines.append(header)
        lines.append("-" * 80)
        
        # Rows
        for weather in sorted(stats['weather'].keys()):
            row = f"{weather:<20}"
            row_total = 0
            for time in all_times:
                count = stats['cross'][weather].get(time, 0)
                row += f"{count:<15}"
                row_total += count
            row += f"{row_total:<10}"
            lines.append(row)
        
        lines.append("")
        lines.append("")
    
    # Recommendations
    lines.append("="*80)
    lines.append("RECOMMENDATIONS")
    lines.append("="*80)
    lines.append("")
    
    # Analyze train set for recommendations
    train_stats = all_stats.get('Train', None)
    if train_stats:
        lines.append("CLASS IMBALANCE ANALYSIS (Train Set)")
        lines.append("-" * 80)
        
        # Weather imbalance
        weather_counts = train_stats['weather']
        max_weather = max(weather_counts.values())
        min_weather = min(weather_counts.values())
        
        lines.append(f"\nWeather Classes:")
        lines.append(f"  Most common: {max(weather_counts, key=weather_counts.get)} ({max_weather} samples)")
        lines.append(f"  Least common: {min(weather_counts, key=weather_counts.get)} ({min_weather} samples)")
        lines.append(f"  Imbalance ratio: {max_weather / min_weather:.1f}:1")
        
        # Identify problematic classes
        lines.append(f"\n  Classes with <1% of data (problematic):")
        for weather, count in weather_counts.items():
            pct = (count / train_stats['total']) * 100
            if pct < 1.0:
                lines.append(f"    - {weather}: {count} samples ({pct:.2f}%) [!]")
        
        # Time imbalance
        time_counts = train_stats['time']
        max_time = max(time_counts.values())
        min_time = min(time_counts.values())
        
        lines.append(f"\nTime of Day Classes:")
        lines.append(f"  Most common: {max(time_counts, key=time_counts.get)} ({max_time} samples)")
        lines.append(f"  Least common: {min(time_counts, key=time_counts.get)} ({min_time} samples)")
        lines.append(f"  Imbalance ratio: {max_time / min_time:.1f}:1")
        
        lines.append(f"\n  Classes with <5% of data (may need attention):")
        for time, count in time_counts.items():
            pct = (count / train_stats['total']) * 100
            if pct < 5.0:
                lines.append(f"    - {time}: {count} samples ({pct:.2f}%) [!]")
        
        lines.append("")
        lines.append("SUGGESTED CLASS WEIGHTS FOR TRAINING")
        lines.append("-" * 80)
        
        lines.append("\nWeather class weights (for config.yaml):")
        lines.append("```python")
        lines.append("weather_class_weights = [")
        weather_weights = compute_class_weights(weather_counts, train_stats['total'])
        for weather in sorted(weather_counts.keys()):
            weight = weather_weights[weather]
            lines.append(f"    {weight:.4f},  # {weather}")
        lines.append("]")
        lines.append("```")
        
        lines.append("\nTime of day class weights (for config.yaml):")
        lines.append("```python")
        lines.append("time_class_weights = [")
        time_weights = compute_class_weights(time_counts, train_stats['total'])
        for time in sorted(time_counts.keys()):
            weight = time_weights[time]
            lines.append(f"    {weight:.4f},  # {time}")
        lines.append("]")
        lines.append("```")
        
        lines.append("")
        lines.append("ACTION ITEMS")
        lines.append("-" * 80)
        
        # Generate action items
        actions = []
        
        for weather, count in weather_counts.items():
            pct = (count / train_stats['total']) * 100
            if pct < 0.5:
                actions.append(f"[!] CRITICAL: '{weather}' has only {count} samples ({pct:.2f}%) - "
                             f"Consider removing or merging with similar class")
            elif pct < 5.0:
                actions.append(f"[!] '{weather}' is underrepresented ({pct:.2f}%) - "
                             f"Use class weights or data augmentation")
        
        for time, count in time_counts.items():
            pct = (count / train_stats['total']) * 100
            if pct < 5.0:
                actions.append(f"[!] '{time}' is underrepresented ({pct:.2f}%) - "
                             f"Use class weights or focal loss")
        
        if actions:
            for i, action in enumerate(actions, 1):
                lines.append(f"{i}. {action}")
        else:
            lines.append("[OK] No critical class imbalance issues detected")
    
    lines.append("")
    lines.append("="*80)
    lines.append("END OF REPORT")
    lines.append("="*80)
    
    return "\n".join(lines)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Analyze all splits
    splits = {
        'Train': os.path.join(base_dir, 'data', 'new_train', 'ann'),
        'Validation': os.path.join(base_dir, 'data', 'new_val', 'ann'),
        'Test': os.path.join(base_dir, 'data', 'val', 'ann'),
    }
    
    print("Analyzing dataset distribution...")
    print("="*80)
    
    all_stats = {}
    for split_name, ann_dir in splits.items():
        print(f"Analyzing {split_name} set: {ann_dir}")
        stats = analyze_split(ann_dir, split_name)
        if stats:
            all_stats[split_name] = stats
            print(f"  ✓ Found {stats['total']} samples")
        else:
            print(f"  ✗ Directory not found")
    
    print("")
    print("Generating report...")
    
    # Generate report
    report = format_report(all_stats)
    
    # Save to file
    output_path = os.path.join(base_dir, 'utils', 'dataset_analysis_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Report saved to: {output_path}")
    print("")
    print("Preview:")
    print("="*80)
    # Print first 50 lines as preview
    for line in report.split('\n')[:50]:
        print(line)
    print("...")
    print(f"\n(Full report saved to {output_path})")
