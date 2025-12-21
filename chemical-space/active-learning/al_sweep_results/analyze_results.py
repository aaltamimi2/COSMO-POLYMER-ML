import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read summary
df = pd.read_csv('al_sweep_results/sweep_summary.csv')

# Filter out errors
df = df[df['best_polymer'] != 'ERROR'].copy()
df['best_score'] = pd.to_numeric(df['best_score'], errors='coerce')
df['efficiency_rank'] = pd.to_numeric(df['efficiency_rank'], errors='coerce')

print("=" * 70)
print("ACTIVE LEARNING SWEEP SUMMARY")
print("=" * 70)
print(f"\nTotal experiments: {len(df)}")
print(f"Average runtime: {df['runtime_sec'].mean():.1f}s")
print(f"\nBest polymer found most often: {df['best_polymer'].mode()[0] if len(df) > 0 else 'N/A'}")
print(f"Average best score: {df['best_score'].mean():.3f}")
print(f"Average efficiency rank: {df['efficiency_rank'].mean():.1f}")

# Group by acquisition function
print("\n" + "-" * 70)
print("Performance by Acquisition Function:")
print("-" * 70)
acq_stats = df.groupby('acquisition').agg({
    'best_score': ['mean', 'std'],
    'efficiency_rank': 'mean',
    'runtime_sec': 'mean'
}).round(3)
print(acq_stats)

# Group by temperature
print("\n" + "-" * 70)
print("Performance by Temperature:")
print("-" * 70)
temp_stats = df.groupby('temperature').agg({
    'best_score': ['mean', 'std'],
    'efficiency_rank': 'mean',
    'best_polymer': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
}).round(3)
print(temp_stats)

# Group by method
print("\n" + "-" * 70)
print("Performance by Scoring Method:")
print("-" * 70)
method_stats = df.groupby('method').agg({
    'best_score': ['mean', 'std'],
    'efficiency_rank': 'mean'
}).round(3)
print(method_stats)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Best score by acquisition
ax = axes[0, 0]
df.boxplot(column='best_score', by='acquisition', ax=ax)
ax.set_title('Best Score by Acquisition Function')
ax.set_xlabel('Acquisition Function')
ax.set_ylabel('Best Score')
plt.sca(ax)
plt.xticks(rotation=45)

# 2. Efficiency rank by acquisition
ax = axes[0, 1]
df.boxplot(column='efficiency_rank', by='acquisition', ax=ax)
ax.set_title('Efficiency Rank by Acquisition Function')
ax.set_xlabel('Acquisition Function')
ax.set_ylabel('Rank (lower is better)')
plt.sca(ax)
plt.xticks(rotation=45)

# 3. Best score by temperature
ax = axes[0, 2]
temp_grouped = df.groupby('temperature')['best_score'].mean()
ax.plot(temp_grouped.index, temp_grouped.values, 'o-', linewidth=2, markersize=8)
ax.set_title('Best Score vs Temperature')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Average Best Score')
ax.grid(True, alpha=0.3)

# 4. Runtime by configuration
ax = axes[1, 0]
df.boxplot(column='runtime_sec', by='acquisition', ax=ax)
ax.set_title('Runtime by Acquisition Function')
ax.set_xlabel('Acquisition Function')
ax.set_ylabel('Runtime (seconds)')
plt.sca(ax)
plt.xticks(rotation=45)

# 5. Best polymer distribution
ax = axes[1, 1]
polymer_counts = df['best_polymer'].value_counts()
ax.bar(range(len(polymer_counts)), polymer_counts.values)
ax.set_xticks(range(len(polymer_counts)))
ax.set_xticklabels(polymer_counts.index, rotation=45, ha='right')
ax.set_title('Best Polymer Found (Frequency)')
ax.set_ylabel('Count')

# 6. Score vs efficiency rank
ax = axes[1, 2]
for acq in df['acquisition'].unique():
    subset = df[df['acquisition'] == acq]
    ax.scatter(subset['efficiency_rank'], subset['best_score'],
              label=acq, alpha=0.7, s=100)
ax.set_xlabel('Efficiency Rank')
ax.set_ylabel('Best Score')
ax.set_title('Score vs Efficiency by Acquisition')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('al_sweep_results/sweep_analysis.png', dpi=200, bbox_inches='tight')
print(f"\n[OK] Saved visualization: al_sweep_results/sweep_analysis.png")

# Save detailed stats
with open('al_sweep_results/detailed_stats.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DETAILED STATISTICS\n")
    f.write("=" * 70 + "\n\n")

    f.write("Overall Statistics:\n")
    f.write("-" * 70 + "\n")
    f.write(df.describe().to_string())
    f.write("\n\n")

    f.write("Top 10 Experiments by Best Score:\n")
    f.write("-" * 70 + "\n")
    f.write(df.nlargest(10, 'best_score')[['experiment', 'temperature', 'acquisition',
                                            'method', 'best_polymer', 'best_score',
                                            'efficiency_rank']].to_string(index=False))
    f.write("\n\n")

    f.write("Top 10 Experiments by Efficiency:\n")
    f.write("-" * 70 + "\n")
    f.write(df.nsmallest(10, 'efficiency_rank')[['experiment', 'temperature', 'acquisition',
                                                   'method', 'best_polymer', 'best_score',
                                                   'efficiency_rank']].to_string(index=False))
    f.write("\n")

print(f"[OK] Saved detailed stats: al_sweep_results/detailed_stats.txt")
