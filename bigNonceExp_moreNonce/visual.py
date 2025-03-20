import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.ticker import PercentFormatter

data = """
NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
0.0,500.0,5,0.0053545043108209935,0.005489123158548599,0.0007644228607401412,0.03681297464281117
0.0,500.0,10,0.006758688248521166,0.005790836173659188,0.0006868131868131869,0.04302143345915282
0.0,500.0,20,0.010317389863585458,0.007953437106859997,0.0001554462721157651,0.051797484923565985

NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
0.0,1000.0,5,0.004866818297379342,0.007817306034339189,0.0,0.0691388209725618
0.0,1000.0,10,0.007001053055953454,0.005634569067844085,0.0007865636422523481,0.03910691236646253
0.0,1000.0,20,0.00899894958141309,0.006817039346292801,0.0013238631587795242,0.035495518151933754

NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
100.0,200.0,5,0.007584352798883308,0.00874419016851186,0.001078167115902965,0.05229613892937427
100.0,200.0,10,0.006942944415379246,0.005020765290837383,0.0011500634517766497,0.032945627802690586
100.0,200.0,20,0.01459093912357682,0.0146709631062408,0.0028538424950737243,0.10809
"""

# Data processing (same as before)
blocks = data.strip().split('\n\n')
dfs = []
for block in blocks:
    lines = block.split('\n')
    header = lines[0].split(',')
    content = '\n'.join(lines[1:])
    df_block = pd.read_csv(StringIO(content), names=header)
    dfs.append(df_block)
df = pd.concat(dfs, ignore_index=True)

# Create focused visualization
plt.figure(figsize=(12, 7))
nonce_ranges = df[['NonceMin', 'NonceMax']].drop_duplicates().values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', 'D']  # Different markers for each range

# Plot averages with high precision
for i, ((min_val, max_val), color, marker) in enumerate(zip(nonce_ranges, colors, markers)):
    subset = df[(df['NonceMin'] == min_val) & (df['NonceMax'] == max_val)]

    # Plot connecting lines with markers
    plt.plot(subset['SampleSize'], subset['AvgHitRate'],
             marker=marker, color=color, linestyle='--',
             markersize=9, linewidth=2,
             label=f'{min_val}-{max_val}')

# Formatting for precision analysis
ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))  # Two decimal places
plt.ylim(0, 0.02)  # Focused range for better resolution
plt.xticks([5, 10, 20], fontsize=11)
plt.yticks(fontsize=11)

# Add precise value annotations
for _, row in df.iterrows():
    plt.text(row['SampleSize'], row['AvgHitRate'] + 0.0015,
             f"{row['AvgHitRate']:.3%}",  # Three decimal places
             ha='center', va='bottom', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Axis labels and title
plt.xlabel('Sample Size', fontsize=12, labelpad=8)
plt.ylabel('Average Hit Rate', fontsize=12, labelpad=8)
plt.title('Precision Comparison of Average Hit Rates', fontsize=14, pad=15)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.legend(loc='upper left', title='Nonce Range', framealpha=0.9)

plt.tight_layout()
plt.show()