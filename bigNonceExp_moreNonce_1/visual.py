import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from matplotlib.ticker import PercentFormatter

data = """
NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
0.0,100.0,5,0.005512567397123755,0.004241908704242438,0.0,0.02033770297243351
0.0,100.0,10,0.009121630598746114,0.019032761749395652,0.001040582726326743,0.18909544783553162
0.0,100.0,20,0.011337451126633914,0.01394514158621301,0.0014662015268719348,0.11684503738454773

NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
0.0,200.0,5,0.0050501717851863725,0.005184270050383699,0.00010077089736484103,0.0325082130032852
0.0,200.0,10,0.007342206384429618,0.0077964431283596,0.0008738794610137001,0.06337463377907263
0.0,200.0,20,0.010298442462803958,0.008003616240260622,0.0010215453194650818,0.04074055821135807

NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
0.0,500.0,5,0.004153573390789994,0.0029202554074119983,0.00019311854259873186,0.017098863710715142
0.0,500.0,10,0.006419934035457327,0.007303684638599593,0.0002849772018238541,0.05664263645726056
0.0,500.0,20,0.010116870019094457,0.00932171283228931,0.00017671818269447034,0.06203478322221046

NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
0.0,1000.0,5,0.004533931191885698,0.006036933447633924,0.0,0.05610030541713551
0.0,1000.0,10,0.007212000156982896,0.008780057958226527,0.0,0.07957559681697612
0.0,1000.0,20,0.009456688166467896,0.009148444311906492,0.0010163892770931266,0.06338181028869112

NonceMin,NonceMax,SampleSize,AvgHitRate,StdDev,MinHitRate,MaxHitRate
0.0,2000.0,5,0.0042115742347439734,0.004623959305395143,0.0,0.035025017869907075
0.0,2000.0,10,0.006824264818107292,0.004053072476247001,0.0012098725600903372,0.022276993079912787
0.0,2000.0,20,0.010509897101074013,0.011078941324718827,0.0016733067729083665,0.08182651991614255
"""

# Data processing
blocks = data.strip().split('\n\n')
dfs = []
for block in blocks:
    lines = block.split('\n')
    header = lines[0].split(',')
    content = '\n'.join(lines[1:])
    df_block = pd.read_csv(StringIO(content), names=header)
    dfs.append(df_block)
df = pd.concat(dfs, ignore_index=True)

# Create visualization
plt.figure(figsize=(14, 8))
nonce_ranges = df[['NonceMin', 'NonceMax']].drop_duplicates().values

# 5-color palette with distinct markers
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', 'D', '^', 'v']  # 5 unique markers

# Plot each nonce range
for i, ((min_val, max_val), color, marker) in enumerate(zip(nonce_ranges, colors, markers)):
    subset = df[(df['NonceMin'] == min_val) & (df['NonceMax'] == max_val)]

    # Plot line with markers
    plt.plot(subset['SampleSize'], subset['AvgHitRate'],
             marker=marker, color=color, linestyle='--',
             markersize=9, linewidth=2,
             label=f'{min_val}-{max_val}')

# Formatting
ax = plt.gca()
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
plt.ylim(0, 0.02)  # Fixed 0-2% range
plt.xticks([5, 10, 20], fontsize=11)
plt.yticks(np.arange(0, 0.021, 0.005), fontsize=11)  # 0.5% increments

# Value labels with adjusted positioning
for _, row in df.iterrows():
    plt.text(row['SampleSize'], row['AvgHitRate'] + 0.0005,  # Reduced offset
             f"{row['AvgHitRate']:.3%}",
             ha='center', va='bottom', fontsize=8.5,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

# Labels and title
plt.xlabel('Sample Size', fontsize=12, labelpad=10)
plt.ylabel('Average Hit Rate', fontsize=12, labelpad=10)
plt.title('Average Hit Rate Comparison (0-2% Range)', fontsize=14, pad=15)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),  # Right-side legend
           title='Nonce Range', framealpha=0.9)

plt.tight_layout()
plt.show()