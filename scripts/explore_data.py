import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('data/raw.csv')

print("=== Thống kê tổng quan ===")

print(df.groupby('label')[['temperature', 'humidity']].describe().round(2))

print(f"\nTổng samples: {len(df)} | Normal: {(df.label==0).sum()} | Anomaly: {(df.label==1).sum()}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

fig.suptitle('Data Exploration – SHT31 Sensor', fontsize=14)

# Temperature timeline

axes[0,0].plot(df[df.label==0]['temperature'].values, label='Normal',  alpha=0.7, color='steelblue')

axes[0,0].plot(df[df.label==1]['temperature'].reset_index(drop=True).values,

               label='Anomaly', alpha=0.8, color='crimson')

axes[0,0].set_title('Temperature over time'); axes[0,0].legend(); axes[0,0].set_ylabel('°C')

# Humidity timeline

axes[0,1].plot(df[df.label==0]['humidity'].values, alpha=0.7, color='steelblue', label='Normal')

axes[0,1].plot(df[df.label==1]['humidity'].reset_index(drop=True).values,

               alpha=0.8, color='crimson', label='Anomaly')

axes[0,1].set_title('Humidity over time'); axes[0,1].legend(); axes[0,1].set_ylabel('%')

# Scatter T vs H

colors = df['label'].map({0: 'steelblue', 1: 'crimson'})

axes[1,0].scatter(df['temperature'], df['humidity'], c=colors, alpha=0.5, s=10)

axes[1,0].set_xlabel('Temperature (°C)'); axes[1,0].set_ylabel('Humidity (%)')

axes[1,0].set_title('T vs H scatter (blue=Normal, red=Anomaly)')

# Distribution

axes[1,1].hist(df[df.label==0]['temperature'], bins=30, alpha=0.6, label='Normal', color='steelblue')

axes[1,1].hist(df[df.label==1]['temperature'], bins=30, alpha=0.6, label='Anomaly', color='crimson')

axes[1,1].set_title('Temperature distribution'); axes[1,1].legend()

plt.tight_layout()

plt.savefig('data/exploration.png', dpi=150)

print("\nBiểu đồ lưu tại: data/exploration.png")
