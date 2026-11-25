import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean visual defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.05)

# ==========================================
# 1. DATA INGESTION & QUALITY AUDIT
# ==========================================
df_raw = pd.read_csv('bmw_pricing_challenge.csv')
df = df_raw.copy()

# Drop zero-variance column ('BMW' for all rows)
df = df.drop(columns=['maker_key'])

# Parse date columns
df['registration_date'] = pd.to_datetime(df['registration_date'])
df['sold_at'] = pd.to_datetime(df['sold_at'])

# ==========================================
# 2. DATA CLEANING & OUTLIER FILTERING
# ==========================================
# Impute 0 kW engine_power on X1 using model-specific median
x1_med_power = df[(df['model_key'] == 'X1') & (df['engine_power'] > 0)]['engine_power'].median()
df.loc[df['engine_power'] == 0, 'engine_power'] = x1_med_power

# Remove negative/corrupt mileage and non-retail price placeholders (< €1k or > €100k)
df_clean = df[
    (df['mileage'] > 0) &
    (df['mileage'] <= 500000) &
    (df['price'] >= 1000) &
    (df['price'] <= 100000)
].copy()

print(f"Cleaned dataset shape: {df_clean.shape} (Removed {len(df_raw) - len(df_clean)} anomalous rows)")

# ==========================================
# 3. FEATURE ENGINEERING (WRANGLING)
# ==========================================
# Vehicle age at time of sale (in years)
df_clean['vehicle_age_years'] = ((df_clean['sold_at'] - df_clean['registration_date']).dt.days / 365.25).round(2)

# Usage intensity (km driven per year, floored at 0.5 yrs to avoid division spikes)
df_clean['km_per_year'] = (df_clean['mileage'] / df_clean['vehicle_age_years'].clip(lower=0.5)).round(0)

# Convert kW to Horsepower (HP)
df_clean['engine_hp'] = (df_clean['engine_power'] * 1.34102).round(0)

# Aggregate count of optional features (feature_1 to feature_8)
feature_cols = [f'feature_{i}' for i in range(1, 9)]
df_clean['total_features'] = df_clean[feature_cols].sum(axis=1)

# Group 75 granular model_keys into high-level BMW Series families
def extract_series(model: str) -> str:
    m = str(model).strip()
    if m.startswith(('M', 'i')):
        return 'M / i Performance'
    elif m.startswith('X'):
        return 'X Series (SUV)'
    elif m.startswith('Z'):
        return 'Z Series (Roadster)'
    elif m.startswith('ActiveHybrid'):
        return '5 Series'
    elif m[0].isdigit():
        return f'{m[0]} Series'
    return 'Other'

df_clean['series'] = df_clean['model_key'].apply(extract_series)

# ==========================================
# 4. EXPLORATORY DATA ANALYSIS DASHBOARD
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Panel 1: Price Distribution
ax1 = axes[0, 0]
sns.histplot(df_clean['price'], bins=45, kde=True, color='#1f4e78', alpha=0.75, ax=ax1)
ax1.axvline(df_clean['price'].median(), color='#2ca02c', linestyle='--', linewidth=2,
            label=f"Median: €{df_clean['price'].median():,.0f}")
ax1.axvline(df_clean['price'].mean(), color='#d95f02', linestyle='-', linewidth=2,
            label=f"Mean: €{df_clean['price'].mean():,.0f}")
ax1.set_title('1. Resale Price Distribution (Right-Skewed)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Sale Price (€)')
ax1.set_ylabel('Vehicle Count')
ax1.legend()

# Panel 2: Depreciation Curve (Price vs. Mileage colored by Age)
ax2 = axes[0, 1]
df_age_cap = df_clean[df_clean['vehicle_age_years'] <= 12]
sc = ax2.scatter(df_age_cap['mileage'] / 1000, df_age_cap['price'] / 1000,
                 c=df_age_cap['vehicle_age_years'], cmap='viridis_r', alpha=0.55, s=18)
df_clean['mileage_bin'] = pd.cut(df_clean['mileage'], bins=np.arange(0, 350000, 25000))
trend = df_clean.groupby('mileage_bin', observed=True)['price'].median() / 1000
ax2.plot([i.mid / 1000 for i in trend.index], trend.values, color='red',
         linewidth=2.5, marker='o', markersize=4, label='Median Price Trend')
plt.colorbar(sc, ax=ax2, label='Vehicle Age (Years)')
ax2.set_title('2. Depreciation Curve: Price vs. Mileage & Age', fontsize=13, fontweight='bold')
ax2.set_xlabel('Mileage (Thousands of km)')
ax2.set_ylabel('Sale Price (€ Thousands)')
ax2.set_xlim(0, 350)
ax2.set_ylim(0, 85)
ax2.legend()

# Panel 3: Price Spread by BMW Series
ax3 = axes[1, 0]
series_order = df_clean.groupby('series')['price'].median().sort_values(ascending=False).index
sns.boxplot(data=df_clean, x='price', y='series', order=series_order, palette='Blues_r',
            flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.4}, ax=ax3)
ax3.set_title('3. Price Spread Across BMW Model Series', fontsize=13, fontweight='bold')
ax3.set_xlabel('Sale Price (€)')
ax3.set_ylabel('')
ax3.set_xlim(0, 95000)

# Panel 4: Equipment Option Price Lift
ax4 = axes[1, 1]
lifts = [
    {
        'feature': f"{f} ({df_clean[f].mean()*100:.0f}% cars)",
        'lift': df_clean[df_clean[f]]['price'].median() - df_clean[~df_clean[f]]['price'].median()
    }
    for f in feature_cols
]
lift_df = pd.DataFrame(lifts).sort_values('lift', ascending=True)
colors = ['#d9534f' if x < 0 else '#2b5c8f' for x in lift_df['lift']]
bars = ax4.barh(lift_df['feature'], lift_df['lift'], color=colors, alpha=0.85)
ax4.axvline(0, color='black', linewidth=1)
for bar in bars:
    w = bar.get_width()
    offset = 250 if w >= 0 else -1100
    ax4.text(w + offset, bar.get_y() + bar.get_height()/2, f"€{w:+,.0f}",
             va='center', fontsize=9.5, fontweight='bold')
ax4.set_title('4. Median Price Lift by Equipment Option (True vs. False)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Median Price Difference (€)')
ax4.set_xlim(-2200, 8800)

plt.suptitle('BMW Used Car Pricing — Exploratory Data Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('bmw_eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
