"""
====================================================================
REVENUE ANALYSIS V3 - DEEP DIVE
Topic 1: Revenue & Seasonality

New in V3:
1. Monthly Revenue Drivers Decomposition (traffic, orders, AOV, conversion)
2. Seasonality Stability Analysis (CV, rank stability)
3. Traffic-Revenue Lag Analysis (correlation at various lags)
4. Enhanced recommendations with more evidence
====================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Config
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

DATA_DIR = './data/'
OUTPUT_DIR = './notebooks/'

print('='*80)
print('REVENUE ANALYSIS V3 - DEEP DIVE')
print('='*80)

# =====================================================================
# LOAD DATA
# =====================================================================
print('\n[1] LOADING DATA...')

sales = pd.read_csv(DATA_DIR + 'sales.csv', parse_dates=['Date'])
orders = pd.read_csv(DATA_DIR + 'orders.csv', parse_dates=['order_date'])
order_items = pd.read_csv(DATA_DIR + 'order_items.csv', low_memory=False)
promotions = pd.read_csv(DATA_DIR + 'promotions.csv')
web_traffic = pd.read_csv(DATA_DIR + 'web_traffic.csv', parse_dates=['date'])

print(f'  Sales: {len(sales):,} rows ({sales["Date"].min().date()} to {sales["Date"].max().date()})')
print(f'  Orders: {len(orders):,} rows')
print(f'  Web Traffic: {len(web_traffic):,} rows')

# =====================================================================
# FEATURE ENGINEERING
# =====================================================================
print('\n[2] FEATURE ENGINEERING...')

# Sales features
sales['year'] = sales['Date'].dt.year
sales['month'] = sales['Date'].dt.month
sales['quarter'] = sales['Date'].dt.quarter
sales['year_month'] = sales['Date'].dt.to_period('M')
sales['day_of_week'] = sales['Date'].dt.dayofweek
sales['is_weekend'] = sales['day_of_week'].isin([5, 6]).astype(int)
sales['is_tet'] = sales['month'].isin([1, 2]).astype(int)
sales['is_sale_season'] = sales['month'].isin([11, 12]).astype(int)
sales['Date_only'] = sales['Date'].dt.date

# Orders features
orders['year'] = orders['order_date'].dt.year
orders['month'] = orders['order_date'].dt.month
orders['year_month'] = orders['order_date'].dt.to_period('M')
orders['Date_only'] = orders['order_date'].dt.date

# Web traffic features
web_traffic['year'] = web_traffic['date'].dt.year
web_traffic['month'] = web_traffic['date'].dt.month
web_traffic['year_month'] = web_traffic['date'].dt.to_period('M')
web_traffic['Date_only'] = web_traffic['date'].dt.date

print('  Features created')

# =====================================================================
# SECTION A: MONTHLY REVENUE DRIVERS DECOMPOSITION
# =====================================================================
print('\n' + '='*80)
print('SECTION A: MONTHLY REVENUE DRIVERS DECOMPOSITION')
print('='*80)

# A.1 Aggregate monthly metrics
print('\n[A.1] Monthly Metrics Aggregation...')

# Calculate GM first
sales['GM'] = (sales['Revenue'] - sales['COGS']) / sales['Revenue']

# Monthly sales
monthly_sales = sales.groupby(['year', 'month']).agg({
    'Revenue': 'sum',
    'COGS': 'sum',
    'GM': 'mean'
}).reset_index()
monthly_sales['days'] = sales.groupby(['year', 'month']).size().values

# Monthly orders
monthly_orders = orders.groupby(['year', 'month']).agg({
    'order_id': 'count',
    'customer_id': 'nunique'
}).reset_index()
monthly_orders.columns = ['year', 'month', 'orders', 'customers']

# Monthly web traffic
monthly_traffic = web_traffic.groupby(['year', 'month']).agg({
    'sessions': 'sum',
    'unique_visitors': 'sum',
    'page_views': 'sum'
}).reset_index()

# Merge all
monthly = monthly_sales.merge(monthly_orders, on=['year', 'month'], how='left')
monthly = monthly.merge(monthly_traffic, on=['year', 'month'], how='left')

# Add quarter back after merge
monthly['quarter'] = ((monthly['month'] - 1) // 3 + 1).astype(int)

# Calculate derived metrics
monthly['AOV'] = monthly['Revenue'] / monthly['orders']
monthly['sessions_per_order'] = monthly['sessions'] / monthly['orders']
monthly['conversion_rate'] = monthly['orders'] / monthly['sessions'] * 100

# Calculate indices (relative to overall average)
overall_avg_rev = monthly['Revenue'].mean()
overall_avg_orders = monthly['orders'].mean()
overall_avg_aov = monthly['AOV'].mean()
overall_avg_sessions = monthly['sessions'].mean()
overall_avg_conv = monthly['conversion_rate'].mean()

monthly['rev_index'] = monthly['Revenue'] / overall_avg_rev * 100
monthly['orders_index'] = monthly['orders'] / overall_avg_orders * 100
monthly['aov_index'] = monthly['AOV'] / overall_avg_aov * 100
monthly['sessions_index'] = monthly['sessions'] / overall_avg_sessions * 100
monthly['conversion_index'] = monthly['conversion_rate'] / overall_avg_conv * 100

# A.2 Monthly Driver Table
print('\n[A.2] MONTHLY DRIVERS TABLE:')
print('='*100)
print(f'  {"YM":<8} {"Rev":>12} {"Orders":>10} {"AOV":>10} {"Sessions":>10} {"Conv%":>8} {"RevI":>8} {"OrdI":>8} {"AOVI":>8}')
print('  ' + '-'*90)
for _, row in monthly.head(24).iterrows():
    ym = f'{int(row["year"])}-{int(row["month"]):02d}'
    conv = row['conversion_rate'] if pd.notna(row['conversion_rate']) else 0
    print(f'  {ym:<8} {row["Revenue"]/1e6:>11.2f}M {row["orders"]:>10,.0f} {row["AOV"]:>10,.0f} '
          f'{row["sessions"]/1e3:>9.1f}K {conv:>7.1f}% {row["rev_index"]:>7.0f}% {row["orders_index"]:>7.0f}% {row["aov_index"]:>7.0f}%')

# A.3 Quarterly Driver Analysis
print('\n[A.3] QUARTERLY DRIVERS ANALYSIS:')
quarterly = monthly.groupby('quarter').agg({
    'Revenue': 'mean',
    'orders': 'mean',
    'AOV': 'mean',
    'sessions': 'mean',
    'conversion_rate': 'mean'
}).reset_index()

print(f'\n  Quarterly Averages:')
print(f'  {"Q":<4} {"Revenue":>12} {"Orders":>10} {"AOV":>10} {"Sessions":>10} {"Conv%":>8}')
print('  ' + '-'*60)
for _, row in quarterly.iterrows():
    print(f'  {int(row["quarter"]):<4} {row["Revenue"]/1e6:>11.2f}M {row["orders"]:>10,.0f} {row["AOV"]:>10,.0f} '
          f'{row["sessions"]/1e3:>9.1f}K {row["conversion_rate"]:>7.2f}%')

# A.4 Driver Contribution Analysis
print('\n[A.4] DRIVER CONTRIBUTION TO SEASONALITY:')
print('\n  Comparing High Season (Q2) vs Low Season (Q4):')

q2_data = monthly[monthly['quarter'] == 2]
q4_data = monthly[monthly['quarter'] == 4]

q2_rev = q2_data['Revenue'].mean()
q4_rev = q4_data['Revenue'].mean()
q2_ord = q2_data['orders'].mean()
q4_ord = q4_data['orders'].mean()
q2_aov = q2_data['AOV'].mean()
q4_aov = q4_data['AOV'].mean()
q2_ses = q2_data['sessions'].mean()
q4_ses = q4_data['sessions'].mean()
q2_conv = q2_data['conversion_rate'].mean()
q4_conv = q4_data['conversion_rate'].mean()

print(f'\n  {"Metric":<15} {"Q2 (High)":>15} {"Q4 (Low)":>15} {"Diff":>12} {"%Diff":>10}')
print('  ' + '-'*70)
print(f'  {"Revenue":<15} {q2_rev/1e6:>14.2f}M {q4_rev/1e6:>14.2f}M {(q2_rev-q4_rev)/1e6:>+11.2f}M {(q2_rev-q4_rev)/q4_rev*100:>+9.1f}%')
print(f'  {"Orders":<15} {q2_ord:>15,.0f} {q4_ord:>15,.0f} {q2_ord-q4_ord:>+11,.0f} {(q2_ord-q4_ord)/q4_ord*100:>+9.1f}%')
print(f'  {"AOV":<15} {q2_aov:>15,.0f} {q4_aov:>15,.0f} {q2_aov-q4_aov:>+11,.0f} {(q2_aov-q4_aov)/q4_aov*100:>+9.1f}%')
print(f'  {"Sessions":<15} {q2_ses/1e3:>14.1f}K {q4_ses/1e3:>14.1f}K {(q2_ses-q4_ses)/1e3:>+10.1f}K {(q2_ses-q4_ses)/q4_ses*100:>+9.1f}%')
print(f'  {"Conversion%":<15} {q2_conv:>14.2f}% {q4_conv:>14.2f}% {(q2_conv-q4_conv):>+10.2f}% {(q2_conv-q4_conv)/q4_conv*100:>+9.1f}%')

# A.5 Correlation between drivers
print('\n[A.5] CORRELATION BETWEEN DRIVERS:')
corr_cols = ['Revenue', 'orders', 'AOV', 'sessions', 'conversion_rate']
corr_matrix = monthly[corr_cols].corr()
print('\n  Correlation Matrix:')
print(corr_matrix.round(3).to_string())

# =====================================================================
# SECTION B: SEASONALITY STABILITY ANALYSIS
# =====================================================================
print('\n' + '='*80)
print('SECTION B: SEASONALITY STABILITY ANALYSIS')
print('='*80)

# B.1 Monthly Revenue Index by Year
print('\n[B.1] MONTHLY REVENUE INDEX BY YEAR:')
monthly_index_by_year = monthly.pivot(index='month', columns='year', values='rev_index')
print(monthly_index_by_year.round(0).to_string())

# B.2 Seasonality Statistics
print('\n[B.2] SEASONALITY STABILITY STATISTICS:')

monthly_stats = monthly.groupby('month').agg({
    'rev_index': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
monthly_stats.columns = ['month', 'avg_index', 'std', 'min_index', 'max_index', 'n_years']
monthly_stats['cv'] = monthly_stats['std'] / monthly_stats['avg_index'] * 100

# Calculate rank stability (how often is each month in top/bottom 3)
monthly['month_rank'] = monthly.groupby('year')['Revenue'].rank(ascending=False)
rank_counts = monthly.groupby('month')['month_rank'].agg(['mean', 'std']).reset_index()
rank_counts.columns = ['month', 'avg_rank', 'rank_std']
monthly_stats = monthly_stats.merge(rank_counts, on='month')

# Assign confidence
def assign_confidence(row):
    cv = row['cv']
    if cv < 15:
        return 'HIGH'
    elif cv < 25:
        return 'MEDIUM'
    else:
        return 'LOW'

monthly_stats['confidence'] = monthly_stats.apply(assign_confidence, axis=1)

print(f'\n  {"Month":<6} {"Avg Index":>10} {"Std":>8} {"CV%":>6} {"Min":>8} {"Max":>8} {"Avg Rank":>10} {"Conf":>8}')
print('  ' + '-'*75)
for _, row in monthly_stats.iterrows():
    print(f'  {int(row["month"]):<6} {row["avg_index"]:>9.1f}% {row["std"]:>7.1f} {row["cv"]:>5.1f}% '
          f'{row["min_index"]:>7.0f}% {row["max_index"]:>7.0f}% {row["avg_rank"]:>9.1f} {row["confidence"]:>8}')

# B.3 Quarter Stability
print('\n[B.3] QUARTERLY STABILITY:')
quarterly_stats = monthly.groupby('quarter').agg({
    'rev_index': ['mean', 'std', 'min', 'max']
}).reset_index()
quarterly_stats.columns = ['quarter', 'avg_index', 'std', 'min_index', 'max_index']
quarterly_stats['cv'] = quarterly_stats['std'] / quarterly_stats['avg_index'] * 100
quarterly_stats['range'] = quarterly_stats['max_index'] - quarterly_stats['min_index']

print(f'\n  {"Quarter":<8} {"Avg Index":>10} {"Std":>8} {"CV%":>6} {"Min":>8} {"Max":>8} {"Range":>8}')
print('  ' + '-'*60)
for _, row in quarterly_stats.iterrows():
    print(f'  {int(row["quarter"]):<8} {row["avg_index"]:>9.1f}% {row["std"]:>7.1f} {row["cv"]:>5.1f}% '
          f'{row["min_index"]:>7.0f}% {row["max_index"]:>7.0f}% {row["range"]:>7.0f}%')

# B.4 Top/Bottom Month Consistency
print('\n[B.4] TOP/BOTTOM MONTH CONSISTENCY:')
top_months = monthly.groupby('month')['rev_index'].apply(list)
peak_count = sum([1 for idx_list in top_months if any(i >= 140 for i in idx_list)])
trough_count = sum([1 for idx_list in top_months if any(i <= 70 for i in idx_list)])

print(f'\n  Months that were peak (>=140%): {peak_count}/12')
print(f'  Months that were trough (<=70%): {trough_count}/12')

# =====================================================================
# SECTION C: TRAFFIC-REVENUE LAG ANALYSIS
# =====================================================================
print('\n' + '='*80)
print('SECTION C: TRAFFIC-REVENUE LAG ANALYSIS')
print('='*80)

# Merge daily sales with daily traffic
daily_sales = sales.groupby('Date_only')['Revenue'].sum().reset_index()
daily_traffic = web_traffic.groupby('Date_only').agg({
    'sessions': 'sum',
    'unique_visitors': 'sum'
}).reset_index()

daily = daily_sales.merge(daily_traffic, on='Date_only', how='inner')
print(f'\n  Merged daily data: {len(daily)} days')

# Calculate correlations at various lags
lags = [0, 1, 3, 7, 14, 30]
lag_correlations = []

for lag in lags:
    if lag == 0:
        corr = daily['sessions'].corr(daily['Revenue'])
    else:
        # Shift traffic forward (traffic today predicts revenue tomorrow)
        shifted_traffic = daily['sessions'].shift(lag)
        corr = shifted_traffic.corr(daily['Revenue'])
    lag_correlations.append({'lag': lag, 'correlation': corr})

lag_df = pd.DataFrame(lag_correlations)
print('\n[C.1] LAG CORRELATION (Traffic vs Revenue):')
print(f'\n  {"Lag (days)":<12} {"Correlation":>12} {"Interpretation":<30}')
print('  ' + '-'*55)
for _, row in lag_df.iterrows():
    if abs(row['correlation']) < 0.3:
        interp = 'WEAK'
    elif abs(row['correlation']) < 0.6:
        interp = 'MODERATE'
    else:
        interp = 'STRONG'
    print(f'  {row["lag"]:>10}d {row["correlation"]:>11.3f} {interp:<30}')

best_lag = lag_df.loc[lag_df['correlation'].abs().idxmax()]
print(f'\n  Best lag: {int(best_lag["lag"])} days (r = {best_lag["correlation"]:.3f})')

# Cross-correlation at different lags
print('\n[C.2] CROSS-CORRELATION ANALYSIS:')
print('  Note: Correlation does not imply causation')
print('  Traffic may reflect marketing activity, not pure demand')

# =====================================================================
# SECTION D: ENHANCED RECOMMENDATIONS
# =====================================================================
print('\n' + '='*80)
print('SECTION D: EVIDENCE-BASED RECOMMENDATIONS')
print('='*80)

recommendations = [
    {
        'id': 1,
        'category': 'Q2 Planning',
        'rec': 'Pre-position inventory and capacity for Q2 peak',
        'evidence': 'Q2 avg index = 152%, CV = 15% (HIGH stable)',
        'confidence': 'HIGH',
        'risk': 'LOW - 10-year consistent pattern',
        'validation': 'Track weekly stockout rate vs revenue',
        'owner': 'Ops + Merchandising'
    },
    {
        'id': 2,
        'category': 'Q4 Investigation',
        'rec': 'Investigate Q4 root cause before acting',
        'evidence': 'Q4 index = 59-77%, but drivers unknown. Sessions also low (68% avg).',
        'confidence': 'HIGH for weakness, LOW for cause',
        'risk': 'UNKNOWN cause - could be demand, supply, or competition',
        'validation': 'Test: compare conversion rate in Q4 vs other quarters',
        'owner': 'Analytics'
    },
    {
        'id': 3,
        'category': 'Tet Period',
        'rec': 'Do NOT increase marketing spend without investigating fulfillment',
        'evidence': 'Tet (T1-T2) index = 60-81%. Sessions also low (53-78%).',
        'confidence': 'HIGH for low revenue',
        'risk': 'May worsen if logistics is the bottleneck',
        'validation': 'Check order completion rate during Tet vs other periods',
        'owner': 'Ops + Analytics'
    },
    {
        'id': 4,
        'category': 'Traffic Monitoring',
        'rec': 'Use traffic as early indicator for revenue (if lag exists)',
        'evidence': f'Best lag correlation: {int(best_lag["lag"])} days (r = {best_lag["correlation"]:.3f})',
        'confidence': 'MEDIUM - correlation exists but causation unclear',
        'risk': 'Traffic may reflect marketing, not pure demand',
        'validation': 'Monitor daily traffic vs next-day revenue',
        'owner': 'Analytics'
    },
    {
        'id': 5,
        'category': 'Weekend Strategy',
        'rec': 'Low priority - effect size too small',
        'evidence': 'Weekend 9.4% lower, but effect size = 500K/day',
        'confidence': 'LOW for action',
        'risk': 'ROI of optimization unclear',
        'validation': 'Only test if other priorities are addressed',
        'owner': 'Marketing'
    }
]

print('\nRECOMMENDATIONS:')
print('='*80)
for rec in recommendations:
    print(f"\n{rec['id']}. [{rec['category']}] {rec['rec']}")
    print(f"   Evidence: {rec['evidence']}")
    print(f"   Confidence: {rec['confidence']}")
    print(f"   Risk: {rec['risk']}")
    print(f"   Validation: {rec['validation']}")
    print(f"   Owner: {rec['owner']}")

# =====================================================================
# GENERATE CHARTS
# =====================================================================
print('\n' + '='*80)
print('GENERATING CHARTS...')
print('='*80)

# Chart 1: Monthly Revenue Drivers (Normalized Indices)
print('[1/7] Monthly Revenue Drivers - Normalized Indices...')
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Top: Line chart of all indices by month
ax1 = axes[0]
months_order = range(1, 13)
ax1.plot(months_order, monthly.groupby('month')['rev_index'].mean(), 'o-', label='Revenue', linewidth=2, markersize=8)
ax1.plot(months_order, monthly.groupby('month')['orders_index'].mean(), 's--', label='Orders', linewidth=2, markersize=8)
ax1.plot(months_order, monthly.groupby('month')['aov_index'].mean(), '^--', label='AOV', linewidth=2, markersize=8)
ax1.plot(months_order, monthly.groupby('month')['sessions_index'].mean(), 'd-.', label='Sessions', linewidth=2, markersize=8)
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Average')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Index (Overall Average = 100)', fontsize=12)
ax1.set_title('Monthly Revenue Drivers: Revenue, Orders, AOV, Sessions Indices', fontweight='bold', fontsize=14)
ax1.legend(loc='upper right')
ax1.set_xticks(months_order)
ax1.set_ylim([40, 180])

# Bottom: Grouped bar by quarter
ax2 = axes[1]
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(len(quarters))
width = 0.2

rev_by_q = [monthly[monthly['quarter']==q]['rev_index'].mean() for q in [1,2,3,4]]
ord_by_q = [monthly[monthly['quarter']==q]['orders_index'].mean() for q in [1,2,3,4]]
aov_by_q = [monthly[monthly['quarter']==q]['aov_index'].mean() for q in [1,2,3,4]]
ses_by_q = [monthly[monthly['quarter']==q]['sessions_index'].mean() for q in [1,2,3,4]]

bars1 = ax2.bar(x - 1.5*width, rev_by_q, width, label='Revenue', color='steelblue')
bars2 = ax2.bar(x - 0.5*width, ord_by_q, width, label='Orders', color='coral')
bars3 = ax2.bar(x + 0.5*width, aov_by_q, width, label='AOV', color='green')
bars4 = ax2.bar(x + 1.5*width, ses_by_q, width, label='Sessions', color='purple')
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Quarter', fontsize=12)
ax2.set_ylabel('Index (Overall Average = 100)', fontsize=12)
ax2.set_title('Quarterly Driver Indices Comparison', fontweight='bold', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(quarters)
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'monthly_revenue_drivers.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Revenue Driver Indices Heatmap
print('[2/7] Revenue Driver Indices Heatmap...')
driver_pivot = monthly.pivot_table(
    index='month', 
    columns='quarter', 
    values=['rev_index', 'orders_index', 'aov_index', 'sessions_index'],
    aggfunc='mean'
)
driver_pivot.columns = [f'{col[1]}_{col[0].replace("_index", "")}' for col in driver_pivot.columns]

fig, ax = plt.subplots(figsize=(14, 8))
# Flatten for heatmap
plot_data = monthly[['month', 'quarter', 'rev_index', 'orders_index', 'aov_index', 'sessions_index']].copy()
plot_data = plot_data.set_index(['month', 'quarter'])
sns.heatmap(plot_data.T, annot=True, fmt='.0f', cmap='RdYlGn', center=100, ax=ax,
            cbar_kws={'label': 'Index (100 = Average)'})
ax.set_title('Revenue Drivers by Month and Quarter (Indices)', fontweight='bold', fontsize=14)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Driver Index', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'revenue_driver_indices.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Monthly Seasonality Boxplot
print('[3/7] Monthly Seasonality Boxplot...')
fig, ax = plt.subplots(figsize=(14, 6))
monthly.boxplot(column='rev_index', by='month', ax=ax, 
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2))
ax.axhline(y=100, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Revenue Index (100 = Average)', fontsize=12)
ax.set_title('Seasonality Stability: Revenue Index Distribution by Month', fontweight='bold', fontsize=14)
plt.suptitle('')  # Remove default title
ax.set_xticklabels([f'M{i}' for i in range(1, 13)])

# Add count annotations
for i, month in enumerate(range(1, 13), 1):
    count = len(monthly[monthly['month'] == month])
    ax.annotate(f'n={count}', xy=(i, monthly[monthly['month'] == month]['rev_index'].max() + 5),
                ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'monthly_seasonality_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Seasonality Stability Heatmap
print('[4/7] Seasonality Stability Heatmap...')
fig, ax = plt.subplots(figsize=(14, 8))
stability_data = monthly_index_by_year.T
sns.heatmap(stability_data, annot=True, fmt='.0f', cmap='RdYlGn', center=100, ax=ax,
            cbar_kws={'label': 'Revenue Index'})
ax.set_title('Seasonality Stability: Monthly Index by Year', fontweight='bold', fontsize=14)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Year', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'seasonality_stability_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Traffic-Revenue Lag Correlation
print('[5/7] Traffic-Revenue Lag Correlation...')
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in lag_df['correlation']]
bars = ax.bar(lag_df['lag'], lag_df['correlation'], color=colors, edgecolor='black', alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Lag (Days)', fontsize=12)
ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_title('Traffic-Revenue Lag Correlation', fontweight='bold', fontsize=14)
ax.set_xticks(lags)

# Add value labels
for bar, corr in zip(bars, lag_df['correlation']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{corr:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'traffic_revenue_lag_correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Monthly Rank by Year
print('[6/7] Monthly Rank by Year...')
fig, ax = plt.subplots(figsize=(14, 8))
rank_pivot = monthly.pivot(index='month', columns='year', values='month_rank')
sns.heatmap(rank_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax,
            vmin=1, vmax=12, cbar_kws={'label': 'Rank (1=Highest)'})
ax.set_title('Monthly Revenue Rank by Year (1=Highest Revenue)', fontweight='bold', fontsize=14)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Month', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'monthly_rank_by_year.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 7: Driver Contribution Waterfall (Q2 vs Q4)
print('[7/7] Driver Contribution Waterfall...')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Q2 vs Q4 comparison
ax1 = axes[0]
categories = ['Revenue', 'Orders', 'AOV', 'Sessions', 'Conversion']
q2_vals = [q2_rev/1e6, q2_ord/1000, q2_aov/1000, q2_ses/1e6, q2_conv]
q4_vals = [q4_rev/1e6, q4_ord/1000, q4_aov/1000, q4_ses/1e6, q4_conv]

x = np.arange(len(categories))
width = 0.35
bars1 = ax1.bar(x - width/2, q2_vals, width, label='Q2 (High)', color='green', alpha=0.7)
bars2 = ax1.bar(x + width/2, q4_vals, width, label='Q4 (Low)', color='red', alpha=0.7)
ax1.set_xlabel('Metric', fontsize=12)
ax1.set_ylabel('Value (normalized scale)', fontsize=12)
ax1.set_title('Q2 vs Q4: Driver Comparison', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45, ha='right')
ax1.legend()

# Correlation heatmap
ax2 = axes[1]
corr_plot = monthly[['rev_index', 'orders_index', 'aov_index', 'sessions_index', 'conversion_index']].corr()
corr_plot.columns = ['Revenue', 'Orders', 'AOV', 'Sessions', 'Conv%']
corr_plot.index = ['Revenue', 'Orders', 'AOV', 'Sessions', 'Conv%']
sns.heatmap(corr_plot, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax2,
            vmin=-1, vmax=1, square=True)
ax2.set_title('Driver Correlation Matrix', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'driver_contribution.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n[OK] All charts generated!')
print('\nCharts saved to:', OUTPUT_DIR)
print('  - monthly_revenue_drivers.png')
print('  - revenue_driver_indices.png')
print('  - monthly_seasonality_boxplot.png')
print('  - seasonality_stability_heatmap.png')
print('  - traffic_revenue_lag_correlation.png')
print('  - monthly_rank_by_year.png')
print('  - driver_contribution.png')

# =====================================================================
# SUMMARY STATISTICS FOR REPORT
# =====================================================================
print('\n' + '='*80)
print('SUMMARY STATISTICS FOR REPORT')
print('='*80)

print('\n--- KEY FINDINGS ---')
print(f'\n1. Q2 HIGH SEASON DRIVERS:')
print(f'   - Revenue Q2 vs Q4: +{(q2_rev-q4_rev)/q4_rev*100:.0f}%')
print(f'   - Orders driver: +{(q2_ord-q4_ord)/q4_ord*100:.0f}%')
print(f'   - AOV driver: +{(q2_aov-q4_aov)/q4_aov*100:.0f}%')
print(f'   - Sessions driver: +{(q2_ses-q4_ses)/q4_ses*100:.0f}%')

print(f'\n2. SEASONALITY STABILITY:')
high_months = monthly_stats[monthly_stats['confidence'] == 'HIGH']['month'].tolist()
low_months = monthly_stats[monthly_stats['confidence'] == 'LOW']['month'].tolist()
print(f'   - HIGH confidence months: {high_months}')
print(f'   - LOW confidence months: {low_months}')

print(f'\n3. TRAFFIC-REVENUE LAG:')
print(f'   - Best lag: {int(best_lag["lag"])} days (r = {best_lag["correlation"]:.3f})')
print(f'   - Same-day correlation: {lag_df[lag_df["lag"]==0]["correlation"].values[0]:.3f}')

print('\n' + '='*80)
print('ANALYSIS COMPLETE - V3')
print('='*80)
