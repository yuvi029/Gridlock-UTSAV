import pandas as pd
import numpy as np

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
sample = pd.read_csv('dataset/sample_submission.csv')

print("=== TRAIN INFO ===")
print(f"Shape: {train.shape}")
print(train.dtypes)

print("\n=== HEAD ===")
print(train.head(5).to_string())

print("\n=== TEST HEAD ===")
print(test.head(5).to_string())

print("\n=== SAMPLE SUBMISSION ===")
print(sample.to_string())

print("\n=== DESCRIBE ===")
print(train.describe().to_string())

print("\n=== NULLS (TRAIN) ===")
print(train.isnull().sum())

print("\n=== NULLS (TEST) ===")
print(test.isnull().sum())

print("\n=== UNIQUE VALUES ===")
for c in train.columns:
    print(f"  {c}: {train[c].nunique()} unique")

print("\n=== DEMAND PERCENTILES ===")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {train['demand'].quantile(p/100):.4f}")

print(f"\n=== DEMAND STATS ===")
print(f"  Mean: {train['demand'].mean():.4f}")
print(f"  Std: {train['demand'].std():.4f}")
print(f"  Min: {train['demand'].min():.4f}")
print(f"  Max: {train['demand'].max():.4f}")
print(f"  Skew: {train['demand'].skew():.4f}")

print("\n=== GEOHASH SAMPLES ===")
print(train['geohash'].value_counts().head(15))
print(f"  Geohash lengths: {train['geohash'].str.len().value_counts().to_dict()}")

print("\n=== DAY VALUES ===")
days = sorted(train['day'].unique())
print(f"  Train days: {days}")
print(f"  Num train days: {len(days)}")
test_days = sorted(test['day'].unique())
print(f"  Test days: {test_days}")
print(f"  Num test days: {len(test_days)}")
print(f"  Overlap: {set(days) & set(test_days)}")

print("\n=== TIMESTAMP RANGE ===")
print(f"  Train: {train['timestamp'].min()} to {train['timestamp'].max()}")
print(f"  Test: {test['timestamp'].min()} to {test['timestamp'].max()}")

print("\n=== WEATHER ===")
print(train['Weather'].value_counts(dropna=False))

print("\n=== ROADTYPE ===")
print(train['RoadType'].value_counts(dropna=False))

print("\n=== LARGE VEHICLES ===")
print(train['LargeVehicles'].value_counts(dropna=False))

print("\n=== LANDMARKS ===")
print(train['Landmarks'].value_counts(dropna=False))

print("\n=== NUMBER OF LANES ===")
print(train['NumberofLanes'].value_counts(dropna=False))

print("\n=== TEMPERATURE ===")
print(f"  Null count: {train['Temperature'].isnull().sum()}")
print(f"  Range: {train['Temperature'].min()} to {train['Temperature'].max()}")

# Check if day is truly chronological
print("\n=== DAY-TIMESTAMP RELATIONSHIP ===")
train_ts = pd.to_datetime(train['timestamp'])
for d in sorted(train['day'].unique())[:5]:
    mask = train['day'] == d
    ts_min = train_ts[mask].min()
    ts_max = train_ts[mask].max()
    print(f"  Day {d}: {ts_min} to {ts_max} (n={mask.sum()})")

# Check demand distribution per day
print("\n=== DEMAND BY DAY ===")
for d in sorted(train['day'].unique()):
    mask = train['day'] == d
    print(f"  Day {d}: mean={train.loc[mask,'demand'].mean():.2f}, std={train.loc[mask,'demand'].std():.2f}, n={mask.sum()}")

# Check geohash overlaps between train/test
train_geo = set(train['geohash'].unique())
test_geo = set(test['geohash'].unique())
print(f"\n=== GEOHASH OVERLAP ===")
print(f"  Train unique geohashes: {len(train_geo)}")
print(f"  Test unique geohashes: {len(test_geo)}")
print(f"  Common: {len(train_geo & test_geo)}")
print(f"  Test-only (unseen): {len(test_geo - train_geo)}")

# Check if Index is sequential and unique
print("\n=== INDEX ===")
print(f"  Train Index range: {train['Index'].min()} to {train['Index'].max()}")
print(f"  Test Index range: {test['Index'].min()} to {test['Index'].max()}")
print(f"  Train Index unique: {train['Index'].nunique()}")
print(f"  Test Index unique: {test['Index'].nunique()}")
