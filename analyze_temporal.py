import pandas as pd
import numpy as np

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# Parse timestamps properly - they are in H:M format
def parse_time(ts):
    parts = str(ts).split(':')
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

train[['hour', 'minute']] = train['timestamp'].apply(lambda x: pd.Series(parse_time(x)))
test[['hour', 'minute']] = test['timestamp'].apply(lambda x: pd.Series(parse_time(x)))

print("=== TRAIN TIMESTAMP DISTRIBUTION ===")
print(f"Unique timestamps: {train['timestamp'].nunique()}")
print(f"Hour range: {train['hour'].min()} to {train['hour'].max()}")
print(f"Minute values: {sorted(train['minute'].unique())}")

# Create a time slot index  
train['time_slot'] = train['hour'] * 4 + train['minute'] // 15
test['time_slot'] = test['hour'] * 4 + test['minute'] // 15

print(f"\nTrain time_slot range: {train['time_slot'].min()} to {train['time_slot'].max()}")
print(f"Test time_slot range: {test['time_slot'].min()} to {test['time_slot'].max()}")

# Day 48 time distribution
print("\n=== DAY 48 TIME DISTRIBUTION ===")
d48 = train[train['day'] == 48]
print(f"Time slots: {sorted(d48['time_slot'].unique())}")
print(f"Count by time slot:")
for ts in sorted(d48['time_slot'].unique()):
    n = (d48['time_slot'] == ts).sum()
    print(f"  Slot {ts} ({ts//4}:{(ts%4)*15:02d}): n={n}")

# Day 49 time distribution  
print("\n=== DAY 49 TRAIN TIME DISTRIBUTION ===")
d49_train = train[train['day'] == 49]
print(f"Time slots: {sorted(d49_train['time_slot'].unique())}")
for ts in sorted(d49_train['time_slot'].unique()):
    n = (d49_train['time_slot'] == ts).sum()
    print(f"  Slot {ts} ({ts//4}:{(ts%4)*15:02d}): n={n}")

# Day 49 test time distribution
print("\n=== DAY 49 TEST TIME DISTRIBUTION ===")
d49_test = test[test['day'] == 49]
print(f"Time slots: {sorted(d49_test['time_slot'].unique())}")
for ts in sorted(d49_test['time_slot'].unique())[:5]:
    n = (d49_test['time_slot'] == ts).sum()
    print(f"  Slot {ts} ({ts//4}:{(ts%4)*15:02d}): n={n}")
print("  ...")
for ts in sorted(d49_test['time_slot'].unique())[-5:]:
    n = (d49_test['time_slot'] == ts).sum()
    print(f"  Slot {ts} ({ts//4}:{(ts%4)*15:02d}): n={n}")

# Check if test has days beyond 49
print("\n=== TEST DAY DISTRIBUTION ===")
print(test['day'].value_counts())

# Demand by hour (critical for understanding patterns)
print("\n=== DEMAND BY HOUR (TRAIN) ===")
for h in sorted(train['hour'].unique()):
    mask = train['hour'] == h
    print(f"  Hour {h:2d}: mean={train.loc[mask,'demand'].mean():.4f}, median={train.loc[mask,'demand'].median():.4f}, n={mask.sum()}")

# Demand by geohash prefix
print("\n=== DEMAND BY GEOHASH PREFIX (top 10) ===")
train['gh3'] = train['geohash'].str[:3]
for gh, grp in train.groupby('gh3')['demand'].agg(['mean','std','count']).sort_values('mean', ascending=False).head(10).iterrows():
    print(f"  {gh}: mean={grp['mean']:.4f}, std={grp['std']:.4f}, n={int(grp['count'])}")

# Demand by geohash (top 10 highest demand)
print("\n=== TOP 10 GEOHASHES BY MEAN DEMAND ===")
geo_demand = train.groupby('geohash')['demand'].agg(['mean','std','count']).sort_values('mean', ascending=False)
print(geo_demand.head(10).to_string())

# Demand by Road Type
print("\n=== DEMAND BY ROADTYPE ===")
for rt, grp in train.groupby('RoadType')['demand'].agg(['mean','std','count']).iterrows():
    print(f"  {rt}: mean={grp['mean']:.4f}, std={grp['std']:.4f}, n={int(grp['count'])}")

# Demand by Weather
print("\n=== DEMAND BY WEATHER ===")
for w, grp in train.groupby('Weather')['demand'].agg(['mean','std','count']).iterrows():
    print(f"  {w}: mean={grp['mean']:.4f}, std={grp['std']:.4f}, n={int(grp['count'])}")

# Demand by LargeVehicles
print("\n=== DEMAND BY LARGE VEHICLES ===")
for lv, grp in train.groupby('LargeVehicles')['demand'].agg(['mean','std','count']).iterrows():
    print(f"  {lv}: mean={grp['mean']:.4f}, std={grp['std']:.4f}, n={int(grp['count'])}")

# Demand by NumberofLanes
print("\n=== DEMAND BY LANES ===")
for nl, grp in train.groupby('NumberofLanes')['demand'].agg(['mean','std','count']).iterrows():
    print(f"  {nl} lanes: mean={grp['mean']:.4f}, std={grp['std']:.4f}, n={int(grp['count'])}")

# Demand by Landmarks
print("\n=== DEMAND BY LANDMARKS ===")
for lm, grp in train.groupby('Landmarks')['demand'].agg(['mean','std','count']).iterrows():
    print(f"  {lm}: mean={grp['mean']:.4f}, std={grp['std']:.4f}, n={int(grp['count'])}")

# Temperature vs Demand correlation
print(f"\n=== TEMP-DEMAND CORRELATION ===")
valid_temp = train.dropna(subset=['Temperature'])
print(f"  Pearson: {valid_temp['Temperature'].corr(valid_temp['demand']):.4f}")

# Check for geohash-hour interaction patterns
print("\n=== GEOHASH-HOUR DEMAND PATTERNS (sample geohash) ===")
sample_gh = train['geohash'].value_counts().index[0]
gh_data = train[train['geohash'] == sample_gh]
print(f"Geohash: {sample_gh}")
for h in sorted(gh_data['hour'].unique()):
    mask = gh_data['hour'] == h
    print(f"  Hour {h:2d}: mean={gh_data.loc[mask,'demand'].mean():.4f}, n={mask.sum()}")

# Critical: understand the chronological ordering
print("\n=== CHRONOLOGICAL ORDERING ===")
train['time_order'] = train['day'] * 96 + train['time_slot']
print(f"Min time_order: {train['time_order'].min()}")
print(f"Max time_order: {train['time_order'].max()}")
print(f"Unique time_orders: {train['time_order'].nunique()}")

test['time_order'] = test['day'] * 96 + test['time_slot']
print(f"Test min time_order: {test['time_order'].min()}")
print(f"Test max time_order: {test['time_order'].max()}")

# Check if train day49 overlaps with test day49
train_d49_slots = set(train[train['day']==49]['time_slot'].unique())
test_d49_slots = set(test[test['day']==49]['time_slot'].unique())
print(f"\nTrain day49 time slots: {sorted(train_d49_slots)}")
print(f"Test day49 time slots (first 10): {sorted(test_d49_slots)[:10]}")
print(f"Overlap: {sorted(train_d49_slots & test_d49_slots)}")
