### Feature Engineering Strategy
This pipeline introduces four high-impact transformations tailored for automotive price prediction:

1. **Temporal & Usage Dynamics**: `vehicle_age` establishes the baseline depreciation timeline, while `mileage_per_year` refines it by quantifying annual wear intensity. Together, they resolve the ambiguity of a high-mileage young car versus a low-mileage old car.
2. **Distribution Normalization**: `log_mileage` and `log_power` apply `log1p` to compress heavy right-tails. This stabilizes variance, mitigates outlier leverage, and aligns feature scales with tree-splitting heuristics and linear assumptions.
3. **Numerical Safety**: All operations use safe division (`epsilon=1e-6`) and `log1p` to gracefully handle zero-values or missing casts, ensuring robust integration into downstream cross-validation and training loops.

## Run stats

- Input rows: 251,079
- Output rows: 251,079
- New columns (4): vehicle_age, mileage_per_year, log_mileage, log_power

## Proposed features

### `vehicle_age`

- **Action:** `year_delta`
- **Justification:** Computes the chronological age of the vehicle, directly capturing the primary depreciation driver. Newer cars retain value better, while age interacts non-linearly with mechanical reliability and market demand.
- **Pandas code:**

```python
df['vehicle_age'] = 2023 - pd.to_numeric(df['year'], errors='coerce')
```

### `mileage_per_year`

- **Action:** `ratio`
- **Justification:** Normalizes cumulative wear by dividing mileage by age. This isolates usage intensity (e.g., high-km daily commuter vs. low-km collector car), a stronger pricing signal than raw mileage alone.
- **Pandas code:**

```python
df['mileage_per_year'] = df['mileage_in_km'] / (df['vehicle_age'] + 1e-6)
```

### `log_mileage`

- **Action:** `log1p`
- **Justification:** Addresses the heavy right-skew in odometer data. Log-transformation linearizes the relationship with log-priced targets and reduces the disproportionate influence of ultra-high-mileage outliers on model training.
- **Pandas code:**

```python
df['log_mileage'] = np.log1p(df['mileage_in_km'])
```

### `log_power`

- **Action:** `log1p`
- **Justification:** Compresses the wide dynamic range of engine power (1kW to >2000kW). Stabilizes variance and improves gradient-based or distance-based model performance while preserving monotonicity with price.
- **Pandas code:**

```python
df['log_power'] = np.log1p(df['power_kw'])
```


## Steps applied (code)

- year_delta vehicle_age = 2023 - year
- ratio mileage_per_year = mileage_in_km / (vehicle_age + 1e-06)
- log1p log_mileage from mileage_in_km
- log1p log_power from power_kw
