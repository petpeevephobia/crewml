## Feature Strategy Summary
Proposed four engineered features targeting core automotive valuation drivers:
1. **vehicle_age** (year_delta) aligns calendar year with depreciation mechanics.
2. **mileage_per_year** (ratio) normalizes wear by age, distinguishing high-utilization fleets from personal vehicles.
3. **log_mileage_in_km** (log1p) corrects heavy right-skew in odometer data for stable gradient-based optimization.
4. **pca_fuel_efficiency** (PCA) compresses two correlated consumption metrics into orthogonal components, mitigating multicollinearity while retaining efficiency signals.
All features use robust clipping/epsilon safeguards against extreme outliers observed in the profile (e.g., future years, zero denominators).

## Run stats

- Input rows: 251,079
- Output rows: 251,079
- New columns (5): vehicle_age, mileage_per_year, log_mileage_in_km, pca_fuel_1, pca_fuel_2

## Proposed features

### `vehicle_age`

- **Action:** `year_delta`
- **Justification:** Converts calendar year to vehicle age, which aligns with real-world depreciation curves and warranty expiration timelines more closely than raw year values.
- **Pandas code:**

```python
df['vehicle_age'] = 2023 - df['year'].clip(upper=2023)
```

### `mileage_per_year`

- **Action:** `ratio`
- **Justification:** Annualized usage intensity separates low-mileage older cars from high-mileage newer ones, providing a stronger signal for wear-and-tear than raw odometer readings.
- **Pandas code:**

```python
df['mileage_per_year'] = df['mileage_in_km'] / (df['vehicle_age'] + 1)
```

### `log_mileage_in_km`

- **Action:** `log1p`
- **Justification:** Mileage exhibits heavy right-skew with a long tail of high-kilometer vehicles; log1p stabilizes variance and improves linear model convergence.
- **Pandas code:**

```python
df['log_mileage_in_km'] = np.log1p(df['mileage_in_km'].clip(lower=0))
```

### `pca_fuel_efficiency`

- **Action:** `pca`
- **Justification:** Volume and mass fuel consumption are highly collinear; PCA orthogonalizes them into efficiency and density components, reducing multicollinearity while preserving predictive variance.
- **Pandas code:**

```python
from sklearn.decomposition import PCA\nX = df[['fuel_consumption_l_100km','fuel_consumption_g_km']].fillna(df[['fuel_consumption_l_100km','fuel_consumption_g_km']].median()).clip(upper=300)\ncomps = PCA(n_components=2).fit_transform(X)\ndf[['pca_fuel_1','pca_fuel_2']] = comps
```


## Steps applied (code)

- year_delta vehicle_age = 2023 - year
- ratio mileage_per_year = mileage_in_km / (vehicle_age + 1.0)
- log1p log_mileage_in_km from mileage_in_km
- pca on ['fuel_consumption_l_100km', 'fuel_consumption_g_km'] → pca_fuel_1..2 (explained variance ratios: [0.558, 0.442])
