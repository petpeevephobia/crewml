## Feature Engineering Strategy
Proposed five derived features to strengthen price prediction signals in a large tabular vehicle dataset:

1. **Age & Usage Intensity:** `vehicle_age` linearizes the depreciation timeline, while `mileage_per_year` (a ratio of total mileage to age) isolates heavy vs. light usage patterns that raw mileage alone cannot distinguish.
2. **Skew Correction & Outlier Mitigation:** `log1p` transforms applied to `mileage_in_km` and `fuel_consumption_l_100km` handle severe right-skew and data-entry anomalies, yielding more Gaussian-like inputs for parametric models.
3. **Latent Signal Extraction:** A 2-component PCA on mileage and fuel-consumption columns compresses correlated efficiency/wear metrics into orthogonal axes (`pca_ue_1`, `pca_ue_2`), reducing multicollinearity and robustly handling ~10–15% missing consumption values via median imputation.

These transformations collectively improve model stability, accelerate training for distance/linear algorithms, and provide clearer interpretability for pricing drivers.

## Run stats

- Input rows: 251,079
- Output rows: 251,079
- New columns (6): vehicle_age, mileage_per_year, log_mileage_in_km, log_fuel_consumption_l_100km, pca_ue_1, pca_ue_2

## Proposed features

### `vehicle_age`

- **Action:** `year_delta`
- **Justification:** Converts calendar year into an age metric that directly correlates with depreciation curves. Clipping negative values handles obvious data-entry typos (e.g., future years).
- **Pandas code:**

```python
df['vehicle_age'] = (2023 - df['year']).clip(lower=0)
```

### `mileage_per_year`

- **Action:** `ratio`
- **Justification:** Captures driving intensity rather than raw distance. A low-age, high-mileage car implies heavy commercial use, while high-age, low-mileage suggests light personal use, both strongly impacting valuation.
- **Pandas code:**

```python
df['mileage_per_year'] = df['mileage_in_km'] / (df['vehicle_age'] + 1e-3)
```

### `log_mileage_in_km`

- **Action:** `log1p`
- **Justification:** Mileage distribution is heavily right-skewed with a long tail. log1p compresses extreme values, stabilizes variance, and improves performance for linear and gradient-based models.
- **Pandas code:**

```python
df['log_mileage_in_km'] = np.log1p(df['mileage_in_km'].clip(lower=0))
```

### `log_fuel_consumption_l_100km`

- **Action:** `log1p`
- **Justification:** Fuel consumption contains outliers and data-entry errors (e.g., max 2023). The log transform normalizes the distribution and reduces the leverage of anomalous records on model coefficients.
- **Pandas code:**

```python
df['log_fuel_consumption_l_100km'] = np.log1p(df['fuel_consumption_l_100km'].clip(lower=0))
```

### `pca_usage_efficiency`

- **Action:** `pca`
- **Justification:** Mileage and dual consumption metrics are moderately correlated. PCA projects them into two orthogonal components capturing 'overall usage load' and 'emission efficiency', reducing multicollinearity while preserving signal across rows with partial missingness.
- **Pandas code:**

```python
from sklearn.decomposition import PCA
cols = ['mileage_in_km', 'fuel_consumption_l_100km', 'fuel_consumption_g_km']
X = df[cols].fillna(df[cols].median())
df[['pca_ue_1', 'pca_ue_2']] = PCA(n_components=2).fit_transform(X)
```


## Steps applied (code)

- year_delta vehicle_age = 2023 - year
- ratio mileage_per_year = mileage_in_km / (vehicle_age + 0.001)
- log1p log_mileage_in_km from mileage_in_km
- log1p log_fuel_consumption_l_100km from fuel_consumption_l_100km
- pca on ['mileage_in_km', 'fuel_consumption_l_100km', 'fuel_consumption_g_km'] → pca_ue_1..2 (explained variance ratios: [0.379, 0.34])
