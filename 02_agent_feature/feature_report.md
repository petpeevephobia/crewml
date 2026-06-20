## Feature Engineering Strategy

This proposal transforms raw automotive attributes into physically meaningful and statistically robust predictors for price regression. The strategy follows four core principles:

1. **Temporal Depreciation Modeling**: `vehicle_age` converts calendar years into a continuous depreciation timeline, aligning directly with industry valuation curves and age-based discounting.
2. **Distribution Normalization**: `log_mileage_in_km` corrects the heavy right-tail of the odometer data. By applying `log1p`, we ensure tree-based splits are more balanced and linear models are not disproportionately influenced by extreme mileage outliers.
3. **Usage Intensity Normalization**: `mileage_per_year` builds directly on the age feature to calculate annualized driving distance. This isolates heavily utilized vehicles from low-mileage collector examples, capturing wear-and-tear dynamics independent of absolute age.
4. **Redundancy & Multicollinearity Reduction**: `power_kw` and `power_ps` convey identical information. Applying PCA extracts a single latent performance factor (`pca_power_1`), reducing feature dimensionality, preventing variance inflation, and improving model convergence stability.

All transformations are designed to execute sequentially after basic null imputation, ensuring compatibility with both gradient-boosted trees and regularized linear pipelines.

## Run stats

- Input rows: 251,079
- Output rows: 251,079
- New columns (4): vehicle_age, log_mileage_in_km, mileage_per_year, pca_power_1

## Proposed features

### `vehicle_age`

- **Action:** `year_delta`
- **Justification:** Captures linear depreciation, which is the dominant pricing factor in the secondary automotive market. Age provides a continuous proxy for technological obsolescence, warranty expiry, and accumulated mechanical wear.
- **Pandas code:**

```python
df['vehicle_age'] = 2024 - df['year']
```

### `log_mileage_in_km`

- **Action:** `log1p`
- **Justification:** Odometer readings are heavily right-skewed with extreme high-end values that distort distance metrics and linear models. The log1p transform compresses the tail, stabilizes variance, and linearizes the non-linear price decay associated with distance.
- **Pandas code:**

```python
df['log_mileage_in_km'] = np.log1p(df['mileage_in_km'])
```

### `mileage_per_year`

- **Action:** `ratio`
- **Justification:** Normalizes total distance traveled by vehicle age to distinguish high-intensity daily drivers from garage-kept or city-only cars. This usage intensity signal often predicts resale value more accurately than raw mileage or age alone.
- **Pandas code:**

```python
df['mileage_per_year'] = df['mileage_in_km'] / (df['vehicle_age'] + 1e-6)
```

### `pca_engine_power`

- **Action:** `pca`
- **Justification:** Engine output is duplicated across kilowatts and metric horsepower, which are linearly dependent (1 kW ≈ 1.341 PS). PCA condenses these redundant features into a single orthogonal component, mitigating multicollinearity while preserving the underlying performance variance for downstream models.
- **Pandas code:**

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
df['pca_power_1'] = pca.fit_transform(df[['power_kw', 'power_ps']].fillna(method='ffill'))[:, 0]
```


## Steps applied (code)

- year_delta vehicle_age = 2024 - year
- log1p log_mileage_in_km from mileage_in_km
- ratio mileage_per_year = mileage_in_km / (vehicle_age + 1e-06)
- pca on ['power_kw', 'power_ps'] → pca_power_1..1 (explained variance ratios: [0.971])
