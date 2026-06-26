This feature engineering strategy focuses on capturing depreciation dynamics, usage intensity, and distributional properties critical for vehicle pricing models. First, `vehicle_age` is derived from the manufacturing year to serve as a direct, interpretable proxy for chronological depreciation. Second, `mileage_per_year` normalizes odometer readings by age, isolating high-wear vehicles from well-preserved ones and providing a clearer signal of actual asset condition. Third, `log_mileage` linearizes the heavy-tailed mileage distribution, improving stability for both gradient-based optimizers and tree-based splits. Finally, `age_bucket` discretizes age into five equal-width bands, enabling the model to learn non-linear price drops across distinct vehicle lifecycle stages without requiring complex interaction terms. Together, these derived features enhance predictive signal while maintaining computational efficiency and interpretability.

## Run stats

- Input rows: 251,079
- Output rows: 251,079
- New columns (4): vehicle_age, mileage_per_year, log_mileage, age_bucket

## Proposed features

### `vehicle_age`

- **Action:** `year_delta`
- **Justification:** Directly quantifies the chronological age of the vehicle, which is the primary driver of depreciation and strongly correlates with market price.
- **Pandas code:**

```python
df['vehicle_age'] = 2023 - df['year']
```

### `mileage_per_year`

- **Action:** `ratio`
- **Justification:** Measures usage intensity by normalizing total distance traveled by the car's age, helping the model distinguish between lightly driven older cars and heavily used newer ones.
- **Pandas code:**

```python
df['mileage_per_year'] = df['mileage_in_km'] / (df['vehicle_age'] + 1e-6)
```

### `log_mileage`

- **Action:** `log1p`
- **Justification:** Applies a logarithmic transform to right-skewed mileage data, linearizing the relationship between distance traveled and price depreciation for better model stability and gradient flow.
- **Pandas code:**

```python
df['log_mileage'] = np.log1p(df['mileage_in_km'])
```

### `age_bucket`

- **Action:** `bin`
- **Justification:** Discretizes vehicle age into equal-width categories, allowing tree-based and linear models to capture non-linear depreciation phases (e.g., steep initial drop vs. gradual decline) without manual threshold tuning.
- **Pandas code:**

```python
df['age_bucket'] = pd.cut(df['vehicle_age'], bins=5, labels=False).astype(int)
```


## Steps applied (code)

- year_delta vehicle_age = 2023 - year
- ratio mileage_per_year = mileage_in_km / (vehicle_age + 1e-06)
- log1p log_mileage from mileage_in_km
- bin age_bucket from vehicle_age (5 bins)
