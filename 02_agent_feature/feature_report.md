# Feature Engineering Strategy for Vehicle Price Prediction

This pipeline introduces four high-value derived features designed to capture core depreciation mechanics and vehicle utility profiles:

1. **Vehicle Age (`vehicle_age`)**: The foundational feature for used car pricing. Linearly models time-based depreciation and anchors all valuation curves.
2. **Mileage Intensity (`mileage_per_year`)**: By dividing total mileage by age, we isolate usage intensity. This is critical for separating ex-fleet or heavily driven cars from well-preserved personal vehicles, resolving multicollinearity between age and raw odometer readings.
3. **Log-Transformed Mileage (`log_mileage`)**: Addresses the pronounced right-skew in odometer data. The log transformation compresses extreme outliers while preserving fine-grained discrimination among low-mileage vehicles, which typically command significant price premiums.
4. **Power-Efficiency Ratio (`power_efficiency_ratio`)**: Divides engine power by fuel consumption to create a continuous proxy for vehicle class and engineering focus. High ratios signal performance or luxury models (higher price tier), while low ratios indicate economy-focused engineering, providing a strong interaction signal for both linear and tree-based models.

These features are engineered sequentially to maximize information gain, maintain numerical stability, and provide interpretable economic signals for downstream pricing algorithms.

## Run stats

- Input rows: 251,079
- Output rows: 251,079
- New columns (0): (none)

## Proposed features

### `vehicle_age`

- **Action:** `{'action': 'year_delta', 'column': 'year', 'reference_year': 2023}`
- **Justification:** Directly captures chronological depreciation, which is the strongest linear driver of used car market value.
- **Pandas code:**

```python
df['vehicle_age'] = 2023 - df['year']
```

### `mileage_per_year`

- **Action:** `{'action': 'ratio', 'column_a': 'mileage_in_km', 'column_b': 'vehicle_age', 'epsilon': 1e-06}`
- **Justification:** Normalizes total distance traveled by age to distinguish heavily used fleet vehicles from lightly owned private cars of identical model years, reducing collinearity between age and raw mileage.
- **Pandas code:**

```python
df['mileage_per_year'] = df['mileage_in_km'] / (df['vehicle_age'] + 1e-6)
```

### `log_mileage`

- **Action:** `{'action': 'log1p', 'column': 'mileage_in_km'}`
- **Justification:** Compresses the heavy right-skew of odometer readings, stabilizing variance and improving model sensitivity to low-mileage premiums while mitigating the impact of extreme high-mileage outliers.
- **Pandas code:**

```python
df['log_mileage'] = np.log1p(df['mileage_in_km'])
```

### `power_efficiency_ratio`

- **Action:** `{'action': 'ratio', 'column_a': 'power_kw', 'column_b': 'fuel_consumption_l_100km', 'epsilon': 1e-06}`
- **Justification:** Captures the engineering trade-off between engine output and fuel economy, providing a robust proxy for vehicle class that helps models differentiate high-value performance/luxury models from economy cars.
- **Pandas code:**

```python
df['power_efficiency_ratio'] = df['power_kw'] / (df['fuel_consumption_l_100km'] + 1e-6)
```


## Steps applied (code)

- skip unknown action: {'action': 'year_delta', 'column': 'year', 'reference_year': 2023}
- skip unknown action: {'action': 'ratio', 'column_a': 'mileage_in_km', 'column_b': 'vehicle_age', 'epsilon': 1e-06}
- skip unknown action: {'action': 'log1p', 'column': 'mileage_in_km'}
- skip unknown action: {'action': 'ratio', 'column_a': 'power_kw', 'column_b': 'fuel_consumption_l_100km', 'epsilon': 1e-06}
