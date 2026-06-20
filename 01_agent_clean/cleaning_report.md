## Data Quality Report

### Findings
- **Wrong Dtypes:** `price_in_euro`, `power_kw`, and `power_ps` are stored as `object` strings but represent continuous numeric values.
- **Sentinel Values:** `fuel_consumption_l_100km` uses empty strings, while `fuel_consumption_g_km` uses `"- (g/km)"` as placeholders. Both columns include unit suffixes and European-style decimal commas.
- **Outliers:** `mileage_in_km` has a maximum value of 3,800,000 km, which is ~48σ above the mean and physically implausible for standard passenger vehicles.
- **Nulls:** Minor missingness in `color` (166), `registration_date` (4), `power_kw` (134), `power_ps` (129), `mileage_in_km` (152), and `offer_description` (1) remains as `NaN` post-cleaning.

### Applied Transformations
1. Replaced sentinel placeholders (`""` and `"- (g/km)"`) with `null`.
2. Stripped units and converted European comma decimals to numeric in fuel consumption columns.
3. Cast `price_in_euro`, `power_kw`, and `power_ps` from object to float.
4. Applied IQR-based clipping (factor=3) to `mileage_in_km` to cap extreme outliers without dropping rows.

All changes preserve the original row count and prioritize in-place value correction.

## Run stats

- Input rows: 251,079
- Output rows: 251,079

## Issues identified

- **price_in_euro** (dtype, high): Stored as object (string), should be numeric
- **power_kw** (dtype, high): Stored as object (string), should be numeric
- **power_ps** (dtype, high): Stored as object (string), should be numeric
- **fuel_consumption_l_100km** (sentinel, high): Contains empty strings, European commas, and units; 10.7% effectively missing
- **fuel_consumption_g_km** (sentinel, medium): Contains '- (g/km)' placeholder text instead of nulls
- **mileage_in_km** (outlier, high): Max value 3,800,000 km is physically unrealistic and ~48 standard deviations above mean
- **color** (null, low): 166 missing values (0.07%)
- **registration_date** (null, low): 4 missing values (0.00%)
- **offer_description** (null, low): 1 missing value (0.00%)

## Steps applied (code)

- replace_sentinels on fuel_consumption_g_km: 35809 values → missing
- replace_sentinels on fuel_consumption_l_100km: 0 values → missing
- parse_european_decimal on fuel_consumption_l_100km: nulls 26873 → 26922
- parse_european_decimal on fuel_consumption_g_km: nulls 35809 → 36770
- astype on price_in_euro → float
- astype on power_kw → float
- astype on power_ps → float
- clip_outliers on mileage_in_km (IQR×3.0): 336 values capped
