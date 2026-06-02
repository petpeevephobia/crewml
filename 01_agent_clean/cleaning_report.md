## Data Quality Summary

**Dataset:** 251,079 rows × 15 columns

### Identified Issues
- **Type Mismatch:** Key numeric columns (`year`, `price_in_euro`, `power_kw`, `power_ps`) and fuel consumption metrics are stored as strings/objects.
- **Formatting & Sentinels:** Fuel consumption columns use European decimal commas with attached units. `fuel_consumption_g_km` uses `- (g/km)` as a missing value sentinel. `fuel_consumption_l_100km` contains ~10.7% nulls.
- **Outliers:** `mileage_in_km` has an extreme maximum of 3,800,000 km, which is physically impossible for passenger cars and will skew modeling.
- **Low-Frequency Nulls:** Minor missingness in `color`, `registration_date`, `power_kw`, `power_ps`, and `mileage_in_km` (<0.1% each).

### Applied Fixes
1. Replaced `- (g/km)` sentinel in `fuel_consumption_g_km` with `null`.
2. Parsed European decimals and stripped unit suffixes from both fuel consumption columns, converting them to numeric types.
3. Cast `year`, `power_kw`, and `power_ps` to `int`, and `price_in_euro` to `float`.
4. Clipped extreme outliers in `mileage_in_km` using the IQR method (factor=3).
5. Preserved low-frequency nulls as-is to maximize data retention. No rows were dropped.

## Run stats

- Input rows: 251,079
- Output rows: 251,079

## Issues identified

- **fuel_consumption_g_km** (sentinel, high): Contains '- (g/km)' indicating missing or unavailable data
- **fuel_consumption_l_100km** (dtype, high): Stored as object with European decimal commas and unit suffixes (e.g., '10,9 l/100 km')
- **fuel_consumption_g_km** (dtype, high): Stored as object with unit suffixes and sentinel values
- **year** (dtype, medium): Stored as object, should be integer
- **price_in_euro** (dtype, high): Stored as object, should be numeric
- **power_kw** (dtype, medium): Stored as object, should be integer
- **power_ps** (dtype, medium): Stored as object, should be integer
- **mileage_in_km** (outlier, high): Maximum value 3,800,000 km is physically unrealistic and likely a data entry error
- **fuel_consumption_l_100km** (null, medium): Contains ~10.7% missing values

## Steps applied (code)

- replace_sentinels on fuel_consumption_g_km: 35809 values → missing
- parse_european_decimal on fuel_consumption_l_100km: nulls 26873 → 26922
- parse_european_decimal on fuel_consumption_g_km: nulls 35809 → 36770
- astype on year → int
- astype on price_in_euro → float
- astype on power_kw → int
- astype on power_ps → int
- clip_outliers on mileage_in_km (IQR×3.0): 336 values capped
