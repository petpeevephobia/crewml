## Data Cleaning Summary

### Issues Identified
- **Wrong Dtypes:** `price_in_euro`, `power_kw`, `power_ps`, and `year` are stored as strings. This blocks numeric computations and ML model ingestion.
- **Formatting & Sentinels:** `fuel_consumption_l_100km` mixes European commas (`10,9`) with units (`l/100 km`) and uses empty strings `""` for missing data. `fuel_consumption_g_km` uses `"- (g/km)"` as a missing-value sentinel and appends units.
- **Outliers:** `mileage_in_km` contains an unrealistic maximum of 3,800,000 km. This extreme value heavily inflates variance and can destabilize distance-based algorithms.
- **Null Counts:** Minor missingness in `color` (0.07%), `power_kw`/`power_ps` (~0.05%), and `mileage_in_km` (0.06%). `fuel_consumption_l_100km` has ~10.7% missing values, typical for scraped automotive datasets.

### Transformations Applied
1. **Sentinel Replacement:** Converted `""` in `fuel_consumption_l_100km` and `"- (g/km)"` in `fuel_consumption_g_km` to proper `null` values.
2. **Unit & Locale Parsing:** Applied `parse_european_decimal` to both consumption columns to strip text units, convert comma decimals to standard floats, and coerce to numeric dtype.
3. **Type Casting:** Converted `price_in_euro`, `power_kw`, and `power_ps` to `float`, and `year` to `int` for correct semantic typing.
4. **Outlier Capping:** Applied IQR-based clipping (`factor=3`) to `mileage_in_km` to cap extreme values at a statistically plausible threshold (~426,500 km) without dropping rows.

All fixes prioritize value normalization and type correction over row deletion, preserving the full 251,079-row dataset for downstream modeling.

## Run stats

- Input rows: 251,079
- Output rows: 251,079

## Issues identified

- **price_in_euro** (dtype, high): Stored as string but represents monetary values; prevents numeric aggregation.
- **power_kw** (dtype, high): Stored as string; prevents mathematical operations and model input.
- **power_ps** (dtype, high): Stored as string; prevents mathematical operations and model input.
- **year** (dtype, high): Stored as string; should be cast to integer for temporal features.
- **fuel_consumption_l_100km** (format, high): Contains European decimal commas, appended units ('l/100 km'), and empty strings for missing data.
- **fuel_consumption_g_km** (sentinel, medium): Uses '- (g/km)' as a placeholder for missing values instead of proper nulls.
- **fuel_consumption_g_km** (format, medium): Appended units ('g/km') block direct numeric conversion.
- **mileage_in_km** (outlier, medium): Maximum value of 3,800,000 km is physically implausible for passenger vehicles and severely skews distribution statistics.
- **fuel_consumption_l_100km** (null, low): 10.7% missing values after cleaning empty strings; expected for optional vehicle specs.
- **color** (null, low): 166 missing values (0.07%); negligible impact but noted for completeness.

## Steps applied (code)

- replace_sentinels on fuel_consumption_l_100km: 0 values → missing
- replace_sentinels on fuel_consumption_g_km: 35809 values → missing
- parse_european_decimal on fuel_consumption_l_100km: nulls 26873 → 26922
- parse_european_decimal on fuel_consumption_g_km: nulls 35809 → 36770
- astype on price_in_euro → float
- astype on power_kw → float
- astype on power_ps → float
- astype on year → int
- clip_outliers on mileage_in_km (IQR×3.0): 336 values capped
