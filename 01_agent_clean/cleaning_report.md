## Data Cleaning Report

### Issues Identified
- **Dtype Mismatches**: `year`, `price_in_euro`, `power_kw`, `power_ps`, `fuel_consumption_l_100km`, and `fuel_consumption_g_km` are currently stored as strings (`object`) instead of numeric types.
- **Sentinel Values**: `fuel_consumption_g_km` uses `"- (g/km)"` to denote missing data. `fuel_consumption_l_100km` contains empty strings `""`.
- **Formatting**: Fuel consumption columns use European decimal format (commas) and contain unit strings that must be stripped.
- **Outliers**: `mileage_in_km` has a maximum value of 3,800,000 km, which is statistically extreme and likely erroneous.
- **Missing Data**: `fuel_consumption_l_100km` has 10.7% missing values. Other columns have minimal missingness (<0.1%).

### Transformations Applied
1. Replaced sentinel values (`- (g/km)` and empty strings) with `NaN` to standardize missing data representation.
2. Applied European decimal parsing to strip units and convert comma-separated decimals to floats for both fuel consumption columns.
3. Converted `year` and `price_in_euro` to integers, and `power_kw`, `power_ps`, and parsed fuel columns to floats to safely accommodate `NaN` values.
4. Applied IQR-based outlier clipping (`factor=3`) on `mileage_in_km` to cap extreme values without dropping rows.

### Result
Dataset is now numerically typed, cleaned of formatting artifacts, and ready for downstream ML feature engineering and modeling.

## Run stats

- Input rows: 251,079
- Output rows: 251,079

## Issues identified

- **year** (dtype, medium): Stored as object/string but contains integer years
- **price_in_euro** (dtype, medium): Stored as object/string but contains numeric values
- **power_kw** (dtype, medium): Stored as object/string, needs numeric conversion for calculations
- **power_ps** (dtype, medium): Stored as object/string, needs numeric conversion for calculations
- **fuel_consumption_l_100km** (dtype, high): Stored as object with European decimals and units (e.g., '10,9 l/100 km')
- **fuel_consumption_g_km** (dtype, high): Stored as object with units and sentinel values
- **fuel_consumption_g_km** (sentinel, high): Contains '- (g/km)' indicating missing/unknown consumption data
- **fuel_consumption_l_100km** (sentinel, medium): Contains empty strings '' representing missing data
- **mileage_in_km** (outlier, high): Max value is 3,800,000 km, likely a data entry error or extreme outlier
- **fuel_consumption_l_100km** (null, medium): 10.7% missing values (26,873 rows), acceptable but requires careful handling

## Steps applied (code)

- replace_sentinels on fuel_consumption_g_km: 35809 values → missing
- replace_sentinels on fuel_consumption_l_100km: 0 values → missing
- parse_european_decimal on fuel_consumption_l_100km: nulls 26873 → 26922
- parse_european_decimal on fuel_consumption_g_km: nulls 35809 → 36770
- astype on year → int
- astype on price_in_euro → int
- astype on power_kw → float
- astype on power_ps → float
- astype on fuel_consumption_l_100km → float
- astype on fuel_consumption_g_km → float
- clip_outliers on mileage_in_km (IQR×3.0): 336 values capped
