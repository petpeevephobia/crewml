## Data Quality & Cleaning Report

**Dataset Overview:** 251,079 rows, 15 columns.

### Key Findings
- **Dtype Mismatches:** Six numeric features (`year`, `price_in_euro`, `power_kw`, `power_ps`, `fuel_consumption_l_100km`, `fuel_consumption_g_km`) are stored as strings, blocking mathematical operations and model ingestion.
- **Formatting & Sentinels:** `fuel_consumption_l_100km` mixes European comma decimals with text units (`l/100 km`). `fuel_consumption_g_km` uses a `- (g/km)` sentinel for unknown values rather than standard nulls.
- **Missing Data:** `fuel_consumption_l_100km` has a 10.7% missing rate (likely empty strings). Minor missingness exists in `power_kw` (0.05%), `power_ps` (0.05%), and `color` (0.07%).
- **Outliers:** `mileage_in_km` reaches 3,800,000 km, which is physically unrealistic for consumer vehicles and heavily distorts the mean (85,340 km) and standard deviation.
- **Metadata:** `Unnamed: 0` is a redundant index column.

### Applied Transformations
1. Replaced `- (g/km)` sentinel in `fuel_consumption_g_km` with proper `NaN` values.
2. Parsed European commas and stripped unit text from both fuel consumption columns, converting them to numeric floats.
3. Cast `year`, `price_in_euro`, `power_kw`, and `power_ps` to `int` and `float` respectively.
4. Applied IQR-based clipping (factor=3) to `mileage_in_km` to cap extreme outliers without dropping records.

All fixes prioritize value preservation over row deletion. Downstream steps should handle the resulting `NaN`s (imputation or model-specific null handling) and drop the `Unnamed: 0` index column before training.

## Run stats

- Input rows: 251,079
- Output rows: 251,079

## Issues identified

- **year** (dtype, high): Stored as string, should be numeric (int)
- **price_in_euro** (dtype, high): Stored as string, should be numeric (float)
- **power_kw** (dtype, high): Stored as string, should be numeric (float)
- **power_ps** (dtype, high): Stored as string, should be numeric (float)
- **fuel_consumption_l_100km** (format, high): Contains European comma decimals and unit suffixes (e.g., '10,9 l/100 km')
- **fuel_consumption_g_km** (sentinel, medium): Contains '- (g/km)' placeholder for missing data instead of null
- **fuel_consumption_g_km** (format, high): Stored as string with unit suffixes, should be numeric
- **mileage_in_km** (outlier, high): Maximum value 3,800,000 km is physically implausible and will skew statistical summaries and model scaling

## Steps applied (code)

- replace_sentinels on fuel_consumption_g_km: 35809 values → missing
- parse_european_decimal on fuel_consumption_l_100km: nulls 26873 → 26922
- parse_european_decimal on fuel_consumption_g_km: nulls 35809 → 36770
- astype on year → int
- astype on price_in_euro → float
- astype on power_kw → float
- astype on power_ps → float
- clip_outliers on mileage_in_km (IQR×3.0): 336 values capped
