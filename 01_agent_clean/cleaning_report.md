## Data Cleaning Summary

### Issues Identified
- **Wrong Dtypes**: `price_in_euro`, `power_kw`, `power_ps`, and `year` are stored as strings (`object`) instead of proper numeric types.
- **Format & Sentinels**: `fuel_consumption_l_100km` contains European-style decimal commas, appended units (`l/100 km`), and has a high missing rate (10.7%). `fuel_consumption_g_km` uses `- (g/km)` as a sentinel for missing values and also contains appended units.
- **Outliers**: `mileage_in_km` contains an extreme maximum of 3,800,000 km, which heavily skews distribution and is unrealistic for standard vehicle datasets.
- **Nulls**: Minor missingness (<0.1%) in `color`, `registration_date`, `power_kw`, `power_ps`, and `mileage_in_km`.

### Applied Transformations
1. Replaced `- (g/km)` sentinel in `fuel_consumption_g_km` with `NULL`.
2. Stripped units and converted European comma decimals to standard floats for both fuel consumption columns.
3. Cast `price_in_euro`, `power_kw`, and `power_ps` to `float`, and `year` to `int`.
4. Applied IQR clipping (factor=3) to `mileage_in_km` to cap extreme outliers at statistically realistic boundaries.
5. Retained `Unnamed: 0` for row tracking; recommend dropping in downstream feature engineering.

## Run stats

- Input rows: 251,079
- Output rows: 251,079

## Issues identified

- **price_in_euro** (dtype, high): Stored as object, should be numeric
- **power_kw** (dtype, high): Stored as object, should be numeric
- **power_ps** (dtype, high): Stored as object, should be numeric
- **year** (dtype, medium): Stored as object, should be integer
- **fuel_consumption_l_100km** (format, high): Contains European comma decimals, appended units (l/100 km), and 10.7% empty/missing values
- **fuel_consumption_g_km** (sentinel, medium): Uses '- (g/km)' as a placeholder for missing CO2 emission data
- **mileage_in_km** (outlier, medium): Maximum value of 3,800,000 km is physically unrealistic for passenger vehicles
- **Unnamed: 0** (metadata, low): Redundant index column exported from original dataset

## Steps applied (code)

- replace_sentinels on fuel_consumption_g_km: 35809 values → missing
- parse_european_decimal on fuel_consumption_l_100km: nulls 26873 → 26922
- parse_european_decimal on fuel_consumption_g_km: nulls 35809 → 36770
- astype on price_in_euro → float
- astype on power_kw → float
- astype on power_ps → float
- astype on year → int
- clip_outliers on mileage_in_km (IQR×3.0): 336 values capped
