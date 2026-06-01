## Data Cleaning Summary
Processed 251,079 vehicle listings across 15 columns. Identified and resolved multiple dtype, format, sentinel, and outlier issues while preserving all rows.

## Issues & Applied Fixes
1. **Sentinel Removal (`fuel_consumption_g_km`)**: Replaced `- (g/km)` placeholders with `NaN` to standardize missing values.
2. **Decimal & Unit Parsing**: Applied European decimal parsing to both `fuel_consumption_l_100km` and `fuel_consumption_g_km`. This strips `l/100 km` / `g/km` units, converts commas to dots, and handles empty strings by casting them to `NaN`.
3. **Type Casting**: Converted `price_in_euro`, `power_kw`, and `power_ps` from `object` to `float` for numerical analysis. Converted `year` from `object` to `int`.
4. **Outlier Clipping (`mileage_in_km`)**: Detected extreme maximum mileage of 3,800,000 km, which severely skews distributions. Applied 3×IQR clipping to cap values at the statistically valid upper bound (~431,288 km).

Minor nulls in `color` (0.07%), `registration_date` (~0%), and `offer_description` (~0%) were retained as they fall well below imputation thresholds and do not impact core modeling features.

## Run stats

- Input rows: 251,079
- Output rows: 251,079

## Issues identified

- **fuel_consumption_g_km** (sentinel, high): Contains '- (g/km)' indicating missing or unavailable CO2 data
- **fuel_consumption_l_100km** (dtype/format, high): Stored as object with European decimal commas and units (l/100 km), 10.7% empty/missing
- **fuel_consumption_g_km** (dtype/format, high): Stored as object with units (g/km)
- **price_in_euro** (dtype, high): Numeric currency values stored as object strings
- **power_kw** (dtype, medium): Numeric power values stored as object strings
- **power_ps** (dtype, medium): Numeric power values stored as object strings
- **year** (dtype, medium): Integer years stored as object strings
- **mileage_in_km** (outlier, medium): Maximum value 3,800,000 km far exceeds expected upper bound (approx. 431,288 km via 3xIQR)

## Steps applied (code)

- replace_sentinels on fuel_consumption_g_km: 35809 values → missing
- parse_european_decimal on fuel_consumption_l_100km: nulls 26873 → 26922
- parse_european_decimal on fuel_consumption_g_km: nulls 35809 → 36770
- astype on price_in_euro → float
- astype on power_kw → float
- astype on power_ps → float
- astype on year → int
- clip_outliers on mileage_in_km (IQR×3.0): 336 values capped
