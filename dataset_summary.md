# Dataset Summary

14 CSV files. Total ~123 MB. Time range: Jul 2012 to Jul 2024.

## Overview

| File | Rows | Cols | Size (MB) | Missing? |
|---|---|---|---|---|
| customers.csv | 121,930 | 7 | 6.75 | No |
| geography.csv | 39,948 | 4 | 1.34 | No |
| inventory.csv | 60,247 | 17 | 5.41 | No |
| order_items.csv | 714,669 | 7 | 22.83 | Yes |
| orders.csv | 646,945 | 8 | 43.83 | No |
| payments.csv | 646,945 | 4 | 17.53 | No |
| products.csv | 2,412 | 8 | 0.19 | No |
| promotions.csv | 50 | 10 | <0.01 | Yes |
| returns.csv | 39,939 | 7 | 2.18 | No |
| reviews.csv | 113,551 | 7 | 6.48 | No |
| sales.csv | 3,833 | 3 | 0.12 | No |
| sample_submission.csv | 548 | 3 | 0.02 | No |
| shipments.csv | 566,067 | 4 | 18.84 | No |
| web_traffic.csv | 3,652 | 7 | 0.20 | No |

---

## Per-File Details

### customers.csv (121,930 rows x 7 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| customer_id | int64 | 0 | 0.00% | 121,930 | 1 |
| zip | int64 | 0 | 0.00% | 31,491 | 15201 |
| city | object | 0 | 0.00% | 42 | Hai Phong |
| signup_date | object | 0 | 0.00% | 3,941 | 2021-12-30 |
| gender | object | 0 | 0.00% | 3 | Female |
| age_group | object | 0 | 0.00% | 5 | 35-44 |
| acquisition_channel | object | 0 | 0.00% | 6 | social_media |

---

### geography.csv (39,948 rows x 4 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| zip | int64 | 0 | 0.00% | 39,948 | 15201 |
| city | object | 0 | 0.00% | 42 | Hai Phong |
| region | object | 0 | 0.00% | 3 | East |
| district | object | 0 | 0.00% | 39 | District #13 |

---

### inventory.csv (60,247 rows x 17 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| snapshot_date | object | 0 | 0.00% | 126 | 2022-10-31 |
| product_id | int64 | 0 | 0.00% | 1,624 | 1 |
| stock_on_hand | int64 | 0 | 0.00% | 1,895 | 3 |
| units_received | int64 | 0 | 0.00% | 360 | 1 |
| units_sold | int64 | 0 | 0.00% | 303 | 1 |
| stockout_days | int64 | 0 | 0.00% | 29 | 2 |
| days_of_supply | float64 | 0 | 0.00% | 9,289 | 90.0 |
| fill_rate | float64 | 0 | 0.00% | 29 | 0.9333 |
| stockout_flag | int64 | 0 | 0.00% | 2 | 1 |
| overstock_flag | int64 | 0 | 0.00% | 2 | 0 |
| reorder_flag | int64 | 0 | 0.00% | 1 | 0 |
| sell_through_rate | float64 | 0 | 0.00% | 4,017 | 0.25 |
| product_name | object | 0 | 0.00% | 1,465 | DragonWear MA-01 |
| category | object | 0 | 0.00% | 4 | Casual |
| segment | object | 0 | 0.00% | 8 | All-weather |
| year | int64 | 0 | 0.00% | 11 | 2022 |
| month | int64 | 0 | 0.00% | 12 | 10 |

Note: `reorder_flag` has only 1 unique value (always 0). Likely not useful.

---

### order_items.csv (714,669 rows x 7 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| order_id | int64 | 0 | 0.00% | 646,945 | 1 |
| product_id | int64 | 0 | 0.00% | 1,598 | 2400 |
| quantity | int64 | 0 | 0.00% | 8 | 7 |
| unit_price | float64 | 0 | 0.00% | 501,330 | 1138.22 |
| discount_amount | float64 | 0 | 0.00% | 204,449 | 0.0 |
| promo_id | object | **438,353** | **61.34%** | 50 | PROMO-0006 |
| promo_id_2 | object | **714,463** | **99.97%** | 2 | PROMO-0015 |

Note: `promo_id` missing 61% (no promo applied). `promo_id_2` missing 99.97% (second stacked promo very rare).

---

### orders.csv (646,945 rows x 8 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| order_id | int64 | 0 | 0.00% | 646,945 | 1 |
| order_date | object | 0 | 0.00% | 3,833 | 2012-07-04 |
| customer_id | int64 | 0 | 0.00% | 90,246 | 58578 |
| zip | int64 | 0 | 0.00% | 29,932 | 1109 |
| order_status | object | 0 | 0.00% | 6 | delivered |
| payment_method | object | 0 | 0.00% | 5 | credit_card |
| device_type | object | 0 | 0.00% | 3 | desktop |
| order_source | object | 0 | 0.00% | 6 | paid_search |

---

### payments.csv (646,945 rows x 4 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| order_id | int64 | 0 | 0.00% | 646,945 | 1 |
| payment_method | object | 0 | 0.00% | 5 | credit_card |
| payment_value | float64 | 0 | 0.00% | 595,420 | 7967.54 |
| installments | int64 | 0 | 0.00% | 5 | 3 |

---

### products.csv (2,412 rows x 8 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| product_id | int64 | 0 | 0.00% | 2,412 | 536 |
| product_name | object | 0 | 0.00% | 2,172 | SaigonFlex UC-01 |
| category | object | 0 | 0.00% | 4 | Streetwear |
| segment | object | 0 | 0.00% | 8 | Everyday |
| size | object | 0 | 0.00% | 4 | S |
| color | object | 0 | 0.00% | 10 | green |
| price | float64 | 0 | 0.00% | 1,990 | 11059.65 |
| cogs | float64 | 0 | 0.00% | 2,381 | 9704.84 |

---

### promotions.csv (50 rows x 10 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| promo_id | object | 0 | 0.00% | 50 | PROMO-0001 |
| promo_name | object | 0 | 0.00% | 50 | Spring Sale 2013 |
| promo_type | object | 0 | 0.00% | 2 | percentage |
| discount_value | float64 | 0 | 0.00% | 6 | 12.0 |
| start_date | object | 0 | 0.00% | 50 | 2013-03-18 |
| end_date | object | 0 | 0.00% | 50 | 2013-04-17 |
| applicable_category | object | **40** | **80.00%** | 2 | Streetwear |
| promo_channel | object | 0 | 0.00% | 5 | email |
| stackable_flag | int64 | 0 | 0.00% | 2 | 1 |
| min_order_value | int64 | 0 | 0.00% | 5 | 0 |

Note: `applicable_category` missing 80% means most promos apply to all categories.

---

### returns.csv (39,939 rows x 7 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| return_id | object | 0 | 0.00% | 39,939 | RET-000001 |
| order_id | int64 | 0 | 0.00% | 36,062 | 2 |
| product_id | int64 | 0 | 0.00% | 1,286 | 609 |
| return_date | object | 0 | 0.00% | 3,806 | 2012-07-25 |
| return_reason | object | 0 | 0.00% | 5 | late_delivery |
| return_quantity | int64 | 0 | 0.00% | 8 | 6 |
| refund_amount | float64 | 0 | 0.00% | 39,560 | 52458.01 |

---

### reviews.csv (113,551 rows x 7 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| review_id | object | 0 | 0.00% | 113,551 | REV-0000001 |
| order_id | int64 | 0 | 0.00% | 111,369 | 1 |
| product_id | int64 | 0 | 0.00% | 1,412 | 2400 |
| customer_id | int64 | 0 | 0.00% | 48,676 | 58578 |
| review_date | object | 0 | 0.00% | 3,825 | 2012-07-24 |
| rating | int64 | 0 | 0.00% | 5 | 5 |
| review_title | object | 0 | 0.00% | 18 | Highly recommend |

---

### sales.csv (3,833 rows x 3 cols) -- TRAIN TARGET

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| Date | object | 0 | 0.00% | 3,833 | 2012-07-04 |
| Revenue | float64 | 0 | 0.00% | 3,833 | 5123547.94 |
| COGS | float64 | 0 | 0.00% | 3,833 | 3982991.19 |

---

### sample_submission.csv (548 rows x 3 cols) -- TEST TARGET

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| Date | object | 0 | 0.00% | 548 | 2023-01-01 |
| Revenue | float64 | 0 | 0.00% | 548 | 2665507.2 |
| COGS | float64 | 0 | 0.00% | 548 | 2518885.15 |

---

### shipments.csv (566,067 rows x 4 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| order_id | int64 | 0 | 0.00% | 566,067 | 1 |
| ship_date | object | 0 | 0.00% | 3,831 | 2012-07-07 |
| delivery_date | object | 0 | 0.00% | 3,831 | 2012-07-11 |
| shipping_fee | float64 | 0 | 0.00% | 1,856 | 1.37 |

---

### web_traffic.csv (3,652 rows x 7 cols)

| Column | Type | Missing | Missing % | Unique | Sample |
|---|---|---|---|---|---|
| date | object | 0 | 0.00% | 3,652 | 2013-01-01 |
| sessions | int64 | 0 | 0.00% | 3,447 | 9760 |
| unique_visitors | int64 | 0 | 0.00% | 3,382 | 7253 |
| page_views | int64 | 0 | 0.00% | 3,620 | 39093 |
| bounce_rate | float64 | 0 | 0.00% | 261 | 0.00514 |
| avg_session_duration_sec | float64 | 0 | 0.00% | 1,771 | 102.9 |
| traffic_source | object | 0 | 0.00% | 6 | organic_search |

Note: web_traffic starts from 2013-01-01, not Jul 2012 like other tables.

---

## Key Observations

- **Missing data is rare.** Only 3 columns across all files have nulls:
  - `order_items.promo_id` — 61.34% null (most items have no promo)
  - `order_items.promo_id_2` — 99.97% null (stacked promos almost never happen)
  - `promotions.applicable_category` — 80% null (most promos apply to all categories)
- **Data is very clean.** No missing values in any numeric or key columns.
- **Biggest files:** `orders.csv` (44MB, 647K rows), `order_items.csv` (23MB, 715K rows), `shipments.csv` (19MB, 566K rows).
- **Smallest files:** `promotions.csv` (50 rows), `products.csv` (2,412 rows), `sales.csv` (3,833 rows).
- **Time coverage:** Most tables span Jul 2012 to Dec 2022. `web_traffic.csv` starts Jan 2013.
- **Target:** Predict 548 days of daily `Revenue` (and `COGS`) from Jan 2023 to Jul 2024.
- **`reorder_flag`** in inventory is always 0. Dead column.
