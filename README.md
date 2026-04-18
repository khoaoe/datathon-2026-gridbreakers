# DATATHON 2026

Datathon 2026 here. VinTelligence make it. You be data scientist fashion e-commerce.

## Challenge
Three parts:
1. MCQ - 20 points
2. EDA - 60 points
3. Revenue Model - 20 points

### Goal
Predict daily revenue from Jan 1, 2023 to Jul 1, 2024. Use old data Jul 4, 2012 to Dec 31, 2022.

## Rules
Test predictions against true revenue. Metrics:
- MAE: Average error.
- RMSE: Square root average squared error. Big error bad.
- R2: Variance predicted.

File: Make CSV like `sample_submission.csv`. Need `Date` and `Revenue`.

## Data Dictionary
Data big. 5 parts.

### 1. Master Data
- `products.csv`: Products. `product_id`, `product_name`, `category`, `segment`, `size`, `color`, `price`, `cogs`.
- `customers.csv`: Customers. `customer_id`, `zip`, `city`, `signup_date`, `gender`, `age_group`, `acquisition_channel`.
- `promotions.csv`: Deals. `promo_id`, `promo_name`, `promo_type`, `discount_value`, `start_date`, `end_date`, `applicable_category`.
- `geography.csv`: Places. `zip`, `city`, `district`, `region`.

### 2. Operations
- `orders.csv`: Orders. `order_id`, `customer_id`, status, time, amount, ship fee.
- `order_items.csv`: Order items. Quantity, `promo_id`, discount.
- `payments.csv`: Money. Date, method, amount.
- `shipments.csv`: Boxes. Courier, level, cost, status.
- `returns.csv`: Take back. Logs, reason.
- `reviews.csv`: Stars. Rating 1-5, title, text.

### 3. Inventory
- `inventory.csv`: Stock. `stock_on_hand`, `units_received`, `units_sold`, `stockout_days`, sell rate, fill rate, flags.

### 4. Web Traffic
- `web_traffic.csv`: Site visits. `sessions`, `unique_visitors`, `page_views`, `bounce_rate`, `conversion_rate`.

### 5. Forecast Target
- `sales.csv`: Old revenue. `Revenue`, `COGS`.
- `sample_submission.csv`: Future target dates.
