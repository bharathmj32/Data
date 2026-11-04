# Verified sample results — Pandas path

These are calculated from `sales_eda_practice.csv` using `eda_pandas_only.py`.

| Check | Expected result |
|---|---:|
| Raw rows | 794 |
| Raw columns | 18 |
| Distinct raw order_line_id | 784 |
| Full-row exact duplicates | 5 |
| Non-null PK extra duplicates | 10 |
| Cleaned unique order lines | 773 |
| Retained distinct orders | 515 |
| Cleaned total revenue (₹) | 3,657,767.57 |
| Average order value (₹) | 7,102.46 |
| Leading category | Electronics (46.4% revenue share) |
| Leading region | North (54.5% revenue share) |
| 3-sigma revenue outlier rows | 4 |

**Important:** The complete Pandas code was executed successfully in this environment. The PySpark code is supplied for Spark 3.5+ / Databricks and syntax checked, but a Spark runtime is not available in this environment for execution verification. Exact approximate percentiles may vary slightly between Pandas and Spark due to `percentile_approx`.
