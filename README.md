# EDA Interview Practice Kit — 0 to 5

A **hands-on practice kit** with the exact requested steps in **Pandas, PySpark, Spark SQL, Matplotlib and Seaborn**.

## Files

| File | What to do |
|---|---|
| `sales_eda_practice.csv` | Start here: 794 deliberately imperfect sales records and 18 columns. |
| `EDA_Interview_Full_Practice.ipynb` | **Best choice**: open in Jupyter or Databricks, execute cells from 0 to 5. |
| `eda_complete_pandas_pyspark.py` | Same sections as notebook, executable as one Python script with Spark. |
| `eda_pandas_only.py` | Run all Pandas and plotting material without a Spark installation. |
| `eda_kpis_spark.sql` | Copy-paste KPI queries into Databricks/Spark SQL **after cleaning/view creation**. |
| `make_dataset.py` | Regenerate the exact same CSV using only standard Python. |
| `build_learning_files.py` | Rebuild the notebook and full script. |

## Quick start

**Pandas only** (simplest on a laptop):

```bash
pip install pandas numpy matplotlib seaborn
python eda_pandas_only.py
```

**Full Notebook (Jupyter with Spark)**:

```bash
pip install pandas numpy matplotlib seaborn pyspark jupyter nbformat
jupyter notebook EDA_Interview_Full_Practice.ipynb
```

**Databricks**: upload the CSV to a workspace-accessible UC Volume or DBFS directory, then replace the `DATA_FILE` assignment in Section 0 with that absolute location (`/Volumes/.../sales_eda_practice.csv`, for example). On notebook runs, `SparkSession.builder.getOrCreate()` uses the available Spark session. If using the SQL file, execute Section 2D to create `sales_clean` first and keep the same Spark session; temporary views are session-scoped. For a permanent SQL table, write the cleaned Delta DataFrame to a managed table with the appropriate permissions.

**PySpark compatibility:** uses Spark 3.5+ `try_to_timestamp` and Spark SQL `try_cast` (Databricks also supports these). Using `try_to_timestamp` avoids errors on malformed dates under ANSI mode.

## Dataset schema and business meaning

| Column | Intended type | Explanation |
|---|---|---|
| `order_line_id` | identifier / primary key | Unique line ID **after deduplication**, e.g. `L00001`. |
| `order_id` | identifier | Repeated across order items; use distinct orders for AOV. |
| `order_date` | date | Sales order date; 2026-01 to 2026-05. |
| `last_updated` | timestamp | Use latest version to resolve duplicate primary keys. |
| `customer_id` | identifier | Customer, may place many orders. |
| `product_id` | identifier | Product/SKU. |
| `product_name` | categorical | Human-readable product. |
| `category` | categorical | Electronics / Fashion / Home / Grocery / Sports. |
| `region` | categorical | North / South / East / West / Central. |
| `channel` | categorical | Online / Retail / Partner. |
| `day_name` | categorical | Derived from order date after cleaning; raw inconsistencies intentional. |
| `quantity` | numeric | Units per order line, must be positive. |
| `unit_price` | numeric | Unit selling price, must be positive. |
| `discount_pct` | numeric | Discount percentage from 0 inclusive to below 100. |
| `m_revenue` | numeric | Net sales revenue, recomputed as `qty * price * (1-discount/100)` when cleaned. |
| `delivery_days` | numeric | 0 to 30 accepted; invalid becomes null. |
| `customer_age` | numeric | Ages 18 to 100 accepted; invalid becomes null. |
| `is_returned` | boolean | `true`/`false`; unknown stays missing (avoid treating as no return). |

**Quality traps intentionally seeded:** full duplicate rows; duplicated `order_line_id` with revised value/later timestamp; positive and negative quality check cases; missing keys/dates/numeric inputs; zero/negative quantity and price; invalid discounts; age 0 or 157; negative/80-day delivery; invalid date; inconsistent upper/lower/mixed casing and whitespace for categorical values; mismatched/missing/negative recorded revenue; unknown return label. Two large but valid bulk-order lines show why three-sigma outliers should **not** automatically be deleted. Customer and region choices are deliberately non-uniform to surface skew.

## Sections and interview checklist

0. **Set up** imports (pandas, NumPy, Matplotlib, Seaborn, PySpark), construct Pandas + Spark DataFrames, register SQL temp view.
1. **Understand data** `head/show/tail`, row/column counts, schema, `describe/summary`, identifier/numeric/categorical/datetime/boolean classification, business relationships.
2. **Quality + cleaning** null counts and %, full-row and primary-key duplicates, range checks, zero/negative/illogical values, inconsistent strings and faulty revenue, preserve raw frame and produce separate cleaned Pandas/Spark frames.
3. **Statistics + relationships** key skew, mean/median/mode/min/max/std, outliers via ±3σ, correlations, histogram, pivot, bar/pie.
4. **KPI + business metrics** SUM, AVG, P90, P95, AOV, category %, top-5 contribution, monthly growth, 7-day calendar moving average, top customers and products in Pandas/PySpark/SQL.
5. **Visuals + insights** Matplotlib bar/line/hist/scatter/pie, Seaborn equivalents (except pie), Spark aggregate/sample -> Pandas -> plots, data-grounded actionable findings.

### The easiest way to adapt to any other dataset

Edit `DATA_FILE`, `RENAME_MAP`, the KPI column names, relevant cleaning rules, and the mappings `ID_COLS`, `NUM_COLS`, `CATEGORY_COLS`, `DATE_COLS`, `BOOLEAN_COLS`, then decide which columns are your primary key, time column, grain, revenue column, relevant categorical columns, and valid business ranges. Replace `order_id` in the AOV denominator only after confirming the grain. **Do not** blindly apply positive-only filters if refunds/credits are valid. If the time series is weekly rather than daily, adjust the rolling time window accordingly. Spark `.toPandas()` on a large raw distributed DataFrame can exhaust driver memory; aggregate/sample/limit first.

### What to say in an interview

> “I identified the grain and keys, profiled schema, nulls and duplicates, and applied business validation. I produced new cleaned DataFrames, deduplicated by last update and standardized dimensions. I checked skew, descriptive stats, three-sigma outliers and correlation, then calculated AOV, category contribution, top-N concentration, growth and a true seven-calendar-day moving average. I used bar, line, histogram, scatter and pie charts to explain observations, business impact and follow-up actions. I would validate assumptions before claiming causation.”
