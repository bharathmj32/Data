# EDA Complete Step-by-Step Copy-Paste Reference

Run all code cells **in order**. PySpark requires a Spark 3.5+ environment or Databricks.

# EDA Interview Practice — Pandas + PySpark + Spark SQL + Charts

**Dataset:** `sales_eda_practice.csv` — 794 raw sales line-items, 18 columns, 520 orders spanning January–May 2026. The data intentionally contains **nulls, conflicting duplicate keys, exact duplicates, invalid quantities/prices/discounts, bad dates, inconsistent strings, erroneous revenue, skew and genuine outliers**.

Follow sections **0 → 5** in order. Each code cell builds on preceding cells. Run everything in a notebook with Pandas and PySpark available (Databricks or Jupyter with PySpark); the accompanying `eda_pandas_only.py` works without Spark.

**Grain:** one row = one order line; `order_line_id` is the primary key. `order_id` repeats legitimately for multi-line orders. **AOV = SUM(m_revenue) / COUNT(DISTINCT order_id)**, not line count.

**Terminology:** `m_revenue` = net sales revenue per order line, derived as `quantity * unit_price * (1 - discount_pct/100)`. In this exercise use ₹ as a hypothetical currency; amounts are synthetic, not real financial figures.

## 0. Initial setup — packages, Pandas DataFrame, Spark DataFrame, SQL table/view

Install libraries once on a local environment with `pip install pandas numpy matplotlib seaborn pyspark jupyter nbformat`. In Databricks, Spark is already available. A **temporary SQL view** is used so no database permissions are needed. For a physical Delta table, see the optional snippet at the end.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import NumericType, BooleanType, DateType, TimestampType

pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")

# Change ONLY this location + the column mapping if adapting a different dataset.
DATA_FILE = Path("sales_eda_practice.csv")
if not DATA_FILE.exists():
    DATA_FILE = Path("/mnt/data/eda_interview_practice/sales_eda_practice.csv")
assert DATA_FILE.exists(), f"CSV not found: {DATA_FILE}"

# To reuse with differently named CSV fields, e.g. {"sales": "m_revenue"}.
RENAME_MAP = {}
ID_COLS = ["order_line_id", "order_id", "customer_id", "product_id"]
NUM_COLS = ["quantity", "unit_price", "discount_pct", "m_revenue", "delivery_days", "customer_age"]
CATEGORY_COLS = ["product_name", "category", "region", "channel", "day_name"]
DATE_COLS = ["order_date", "last_updated"]
BOOLEAN_COLS = ["is_returned"]

# Pandas DataFrame. String IDs prevent unwanted conversion of IDs to numbers.
pd_raw = pd.read_csv(DATA_FILE, dtype={col: "string" for col in ID_COLS}).rename(columns=RENAME_MAP)

# Spark DataFrame: retain strings until explicit, safe type conversion in Section 2.
spark = SparkSession.builder.appName("EDA_Interview_Practice").getOrCreate()
sp_raw = spark.read.option("header", True).option("inferSchema", False).csv(str(DATA_FILE))
for old_name, new_name in RENAME_MAP.items():
    sp_raw = sp_raw.withColumnRenamed(old_name, new_name)

# SQL queryable view (raw), followed by a cleaned one in Section 2.
sp_raw.createOrReplaceTempView("sales_raw")
spark.sql("SELECT COUNT(*) AS raw_rows FROM sales_raw").show()
print("Pandas shape:", pd_raw.shape)
```

## 1. Data understanding and exploration — Pandas + PySpark

**Questions to answer:** What does a row represent? Which identifiers define uniqueness? What are the date range and field types? What relationships exist among columns? How many missing/inconsistent values appear? Spark has `show()` and `tail()`, but neither guarantees a meaningful order unless you explicitly sort.

```python
# 1A — Pandas: preview, dimensions, columns, schema, summary
print("First 5:")
print(pd_raw.head(5).to_string(index=False))
print("Last 5:")
print(pd_raw.tail(5).to_string(index=False))
print("Row count:", len(pd_raw), "Column count:", pd_raw.shape[1])
print("Column names:", pd_raw.columns.tolist())
print("Dtypes:")
print(pd_raw.dtypes)
print("Numeric summary:")
print(pd_raw.describe(include="number"))  # numeric only where inferred
print("All-column summary:")
print(pd_raw.describe(include="all").to_string())
print("Distinct per column:\n", pd_raw.nunique(dropna=False))
```

```python
# 1B — PySpark: preview, dimensions, columns, schema, describe and summary
sp_raw.show(5, truncate=False)
print("Spark tail (not meaningful unless sorted):", sp_raw.tail(3))
sp_raw.orderBy(F.col("order_line_id").desc()).show(5, truncate=False)
print("Rows:", sp_raw.count(), "Columns:", len(sp_raw.columns))
print("Column names:", sp_raw.columns)
sp_raw.printSchema()  # all raw CSV fields are strings by design
sp_raw.describe("quantity", "unit_price", "m_revenue").show()
sp_raw.select("quantity", "unit_price", "m_revenue").summary(
    "count", "mean", "stddev", "min", "25%", "50%", "75%", "max"
).show()
# For numeric-only summary, explicitly cast strings first (see Section 2).
```

```python
# 1C — Automated column classification by business-name rules and data types.
# Reuse these column-name overrides on other datasets (date text still classified as date).
def classify_pandas(df):
    groups = {"identifier": [], "datetime": [], "boolean": [],
              "numerical": [], "categorical": []}
    for c in df.columns:
        if c in ID_COLS or c.lower().endswith("_id"):
            group = "identifier"
        elif c in DATE_COLS or pd.api.types.is_datetime64_any_dtype(df[c]):
            group = "datetime"
        elif c in BOOLEAN_COLS or pd.api.types.is_bool_dtype(df[c]):
            group = "boolean"
        elif pd.api.types.is_numeric_dtype(df[c]) or c in NUM_COLS:
            group = "numerical"
        else:
            group = "categorical"
        groups[group].append(c)
    return groups

def classify_spark(df):
    groups = {"identifier": [], "datetime": [], "boolean": [],
              "numerical": [], "categorical": []}
    for field in df.schema.fields:
        c, typ = field.name, field.dataType
        if c in ID_COLS or c.lower().endswith("_id"):
            group = "identifier"
        elif c in DATE_COLS or isinstance(typ, (DateType, TimestampType)):
            group = "datetime"
        elif c in BOOLEAN_COLS or isinstance(typ, BooleanType):
            group = "boolean"
        elif isinstance(typ, NumericType) or c in NUM_COLS:
            group = "numerical"
        else:
            group = "categorical"
        groups[group].append(c)
    return groups

print("Pandas classification:", classify_pandas(pd_raw))
print("Spark classification:", classify_spark(sp_raw))
```

```python
# 1D — Relationships and business interpretation: inspect join-cardinality and revenue drivers.
# One customer -> many orders; one order -> many order lines; product -> category.
print("Orders:", pd_raw["order_id"].nunique(),
      "Order lines:", pd_raw["order_line_id"].nunique(),
      "Customers:", pd_raw["customer_id"].nunique())
print("Order lines per order (sample):")
print(pd_raw.groupby("order_id")["order_line_id"].nunique().sort_values(ascending=False).head())
print("Product -> category variants (raw string inconsistencies show up):")
print(pd_raw.groupby("product_id")["category"].nunique().sort_values(ascending=False).head())
sp_raw.groupBy("order_id").agg(F.countDistinct("order_line_id").alias("lines_per_order")) \
    .orderBy(F.desc("lines_per_order")).show(5)
# Business hypotheses: region/category/channel influence revenue; quantity and price drive
# m_revenue; order_date enables growth and seasonality; is_returned signals return rate.
```

## 2. Data quality and cleaning — Pandas + PySpark

Keep raw DataFrames untouched. In order: **profile nulls → identify duplicates → test ranges/logical values → check strings and revenue → remove exact duplicates → type-cast → keep latest primary-key version → validate transactions → normalize strings/Booleans → derive accurate revenue and weekdays**.

**Rules for this practice dataset:** drop rows with missing business keys, invalid dates, nonpositive quantity/price, or discount outside `[0, 100)`; keep the latest `last_updated` for conflicting `order_line_id`; set unreasonable age/delivery days to null (not 0); fill unknown dimensions as `Unknown`; recalculate `m_revenue`. These are *example business rules*—change them based on each real dataset's semantics.

```python
# 2A — Pandas quality report: per-column null counts and percentages
null_report_pd = pd.DataFrame({
    "null_count": pd_raw.isna().sum(),
    "null_pct": (pd_raw.isna().mean() * 100).round(2)
}).sort_values("null_pct", ascending=False)
print(null_report_pd.to_string())

# Duplicate full rows vs duplicate PKs are different problems.
print("Exact duplicate rows:", int(pd_raw.duplicated().sum()))
print("Repeated non-null order_line_id entries:", int(pd_raw.loc[
    pd_raw["order_line_id"].notna(), "order_line_id"].duplicated().sum()))
print("Legitimate repeat order_id count:", int(pd_raw["order_id"].duplicated().sum()))

# Range, zero, negative, illogical and bad-date checks without changing raw input.
raw_qty = pd.to_numeric(pd_raw["quantity"], errors="coerce")
raw_price = pd.to_numeric(pd_raw["unit_price"], errors="coerce")
raw_discount = pd.to_numeric(pd_raw["discount_pct"], errors="coerce")
raw_revenue = pd.to_numeric(pd_raw["m_revenue"], errors="coerce")
raw_age = pd.to_numeric(pd_raw["customer_age"], errors="coerce")
raw_delivery = pd.to_numeric(pd_raw["delivery_days"], errors="coerce")
raw_dates = pd.to_datetime(pd_raw["order_date"], errors="coerce", format="%Y-%m-%d")
checks_pd = {
    "quantity <= 0": int(raw_qty.le(0).sum()),
    "unit_price <= 0": int(raw_price.le(0).sum()),
    "discount not in [0,100)": int((raw_discount.lt(0) | raw_discount.ge(100)).sum()),
    "m_revenue <= 0": int(raw_revenue.le(0).sum()),
    "age not in 18..100": int((raw_age.lt(18) | raw_age.gt(100)).sum()),
    "delivery_days not in 0..30": int((raw_delivery.lt(0) | raw_delivery.gt(30)).sum()),
    "missing/unparseable order_date": int(raw_dates.isna().sum()),
}
print("Pandas validation failures:", checks_pd)
print("Category spelling/case variants:", sorted(pd_raw["category"].dropna().unique())[:20])
print("Normalised category values:", sorted(pd_raw["category"].dropna().str.strip().str.title().unique()))
expected = raw_qty * raw_price * (1 - raw_discount / 100)
print("Revenue mismatches (> ₹1):", int(((raw_revenue - expected).abs() > 1).sum()))
```

```python
# 2B — Spark quality: null/blank % by column, duplicates, invalid values and strings
sp_total = sp_raw.count()
sp_null_exprs = [F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), 1)
                       .otherwise(0)).alias(c) for c in sp_raw.columns]
sp_null_wide = sp_raw.agg(*sp_null_exprs)
sp_null_long = sp_null_wide.select(F.explode(F.array(*[
    F.struct(F.lit(c).alias("column"), F.col(c).alias("null_count"))
    for c in sp_raw.columns
])).alias("entry")).select("entry.*").withColumn(
    "null_pct", F.round(F.col("null_count") * 100 / F.lit(sp_total), 2))
sp_null_long.orderBy(F.desc("null_pct")).show(30, truncate=False)

print("Exact duplicate rows:", sp_total - sp_raw.dropDuplicates().count())
print("Repeated PK rows:", sp_raw.filter(F.col("order_line_id").isNotNull())
      .groupBy("order_line_id").count().filter("count > 1")
      .agg(F.sum(F.col("count") - 1)).first()[0])

# try_cast prevents bad strings/empty cells failing in ANSI mode.
sp_quality = sp_raw.select(
    "*", F.expr("try_cast(quantity AS DOUBLE)").alias("q_num"),
    F.expr("try_cast(unit_price AS DOUBLE)").alias("p_num"),
    F.expr("try_cast(discount_pct AS DOUBLE)").alias("d_num"),
    F.expr("try_cast(m_revenue AS DOUBLE)").alias("r_num"),
    F.expr("try_cast(customer_age AS INT)").alias("age_num"),
    F.expr("try_cast(delivery_days AS INT)").alias("days_num"),
    F.to_date(F.try_to_timestamp(F.col("order_date"), F.lit("yyyy-MM-dd"))).alias("date_num")
)
sp_quality.select(
    F.sum(F.when(F.col("q_num") <= 0, 1).otherwise(0)).alias("qty_nonpositive"),
    F.sum(F.when(F.col("p_num") <= 0, 1).otherwise(0)).alias("price_nonpositive"),
    F.sum(F.when((F.col("d_num") < 0) | (F.col("d_num") >= 100), 1).otherwise(0)).alias("discount_invalid"),
    F.sum(F.when(F.col("r_num") <= 0, 1).otherwise(0)).alias("revenue_nonpositive"),
    F.sum(F.when((F.col("age_num") < 18) | (F.col("age_num") > 100),1).otherwise(0)).alias("age_invalid"),
    F.sum(F.when((F.col("days_num") < 0) | (F.col("days_num") > 30),1).otherwise(0)).alias("delivery_invalid"),
    F.sum(F.when(F.col("date_num").isNull(), 1).otherwise(0)).alias("date_missing_invalid")
).show()
sp_raw.groupBy("category").count().orderBy(F.desc("count")).show(30, truncate=False)
sp_quality.filter((F.abs(F.col("r_num") - F.col("q_num") * F.col("p_num") *
                         (1 - F.col("d_num") / 100)) > 1)).select(
    "order_line_id", "m_revenue", "quantity", "unit_price", "discount_pct").show()
```

```python
# 2C — Pandas cleaning. Keep a NEW frame: pd_clean (raw unchanged).
pd_clean = pd_raw.drop_duplicates().copy()
pd_clean[ID_COLS] = pd_clean[ID_COLS].apply(lambda s: s.str.strip().replace("", pd.NA))
for c in NUM_COLS:
    pd_clean[c] = pd.to_numeric(pd_clean[c], errors="coerce")
pd_clean["order_date"] = pd.to_datetime(pd_clean["order_date"], format="%Y-%m-%d", errors="coerce")
pd_clean["last_updated"] = pd.to_datetime(pd_clean["last_updated"], format="%Y-%m-%d %H:%M:%S", errors="coerce")

# Latest record wins when the same primary key occurs more than once.
pd_clean = (pd_clean.sort_values(["last_updated", "order_line_id"], na_position="first")
            .drop_duplicates(subset=["order_line_id"], keep="last").copy())
pd_clean = pd_clean.dropna(subset=ID_COLS + ["order_date", "quantity", "unit_price", "discount_pct"])
pd_clean = pd_clean.loc[
    (pd_clean["quantity"] > 0) & (pd_clean["unit_price"] > 0) &
    pd_clean["discount_pct"].between(0, 100, inclusive="left")
].copy()
for c in CATEGORY_COLS:
    pd_clean[c] = pd_clean[c].astype("string").str.strip().str.title().fillna("Unknown")
    pd_clean[c] = pd_clean[c].replace("", "Unknown")
# Use the trustworthy order date rather than the often-misspelled supplied weekday.
pd_clean["day_name"] = pd_clean["order_date"].dt.day_name()
pd_clean.loc[~pd_clean["customer_age"].between(18, 100), "customer_age"] = np.nan
pd_clean.loc[~pd_clean["delivery_days"].between(0, 30), "delivery_days"] = np.nan
pd_clean["is_returned"] = (pd_clean["is_returned"].astype("string").str.strip().str.lower()
    .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    .astype("boolean"))
# Policy: keep a record of original reported revenue, recalculate net sales.
pd_clean["m_revenue_original"] = pd_clean["m_revenue"]
pd_clean["m_revenue"] = (pd_clean["quantity"] * pd_clean["unit_price"] *
    (1 - pd_clean["discount_pct"] / 100)).round(2)
print("Pandas cleaned rows:", len(pd_clean), "unique line IDs:", pd_clean["order_line_id"].nunique())
assert pd_clean["order_line_id"].is_unique
assert pd_clean["m_revenue"].gt(0).all()
print(pd_clean.head(3).to_string(index=False))
```

```python
# 2D — PySpark cleaning with explicit typed columns, Window PK dedup & safeguards.
sp_clean = sp_raw.dropDuplicates()
for c in ID_COLS:
    sp_clean = sp_clean.withColumn(c, F.when(F.trim(F.col(c)) == "", None)
                                    .otherwise(F.trim(F.col(c))))
for c in NUM_COLS:
    sp_clean = sp_clean.withColumn(c, F.expr(f"try_cast({c} AS DOUBLE)"))
sp_clean = sp_clean.withColumn(
    "order_date", F.to_date(F.try_to_timestamp(F.col("order_date"), F.lit("yyyy-MM-dd"))))
sp_clean = sp_clean.withColumn(
    "last_updated", F.try_to_timestamp(F.col("last_updated"), F.lit("yyyy-MM-dd HH:mm:ss")))

latest = Window.partitionBy("order_line_id").orderBy(
    F.col("last_updated").desc_nulls_last(), F.col("order_id").desc_nulls_last())
sp_clean = (sp_clean.withColumn("rn", F.row_number().over(latest))
            .filter(F.col("rn") == 1).drop("rn"))
sp_clean = sp_clean.dropna(subset=ID_COLS + ["order_date", "quantity", "unit_price", "discount_pct"])
sp_clean = sp_clean.filter(
    (F.col("quantity") > 0) & (F.col("unit_price") > 0) &
    (F.col("discount_pct") >= 0) & (F.col("discount_pct") < 100))
for c in CATEGORY_COLS:
    sp_clean = sp_clean.withColumn(c,
        F.when(F.trim(F.col(c)).isNull() | (F.trim(F.col(c)) == ""), F.lit("Unknown"))
         .otherwise(F.initcap(F.trim(F.col(c)))))
sp_clean = sp_clean.withColumn("day_name", F.date_format("order_date", "EEEE"))
sp_clean = sp_clean.withColumn("customer_age", F.when(
    F.col("customer_age").between(18, 100), F.col("customer_age")))
sp_clean = sp_clean.withColumn("delivery_days", F.when(
    F.col("delivery_days").between(0, 30), F.col("delivery_days")))
sp_clean = sp_clean.withColumn("is_returned", F.when(
    F.lower(F.trim(F.col("is_returned"))).isin("true", "1", "yes"), F.lit(True))
    .when(F.lower(F.trim(F.col("is_returned"))).isin("false", "0", "no"), F.lit(False))
    .otherwise(F.lit(None).cast("boolean")))
sp_clean = sp_clean.withColumnRenamed("m_revenue", "m_revenue_original")
sp_clean = sp_clean.withColumn("m_revenue", F.round(F.col("quantity") *
    F.col("unit_price") * (1 - F.col("discount_pct") / 100), 2))

# Usable SQL table/view after cleaning; no permanent writes needed.
sp_clean.createOrReplaceTempView("sales_clean")
print("Spark cleaned rows:", sp_clean.count(), "distinct PK:",
      sp_clean.select("order_line_id").distinct().count())
sp_clean.select("order_line_id", "order_date", "category", "m_revenue").show(5)
```

## 3. Statistical and relationship analysis — Pandas + PySpark

**Key distribution/skew:** group by a key and compare counts. **Descriptives:** mean, median, mode, min, max, sample standard deviation. **Outliers:** `mean ± 3 × std` (two-sided 3-sigma test; not a universal data-cleansing rule). **Correlation:** numeric–numeric Pearson r, not proof of causality. **Pivot:** category by region with `SUM(m_revenue)`. **Histogram, bar and pie:** chart examples are in Section 5.

```python
# 3A — Pandas: skew distribution and top-key share (region/customer/category)
for key in ["region", "category", "customer_id"]:
    counts = pd_clean[key].value_counts(dropna=False)
    print(f"\n{key} top 10 counts:\n", counts.head(10).to_string())
    print(f"Most common {key} share: {counts.iloc[0] / counts.sum() * 100:.2f}%")
    print(f"Max-to-average-key-count skew ratio: {counts.max() / counts.mean():.2f}x")

# Pandas describe and explicit statistics; choose meaningful numeric columns.
print(pd_clean[NUM_COLS].describe().to_string())
print("Revenue mean:", pd_clean["m_revenue"].mean())
print("Revenue median:", pd_clean["m_revenue"].median())
print("Quantity mode(s):", pd_clean["quantity"].mode().tolist())
print("Revenue min/max:", pd_clean["m_revenue"].min(), pd_clean["m_revenue"].max())
print("Revenue sample std:", pd_clean["m_revenue"].std(ddof=1))
```

```python
# 3B — Spark: skew and descriptive statistics
for key in ["region", "category", "customer_id"]:
    key_counts = sp_clean.groupBy(key).count().orderBy(F.desc("count"))
    key_counts.show(10)
    top_count = key_counts.first()["count"]
    distinct_keys = key_counts.count()
    print(key, "top key share %:", round(100 * top_count / sp_clean.count(), 2),
          "max/avg ratio:", round(top_count / (sp_clean.count()/distinct_keys), 2))

sp_clean.select(*NUM_COLS).summary().show()
sp_clean.agg(
    F.mean("m_revenue").alias("mean_revenue"),
    F.expr("percentile_approx(m_revenue, 0.5)").alias("median_revenue"),
    F.min("m_revenue").alias("min_revenue"),
    F.max("m_revenue").alias("max_revenue"),
    F.stddev_samp("m_revenue").alias("sample_std_revenue")
).show(truncate=False)
sp_clean.groupBy("quantity").count().orderBy(F.desc("count"), "quantity").show(5)  # mode = top result
```

```python
# 3C — Outliers by 3-sigma (do NOT automatically drop genuine bulk orders)
mu = pd_clean["m_revenue"].mean()
sigma = pd_clean["m_revenue"].std(ddof=1)
lower, upper = mu - 3 * sigma, mu + 3 * sigma
pd_outliers = pd_clean.loc[
    (pd_clean["m_revenue"] < lower) | (pd_clean["m_revenue"] > upper)
].copy() if pd.notna(sigma) and sigma > 0 else pd_clean.iloc[0:0].copy()
print(f"Pandas mean={mu:.2f}, std={sigma:.2f}, outlier interval=[{lower:.2f}, {upper:.2f}]")
print(pd_outliers[["order_line_id", "product_name", "quantity", "m_revenue"]]
      .sort_values("m_revenue", ascending=False).head(10).to_string(index=False))

spark_stats = sp_clean.agg(F.avg("m_revenue").alias("mu"),
    F.stddev_samp("m_revenue").alias("sigma")).first()
if spark_stats["sigma"] is not None and spark_stats["sigma"] > 0:
    spark_lower = spark_stats["mu"] - 3 * spark_stats["sigma"]
    spark_upper = spark_stats["mu"] + 3 * spark_stats["sigma"]
    sp_outliers = sp_clean.filter((F.col("m_revenue") < spark_lower) |
                                  (F.col("m_revenue") > spark_upper))
else:
    sp_outliers = sp_clean.limit(0)
print("Spark 3-sigma outlier rows:", sp_outliers.count())
sp_outliers.select("order_line_id", "quantity", "m_revenue").show(10)
```

```python
# 3D — Numeric correlations; Pearson coefficient ranges from -1 to +1.
correlation_cols = ["quantity", "unit_price", "discount_pct", "m_revenue", "delivery_days", "customer_age"]
print("Pandas Pearson correlation matrix:\n", pd_clean[correlation_cols].corr().round(3))
for col in ["quantity", "unit_price", "discount_pct", "delivery_days"]:
    print(f"Spark corr({col}, m_revenue):", sp_clean.stat.corr(col, "m_revenue"))

# Pivot = sum of revenue by category and region (wide tabular summary).
pivot_pd = pd.pivot_table(pd_clean, index="region", columns="category",
    values="m_revenue", aggfunc="sum", fill_value=0)
print("Pandas pivot:\n", pivot_pd.round(2).to_string())
pivot_sp = sp_clean.groupBy("region").pivot("category").agg(F.sum("m_revenue")).fillna(0)
pivot_sp.show(truncate=False)   # Use only for LOW-cardinality pivot columns.

# Histogram distribution is shown in Section 5 using plt.hist/sns.histplot.
# For Spark big data, use .sample(...).limit(...).toPandas() instead of collecting everything.
```

## 4. KPI and business metrics — Pandas, PySpark and SQL

Metrics: **total revenue, mean, P90/P95, Average Order Value, category revenue contribution %, top-N contribution %, month-over-month growth %, 7-calendar-day moving average, top 5 customers, top 5 products.**

Define the denominator and data grain first. `m_revenue` is line-level; orders are distinct `order_id`. Percentiles below refer to *per-order net revenue* (AOV distribution) and line-level revenue (shown separately). For sparse dates, build a date spine before rolling averages so `rowsBetween(-6, 0)` truly represents seven **calendar days**.

```python
# 4A — Pandas KPIs: SUM, AVG, P90 / P95, AOV
order_revenue_pd = pd_clean.groupby("order_id", as_index=False)["m_revenue"].sum()
metrics_pd = {
    "total_revenue": pd_clean["m_revenue"].sum(),
    "mean_line_revenue": pd_clean["m_revenue"].mean(),
    "p90_line_revenue": pd_clean["m_revenue"].quantile(.90),
    "p95_line_revenue": pd_clean["m_revenue"].quantile(.95),
    "order_count": pd_clean["order_id"].nunique(),
    "avg_per_order_AOV": pd_clean["m_revenue"].sum() / pd_clean["order_id"].nunique(),
    "p90_order_revenue": order_revenue_pd["m_revenue"].quantile(.90),
    "p95_order_revenue": order_revenue_pd["m_revenue"].quantile(.95),
}
print(pd.Series(metrics_pd).round(2).to_string())

# Category share (%) = each category's revenue / all revenue * 100.
category_pd = pd_clean.groupby("category", as_index=False).agg(
    revenue=("m_revenue", "sum"), order_lines=("order_line_id", "count"))
category_pd["share_pct"] = 100 * category_pd["revenue"] / category_pd["revenue"].sum()
category_pd = category_pd.sort_values("revenue", ascending=False)
print("Category share:\n", category_pd.to_string(index=False))

# Top N contribution and top-5 customers / products (by revenue).
customer_pd = pd_clean.groupby("customer_id", as_index=False)["m_revenue"].sum() \
    .rename(columns={"m_revenue": "revenue"}).sort_values("revenue", ascending=False)
product_pd = pd_clean.groupby(["product_id", "product_name"], as_index=False)["m_revenue"].sum() \
    .rename(columns={"m_revenue": "revenue"}).sort_values("revenue", ascending=False)
print("Top 5 customers:\n", customer_pd.head(5).to_string(index=False))
print("Top 5 products:\n", product_pd.head(5).to_string(index=False))
print("Top 5 customers contribution %:", round(
    100 * customer_pd.head(5)["revenue"].sum() / metrics_pd["total_revenue"], 2))
print("Top 5 products contribution %:", round(
    100 * product_pd.head(5)["revenue"].sum() / metrics_pd["total_revenue"], 2))
```

```python
# 4B — Pandas monthly growth %, daily sales, true 7-calendar-day moving average
monthly_pd = pd_clean.assign(month=pd_clean["order_date"].dt.to_period("M").astype(str)) \
    .groupby("month", as_index=False)["m_revenue"].sum().rename(columns={"m_revenue": "revenue"})
monthly_pd["prev_revenue"] = monthly_pd["revenue"].shift(1)
monthly_pd["growth_pct"] = np.where(monthly_pd["prev_revenue"] > 0,
    (monthly_pd["revenue"] - monthly_pd["prev_revenue"]) /
    monthly_pd["prev_revenue"] * 100, np.nan)
print("Month-over-month growth:\n", monthly_pd.round(2).to_string(index=False))

# Resample fills every day, including zero-sales days; min_periods=1 for first 6 days.
daily_pd = (pd_clean.set_index("order_date")["m_revenue"].resample("D").sum()
            .rename("daily_revenue").to_frame())
daily_pd["moving_avg_7d"] = daily_pd["daily_revenue"].rolling(7, min_periods=1).mean()
daily_pd = daily_pd.reset_index()
print("Daily revenue / 7-day moving avg:\n", daily_pd.tail(10).round(2).to_string(index=False))
```

```python
# 4C — Spark KPIs, percentiles, category share and top customers/products
sp_orders = sp_clean.groupBy("order_id").agg(F.sum("m_revenue").alias("order_revenue"))
sp_metrics = sp_clean.agg(
    F.sum("m_revenue").alias("total_revenue"),
    F.avg("m_revenue").alias("avg_line_revenue"),
    F.expr("percentile_approx(m_revenue, 0.90)").alias("p90_line_revenue"),
    F.expr("percentile_approx(m_revenue, 0.95)").alias("p95_line_revenue"),
    F.countDistinct("order_id").alias("order_count")
).withColumn("avg_per_order_AOV", F.col("total_revenue") / F.col("order_count"))
sp_metrics.show(truncate=False)
sp_orders.agg(F.expr("percentile_approx(order_revenue, .90)").alias("p90_order"),
    F.expr("percentile_approx(order_revenue, .95)").alias("p95_order")).show()

sp_categories = sp_clean.groupBy("category").agg(F.sum("m_revenue").alias("revenue"))
sp_category_total = sp_categories.agg(F.sum("revenue").alias("all_revenue"))
sp_category_share = sp_categories.crossJoin(sp_category_total).withColumn(
    "share_pct", F.round(F.col("revenue") / F.col("all_revenue") * 100, 2)) \
    .drop("all_revenue").orderBy(F.desc("revenue"))
sp_category_share.show()

sp_customers = sp_clean.groupBy("customer_id").agg(F.sum("m_revenue").alias("revenue"))
sp_products = sp_clean.groupBy("product_id", "product_name").agg(F.sum("m_revenue").alias("revenue"))
sp_customers.orderBy(F.desc("revenue")).show(5)
sp_products.orderBy(F.desc("revenue")).show(5)

# Contribution = top-5 customer revenue / all revenue * 100.
top5_customer_revenue = sp_customers.orderBy(F.desc("revenue")).limit(5) \
    .agg(F.sum("revenue")).first()[0]
all_revenue = sp_metrics.first()["total_revenue"]
print("Spark top-5 customer revenue share %:", round(100*top5_customer_revenue/all_revenue, 2))
```

```python
# 4D — Spark growth MoM using LAG, and 7-day moving average using date spine.
sp_monthly = sp_clean.groupBy(F.date_format("order_date", "yyyy-MM").alias("month")) \
    .agg(F.sum("m_revenue").alias("revenue"))
month_window = Window.orderBy("month")
sp_monthly = sp_monthly.withColumn("prev_revenue", F.lag("revenue").over(month_window)) \
    .withColumn("growth_pct", F.when(F.col("prev_revenue") > 0,
        F.round((F.col("revenue") - F.col("prev_revenue")) / F.col("prev_revenue") * 100,2)))
sp_monthly.orderBy("month").show()

sp_daily_sum = sp_clean.groupBy("order_date").agg(F.sum("m_revenue").alias("daily_revenue"))
# Generate every date from min to max, then fill no-sales days as 0.
sp_bounds = sp_clean.agg(F.min("order_date").alias("start_date"),
                         F.max("order_date").alias("end_date"))
sp_calendar = sp_bounds.select(F.explode(F.sequence(
    F.col("start_date"), F.col("end_date"), F.expr("interval 1 day"))).alias("order_date"))
rolling7 = Window.orderBy("order_date").rowsBetween(-6, 0)
sp_daily = sp_calendar.join(sp_daily_sum, "order_date", "left") \
    .fillna({"daily_revenue": 0.0}) \
    .withColumn("moving_avg_7d", F.avg("daily_revenue").over(rolling7))
sp_daily.orderBy(F.desc("order_date")).show(10)
```

### 4E. SQL queries (Spark SQL, directly executable)

The SQL below runs on the cleaned view named `sales_clean`. The full set is also saved in `eda_kpis_spark.sql` to copy into a Databricks SQL editor. For databases other than Spark SQL, adjust `date_format`, `percentile_approx`, `sequence` and `explode` syntax.

```python
# KPI 1 — SUM, AVG, P90, P95 and AOV
spark.sql("""
SELECT ROUND(SUM(m_revenue),2) AS total_revenue,
       ROUND(AVG(m_revenue),2) AS avg_line_revenue,
       percentile_approx(m_revenue,0.90) AS p90_line_revenue,
       percentile_approx(m_revenue,0.95) AS p95_line_revenue,
       COUNT(DISTINCT order_id) AS orders,
       ROUND(SUM(m_revenue)/COUNT(DISTINCT order_id),2) AS avg_per_order
FROM sales_clean
""").show()

# KPI 2 — Category % share: category / global total * 100
spark.sql("""
WITH category_sales AS (
    SELECT category, SUM(m_revenue) AS revenue
    FROM sales_clean GROUP BY category
)
SELECT category, ROUND(revenue,2) AS revenue,
       ROUND(100*revenue/SUM(revenue) OVER (),2) AS share_pct
FROM category_sales ORDER BY revenue DESC
""").show()

# KPI 3 — Top 5 customers + contribution of TOP 5 customers
spark.sql("""
WITH customer_sales AS (
    SELECT customer_id, SUM(m_revenue) AS revenue
    FROM sales_clean GROUP BY customer_id
), ranked AS (
    SELECT customer_id, revenue,
           ROW_NUMBER() OVER (ORDER BY revenue DESC, customer_id) AS rn
    FROM customer_sales
)
SELECT customer_id, ROUND(revenue,2) AS revenue
FROM ranked WHERE rn <= 5 ORDER BY rn
""").show()
spark.sql("""
WITH cs AS (SELECT customer_id, SUM(m_revenue) AS revenue
            FROM sales_clean GROUP BY customer_id),
     ranked AS (SELECT revenue, ROW_NUMBER() OVER (ORDER BY revenue DESC) rn FROM cs)
SELECT ROUND(100 * SUM(CASE WHEN rn <= 5 THEN revenue ELSE 0 END)
             / SUM(revenue),2) AS top5_customer_contribution_pct FROM ranked
""").show()

# KPI 4 — Top 5 products
spark.sql("""
SELECT product_id, product_name, ROUND(SUM(m_revenue),2) AS revenue
FROM sales_clean GROUP BY product_id, product_name
ORDER BY revenue DESC LIMIT 5
""").show()
```

```python
# KPI 5 — Month-over-month growth, safe against missing/zero previous month
spark.sql("""
WITH month_sales AS (
    SELECT date_format(order_date,'yyyy-MM') AS month,
           SUM(m_revenue) AS revenue
    FROM sales_clean GROUP BY date_format(order_date,'yyyy-MM')
), previous AS (
    SELECT month, revenue, LAG(revenue) OVER (ORDER BY month) AS prev_revenue
    FROM month_sales
)
SELECT month, ROUND(revenue,2) AS revenue,
       CASE WHEN prev_revenue > 0
            THEN ROUND(100*(revenue-prev_revenue)/prev_revenue,2)
            ELSE NULL END AS growth_pct
FROM previous ORDER BY month
""").show()

# KPI 6 — True 7 CALENDAR day revenue moving average (including zero-sales days)
spark.sql("""
WITH limits AS (
    SELECT MIN(order_date) AS min_dt, MAX(order_date) AS max_dt FROM sales_clean
), calendar AS (
    SELECT explode(sequence(min_dt, max_dt, interval 1 day)) AS day FROM limits
), daily AS (
    SELECT order_date AS day, SUM(m_revenue) AS revenue
    FROM sales_clean GROUP BY order_date
), filled AS (
    SELECT c.day, COALESCE(d.revenue,0) AS daily_revenue
    FROM calendar c LEFT JOIN daily d ON c.day=d.day
)
SELECT day, ROUND(daily_revenue,2) AS daily_revenue,
       ROUND(AVG(daily_revenue) OVER (
           ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),2) AS moving_avg_7d
FROM filled ORDER BY day
""").show(10)
```

## 5. Visualization and business insights — Matplotlib, Seaborn, PySpark + Matplotlib

Five essential graphs: **bar (compare categories), line (trend), histogram (distribution), scatter (relationship), pie (mix/share).** Always say **finding → business meaning → suggested action**. The chart cells use cleaned data; the Spark cells first aggregate or sample, then convert the *small* result to Pandas. Spark itself is a distributed compute engine, not a plotting package.

```python
# 5A — MATPLOTLIB: 5 charts, each with a question and how to interpret it
# BAR: Which category generates the most revenue?
plt.figure(figsize=(8,4))
plt.bar(category_pd["category"], category_pd["revenue"])
plt.title("Bar — Revenue by category"); plt.xlabel("Category"); plt.ylabel("Revenue (₹)")
plt.xticks(rotation=25); plt.tight_layout(); plt.show()

# LINE: Is revenue rising/falling? Does it show spikes?
plt.figure(figsize=(10,4))
plt.plot(daily_pd["order_date"], daily_pd["daily_revenue"], alpha=.5, label="Daily")
plt.plot(daily_pd["order_date"], daily_pd["moving_avg_7d"], linewidth=2,
         label="7-day moving average")
plt.title("Line — Revenue trend"); plt.xlabel("Order date"); plt.ylabel("Revenue (₹)")
plt.legend(); plt.tight_layout(); plt.show()

# HISTOGRAM: Is line revenue right-skewed? Are a few sales very large?
plt.figure(figsize=(8,4))
plt.hist(pd_clean["m_revenue"], bins=30, edgecolor="white")
plt.title("Histogram — Line revenue distribution")
plt.xlabel("Revenue per order line (₹)"); plt.ylabel("Frequency")
plt.tight_layout(); plt.show()

# SCATTER: Does quantity correlate with revenue? What are the outliers?
plt.figure(figsize=(8,4))
plt.scatter(pd_clean["quantity"], pd_clean["m_revenue"], alpha=.5)
plt.title("Scatter — Quantity vs revenue"); plt.xlabel("Quantity"); plt.ylabel("Revenue (₹)")
plt.tight_layout(); plt.show()

# PIE: What % of total revenue comes from each category?
plt.figure(figsize=(7,6))
plt.pie(category_pd["revenue"], labels=category_pd["category"],
        autopct="%1.1f%%", startangle=90)
plt.title("Pie — Category revenue contribution"); plt.tight_layout(); plt.show()
```

```python
# 5B — SEABORN: 4 native chart types. Pie uses Matplotlib (Seaborn has no pie API).
plt.figure(figsize=(8,4))
sns.barplot(data=category_pd, x="category", y="revenue", hue="category", legend=False)
plt.title("Seaborn bar — Category revenue"); plt.xticks(rotation=25)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
sns.lineplot(data=daily_pd, x="order_date", y="moving_avg_7d")
plt.title("Seaborn line — Revenue 7-day moving average")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.histplot(data=pd_clean, x="m_revenue", bins=30, kde=True, color=sns.color_palette()[0])
plt.title("Seaborn histogram — Revenue distribution")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.scatterplot(data=pd_clean, x="quantity", y="m_revenue", hue="category", alpha=.7)
plt.title("Seaborn scatter — Quantity, revenue and category")
plt.tight_layout(); plt.show()

# Seaborn pie chart equivalent: plt.pie(...) as in cell 5A above.
```

```python
# 5C — PYSPARK: aggregate/sample in Spark, plot SMALL Pandas results using Matplotlib.
# BAR / PIE: 5 categories is safe to collect.
chart_category = sp_category_share.toPandas()
plt.figure(figsize=(8,4))
plt.bar(chart_category["category"], chart_category["revenue"])
plt.title("Spark aggregation -> Matplotlib bar")
plt.tight_layout(); plt.show()

plt.figure(figsize=(7,6))
plt.pie(chart_category["revenue"], labels=chart_category["category"], autopct="%1.1f%%")
plt.title("Spark aggregation -> Matplotlib pie")
plt.tight_layout(); plt.show()

# LINE: 150 daily rows is safe to collect.
chart_daily = sp_daily.orderBy("order_date").toPandas()
plt.figure(figsize=(10,4))
plt.plot(chart_daily["order_date"], chart_daily["moving_avg_7d"])
plt.title("Spark daily aggregation -> Matplotlib line"); plt.tight_layout(); plt.show()

# HISTOGRAM and SCATTER: do not collect millions of raw observations.
# Sample in Spark first. For tiny practice files increase sample fraction to 1.0.
chart_sample = sp_clean.select("quantity", "m_revenue").sample(
    withReplacement=False, fraction=0.6, seed=42).limit(500).toPandas()
plt.figure(figsize=(8,4))
plt.hist(chart_sample["m_revenue"], bins=25, edgecolor="white")
plt.title("Spark sampled -> Matplotlib histogram"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.scatter(chart_sample["quantity"], chart_sample["m_revenue"], alpha=.6)
plt.title("Spark sampled -> Matplotlib scatter"); plt.tight_layout(); plt.show()
```

```python
# 5D — Generate interview-ready data-driven observations, not guessed findings.
lead_category = category_pd.iloc[0]
lead_customer = customer_pd.iloc[0]
lead_product = product_pd.iloc[0]
latest_month = monthly_pd.iloc[-1]
region_share = pd_clean.groupby("region")["m_revenue"].sum().sort_values(ascending=False)
returned = pd_clean["is_returned"].dropna().astype(int)
print("INSIGHTS — finding / meaning / action")
print(f"1. Category: {lead_category['category']} leads with ₹{lead_category['revenue']:,.0f} "
      f"({lead_category['share_pct']:.1f}% share). Action: investigate its margins and inventory.")
print(f"2. Customer concentration: top 5 contribute "
      f"{100*customer_pd.head(5)['revenue'].sum()/metrics_pd['total_revenue']:.1f}% of revenue. "
      "Action: evaluate loyalty and concentration risk.")
print(f"3. Best customer: {lead_customer['customer_id']} (₹{lead_customer['revenue']:,.0f}). "
      "Action: review repeat purchases.")
print(f"4. Best product: {lead_product['product_name']} (₹{lead_product['revenue']:,.0f}). "
      "Action: check stock availability.")
print(f"5. Largest regional sales share: {region_share.index[0]} "
      f"({100*region_share.iloc[0]/region_share.sum():.1f}%). "
      "Action: assess regional dependence; do not assume it is demand without market sizing.")
print(f"6. Latest period: {latest_month['month']} growth "
      f"{latest_month['growth_pct']:.1f}%. Action: check promotions/seasonality before attributing cause.")
print(f"7. 3-sigma high/low revenue lines: {len(pd_outliers)}. "
      "Action: inspect legitimate bulk purchases before considering removal.")
print(f"8. Return rate among known labels: {100*returned.mean():.1f}% "
      f"(known labels: {len(returned)}). Action: segment by category and channel.")
print("Note: These are synthetic data. Correlation is not causation; prioritize an actionable test.")
```

## Interview explanation template (30–45 seconds)

> “I first confirmed the dataset grain and keys, checked data types and relationships, and profiled completeness, duplicate primary keys, invalid ranges and inconsistent dimensions. I created separate cleaned Pandas and Spark DataFrames, deduplicated by the latest update, standardized dimensions and recalculated net revenue. I then measured region/customer skew, descriptive statistics, three-sigma outliers, correlations and pivots. I built total sales, AOV, percentiles, category contribution, top customers/products, monthly growth and a true seven-day moving average. Finally, I used five charts to separate observations from business recommendations, checking possible seasonality, product concentration and data limitations.”

**Adapt to a new dataset:** change `DATA_FILE`, `RENAME_MAP`, the five column groups in Section 0, KPI column names and denominators, and the business validation rules. If order grain is one row per order, use `COUNT(*)` instead of `COUNT(DISTINCT order_id)`; if sales refunds are negative, change positive-value validity rules. **Never drop outliers by default.**
