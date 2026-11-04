-- EDA INTERVIEW | Spark SQL | Run only AFTER the notebook/script creates
-- the temporary view `sales_clean` using sp_clean.createOrReplaceTempView("sales_clean").
-- Alternatively, persist in Databricks: sp_clean.write.mode("overwrite").format("delta").saveAsTable("sales_clean_demo")
-- and replace `sales_clean` below with `sales_clean_demo`.

-- 0. Sample records + total raw rows (raw view created in setup)
SELECT * FROM sales_clean LIMIT 5;
SELECT COUNT(*) AS clean_rows, COUNT(DISTINCT order_line_id) AS unique_order_lines,
       COUNT(DISTINCT order_id) AS orders, COUNT(DISTINCT customer_id) AS customers
FROM sales_clean;

-- 4A. SUM, AVG, P90/P95 of LINE sales, and AOV: total revenue / distinct orders
SELECT ROUND(SUM(m_revenue),2) AS total_revenue,
       ROUND(AVG(m_revenue),2) AS mean_line_revenue,
       percentile_approx(m_revenue, 0.90) AS p90_line_revenue,
       percentile_approx(m_revenue, 0.95) AS p95_line_revenue,
       COUNT(DISTINCT order_id) AS order_count,
       ROUND(SUM(m_revenue)/COUNT(DISTINCT order_id),2) AS avg_per_order
FROM sales_clean;

-- 4B. Median, P90, P95 of revenue per ORDER (not line)
WITH per_order AS (
    SELECT order_id, SUM(m_revenue) AS order_revenue
    FROM sales_clean GROUP BY order_id
)
SELECT percentile_approx(order_revenue, 0.50) AS median_order_revenue,
       percentile_approx(order_revenue, 0.90) AS p90_order_revenue,
       percentile_approx(order_revenue, 0.95) AS p95_order_revenue
FROM per_order;

-- 4C. Category share (%) = category revenue / grand total * 100
WITH category_sales AS (
    SELECT category, SUM(m_revenue) AS revenue
    FROM sales_clean GROUP BY category
)
SELECT category, ROUND(revenue,2) AS revenue,
       ROUND(100*revenue/SUM(revenue) OVER (),2) AS share_pct
FROM category_sales ORDER BY revenue DESC;

-- 4D. Top 5 customers by revenue
SELECT customer_id, ROUND(SUM(m_revenue),2) AS revenue
FROM sales_clean GROUP BY customer_id
ORDER BY revenue DESC LIMIT 5;

-- 4E. Top 5 products by revenue
SELECT product_id, product_name, ROUND(SUM(m_revenue),2) AS revenue
FROM sales_clean GROUP BY product_id, product_name
ORDER BY revenue DESC LIMIT 5;

-- 4F. Contribution: top 5 customers' sales / all customers' sales * 100
WITH cs AS (
    SELECT customer_id, SUM(m_revenue) AS revenue
    FROM sales_clean GROUP BY customer_id
), ranked AS (
    SELECT customer_id, revenue,
           ROW_NUMBER() OVER (ORDER BY revenue DESC, customer_id) AS rn FROM cs
)
SELECT ROUND(100 * SUM(CASE WHEN rn <= 5 THEN revenue ELSE 0 END)
             / SUM(revenue),2) AS top_5_customers_contribution_pct
FROM ranked;

-- 4G. Contribution: top 5 products' sales / all products' sales * 100
WITH ps AS (
    SELECT product_id, SUM(m_revenue) AS revenue
    FROM sales_clean GROUP BY product_id
), ranked AS (
    SELECT product_id, revenue,
           ROW_NUMBER() OVER (ORDER BY revenue DESC, product_id) AS rn FROM ps
)
SELECT ROUND(100 * SUM(CASE WHEN rn <= 5 THEN revenue ELSE 0 END)
             / SUM(revenue),2) AS top_5_products_contribution_pct
FROM ranked;

-- 4H. Month-over-month growth: (current - previous) / previous * 100
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
FROM previous ORDER BY month;

-- 4I. True 7-calendar-day moving average, including zero-sale days
WITH limits AS (
    SELECT MIN(order_date) AS min_dt, MAX(order_date) AS max_dt FROM sales_clean
), calendar AS (
    SELECT explode(sequence(min_dt,max_dt,interval 1 day)) AS day FROM limits
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
FROM filled ORDER BY day;

-- 3. PIVOT support: grouped category revenue by region (before plotting bar/pie)
SELECT region, category, ROUND(SUM(m_revenue),2) AS revenue
FROM sales_clean GROUP BY region, category ORDER BY region, revenue DESC;
