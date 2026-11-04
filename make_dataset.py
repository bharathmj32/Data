"""Deterministically generate a realistic *intentionally imperfect* sales CSV for EDA practice.
Run: python make_dataset.py
Uses only the Python standard library.
"""
import csv
import random
from datetime import date, datetime, timedelta
from pathlib import Path

rng = random.Random(2026)
PRODUCTS = [
    ('P001', 'Wireless Headphones', 'Electronics', 2699),
    ('P002', 'Smart Watch', 'Electronics', 4599),
    ('P003', 'Bluetooth Speaker', 'Electronics', 1899),
    ('P004', 'USB-C Charger', 'Electronics', 999),
    ('P005', 'Running Shoes', 'Fashion', 2499),
    ('P006', 'Cotton T-Shirt', 'Fashion', 799),
    ('P007', 'Coffee Maker', 'Home', 3499),
    ('P008', 'Bedsheet Set', 'Home', 1599),
    ('P009', 'Rice Bag 5kg', 'Grocery', 549),
    ('P010', 'Olive Oil', 'Grocery', 849),
    ('P011', 'Yoga Mat', 'Sports', 1199),
    ('P012', 'Badminton Racket', 'Sports', 2299),
]
HEADERS = [
    'order_line_id','order_id','order_date','last_updated','customer_id',
    'product_id','product_name','category','region','channel','day_name',
    'quantity','unit_price','discount_pct','m_revenue','delivery_days',
    'customer_age','is_returned'
]
START = date(2026, 1, 1)

def noisy_case(s):
    return rng.choice([s, s.lower(), s.upper(), s[0].lower() + s[1:].upper(), '  ' + s + '  '])

rows = []
line_counter = 0
# 520 distinct orders. Each order contains one to three item lines.
# The nonuniform choices purposely create high-frequency products/regions/customers.
for order_n in range(1, 521):
    when = START + timedelta(days=rng.randrange(150))
    cust = rng.choices([f'C{x:03}' for x in range(1, 81)], weights=[16 if x <= 8 else 2 if x <= 28 else 1 for x in range(1, 81)])[0]
    region = rng.choices(['North','South','East','West','Central'], [57,18,11,9,5])[0]
    channel = rng.choices(['Online','Retail','Partner'], [60,28,12])[0]
    age = rng.randint(18, 70)
    n_lines = rng.choices([1,2,3], [55,35,10])[0]
    product_indices = rng.sample(range(len(PRODUCTS)), k=n_lines)
    for idx in product_indices:
        line_counter += 1
        sku, product_name, category, base_price = PRODUCTS[idx]
        qty = rng.choices([1,2,3,4,5,6,7,8], [42,25,12,9,5,3,2,2])[0]
        unit_price = round(base_price * rng.uniform(0.9, 1.12), 2)
        discount = rng.choice([0,0,0,5,10,15,20])
        revenue = round(qty * unit_price * (1 - discount / 100), 2)
        last_updated = datetime.combine(when, datetime.min.time()).replace(hour=rng.randint(8,21), minute=rng.randint(0,59))
        rows.append(dict(zip(HEADERS, [
            f'L{line_counter:05}', f'O{order_n:05}', when.isoformat(),
            last_updated.strftime('%Y-%m-%d %H:%M:%S'), cust,
            sku, product_name, noisy_case(category), noisy_case(region),
            noisy_case(channel), noisy_case(when.strftime('%A')),
            qty, unit_price, discount, revenue, rng.randint(1,12),
            age, rng.choice(['true','false','false','false','false','false','false','false'])
        ])))

# Bulk order = a *valid* unusually high revenue observation for outlier detection.
rows[51]['quantity'] = 55
rows[51]['m_revenue'] = round(55 * float(rows[51]['unit_price']) * (1 - float(rows[51]['discount_pct'])/100), 2)
rows[248]['quantity'] = 48
rows[248]['m_revenue'] = round(48 * float(rows[248]['unit_price']) * (1 - float(rows[248]['discount_pct'])/100), 2)

# Known defects: missing fields, invalid ranges, invalid dates, malformed metrics,
# conflicting timestamps/IDs, inconsistent casing and unknown Boolean label.
for i, key in [(8,'customer_id'),(46,'category'),(92,'region'),(123,'unit_price'),
               (187,'m_revenue'),(205,'product_id'),(285,'order_date'),
               (363,'quantity'),(471,'customer_age'),(590,'channel')]:
    rows[i][key] = ''
rows[10]['quantity'] = 0
rows[32]['quantity'] = -3
rows[76]['unit_price'] = -250
rows[110]['discount_pct'] = 125
rows[151]['discount_pct'] = -5
rows[192]['customer_age'] = 157
rows[215]['customer_age'] = 0
rows[260]['delivery_days'] = -4
rows[271]['delivery_days'] = 80
rows[290]['order_date'] = '2026-02-30'
rows[311]['m_revenue'] = -200
rows[330]['m_revenue'] = 0
rows[350]['m_revenue'] = 999999.00
rows[419]['is_returned'] = 'unknown'
rows[450]['day_name'] = 'SunDay' # deliberately mismatches date of transaction too

# Identical duplicate rows (full-row duplication)
for i in [12, 86, 199, 300, 499]:
    rows.append(rows[i].copy())

# Conflicting duplicate primary keys; later update should win.
for i in [62, 171, 239, 420, 680]:
    amended = rows[i].copy()
    when = datetime.strptime(amended['last_updated'], '%Y-%m-%d %H:%M:%S') + timedelta(days=2)
    amended['last_updated'] = when.strftime('%Y-%m-%d %H:%M:%S')
    amended['quantity'] = int(amended['quantity']) + 1
    amended['m_revenue'] = round(int(amended['quantity']) * float(amended['unit_price']) *
                                 (1 - float(amended['discount_pct']) / 100), 2)
    rows.append(amended)

rng.shuffle(rows)
output = Path(__file__).with_name('sales_eda_practice.csv')
with output.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=HEADERS)
    writer.writeheader()
    writer.writerows(rows)
print(f'Created {output} ({len(rows):,} rows, {len(HEADERS)} columns)')
