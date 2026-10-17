# Warehouse Operations & Process Improvement Analysis

**Question:** Where is the warehouse losing time, why might it be happening, and what should management investigate first?

## Tools
- Python (pandas, matplotlib): data cleaning, KPIs, analysis, charts
- Excel: PivotTables, dashboard, data quality checks and documentation

## Data
Synthetic dataset of 10,000 warehouse orders (August 2026), generated with Excel formulas with deliberate patterns (for example, Zone C slower, evenings busier, new workers slower). This is not Amazon data.

## Method
1. Generate the dataset (Excel)
2. Clean and validate it: missing values, duplicates, data types, impossible values (Python and Excel)
3. Calculate KPIs: processing time, error rate, delayed-order % (delayed = over 15 minutes, an assumption)
4. Investigate by zone, shift, workload and worker experience
5. Cross-check the Python results against the Excel PivotTables (they match)
6. Build an Excel dashboard and a short report

## Key findings
- Zone C has the highest processing time (14.1 min), error rate (7.7%) and delayed rate (33.8% of its own orders). It handles about 25% of all orders but accounts for about 62% of all delayed orders.
- Evening is the slowest shift (12.8 min); Zone C in the evening is the worst combination (15.0 min).
- New workers are slower and less accurate than experienced workers.
- Delays rise with workload.

These are patterns that warrant investigation, not proven causes.

## Recommendation
Prioritise an operational review of Zone C during peak evening periods (staffing, product placement, new-worker support), and test an additional experienced picker in the busiest evening period, comparing against the current baseline.

## Limitations
Simulated student dataset built to demonstrate the method. Patterns were built into the data, so results show the analytical approach rather than real warehouse behaviour.

## How to run
1. Install Python, then run: pip install -r requirements.txt
2. Open python/warehouse_analysis.ipynb in Jupyter
3. Run all cells. The notebook reads data/raw_data.csv (generated in Excel) and writes data/clean_data.csv and the charts in charts/.

## Folder structure
- data/ raw_data.csv, clean_data.csv
- python/ warehouse_analysis.ipynb
- charts/ five PNG charts
- excel/ warehouse_dashboard.xlsx
- report/ project_summary.pdf