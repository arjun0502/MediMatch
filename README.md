# MediMatch

Predictive offer generation for clinics using inventory and order history.

MediMatch predicts which items a clinic is likely to need and suggests
offers with a probability score and recommended quantity. It supports two
modes: CSV-based (for testing) and Supabase-backed (for production).

## Highlights
- Predicts need using reorder timing, order consistency, and seasonal patterns
- Honors explicit clinic requests (needs list) with a transparent boost
- Skips near-expiry stock and suggests quantities based on historical units

## Repo layout (important files)
- `model_with_test_data.py` — CSV-mode pipeline (reads CSVs, writes `offers_out.csv`)
- `model_with_supabase.py` — Production pipeline (reads/writes Supabase)
- `test_supabase.py` — DB connectivity/schema helper
- `test_data/` — example CSVs for quick testing
- `offers_out.csv` — sample output produced by the script

## Quick start (CSV mode)
1. Create & activate a venv, then install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run with example data:

```powershell
python model_with_test_data.py \
  --order_history ./test_data/order_history.csv \
  --inventory ./test_data/inventory.csv \
  --needslist ./test_data/needslist.csv \
  --out_csv offers_out.csv
```

## Quick start (Supabase mode)
Set Supabase env vars, then run for a clinic:

```powershell
$env:SUPABASE_URL = 'https://<your-instance>.supabase.co'
$env:SUPABASE_ANON_KEY = '<anon-key>'
python model_with_supabase.py --clinic-name "Community Health Clinic"
```

## Input requirements (CSV mode)
- `order_history.csv`: must include `OUTBOUND RECIPIENT`, `ITEMIZED LIST`, `DATE REDISTRIBUTED`, `UNITS DISTRIBUTED`
- `inventory.csv`: must include `GENERIC NAME`, `SKU ID`, `EXPIRATION_DATE`, `CASE QTY`
- `needslist.csv` (optional): should include `Clinic Name`, `Exact Item Name (Inventory)`, optional `NEEDS_ITEM`

## What the model does (concise)
For each clinic×item pair present in history and inventory it:
- Computes average reorder interval and days since last order
- Classifies timing (too-soon / approaching / overdue) and assigns a base probability
- Applies a consistency bonus and seasonal multiplier where applicable
- Adds a transparent boost if the item appears on a needs list
- Floors/caps probabilities and suggests quantities from historical units
- Filters out items expiring within the configured threshold

Output (`offers_out.csv`) includes columns such as `clinic`, `sku_id`, `item`, `probability`, `suggested_quantity`, `available_in_stock`, `expiration_date`, `days_since_last_order`, `avg_reorder_interval`, `seasonal_factor`, `on_needslist`, and a `reason`.

## Configuration / tuning
- `--predicted_threshold` (default 0.40): min probability to include items not on needs list
- `--delivery_days` (default 3): delivery lead time used in timing calculations
- `--expiration_threshold_days` (default 30): skip items expiring within this window

## Contributing
- Open issues for bugs or feature requests
- PRs welcome — include tests for new behavior

