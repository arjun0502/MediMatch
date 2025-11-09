# MediMatch

Predictive offer generation for clinics from inventory and order history.

This repository contains a compact implementation of a unified probability
model that predicts whether a clinic will need a specific inventory item
and generates offers accordingly.

## Repository structure

- `model_with_test_data.py` - Main script containing the probability model
  and offer-generation pipeline. Provides a CLI and documents expected
  CSV input formats.
- `data/` - Example data folder (not required) containing:
  - `inventory.csv`
  - `needslist.csv`
  - `order_history.csv`
- `offers_out.csv` - Example output produced by the script.
- `venv/` - Local Python virtual environment (ignored via `.gitignore`).

## High-level design

Inputs
- `order_history.csv` (required): historical order records. Must include
  the columns: `OUTBOUND RECIPIENT`, `ITEMIZED LIST`, `DATE REDISTRIBUTED`,
  `UNITS DISTRIBUTED`.
- `inventory.csv` (required): current inventory. Must include the columns:
  `GENERIC NAME`, `SKU ID`, `EXPIRATION_DATE`, `CASE QTY` (case quantity).
- `needslist.csv` (optional): explicit requests from clinics. Expected
  columns are `Clinic Name`, `Exact Item Name (Inventory)` and optional
  `NEEDS_ITEM` boolean-like flag.

Processing steps (simplified)
1. Load input CSVs into pandas DataFrames.
2. For each clinic×item pair present in both order history and inventory,
   compute a predicted probability of need using `calculate_probability`:
   - Compute historical reorder intervals and timing relative to last order.
   - Classify the timing as Too Soon / Approaching / Overdue and assign
     a base probability.
   - Apply a consistency bonus for predictable patterns.
   - Apply seasonal multipliers computed from historical orders.
   - If on the needslist, add a transparent +25% boost and floor.
   - Cap and floor the final probability to sensible ranges.
   - Suggest a quantity using historical units distributed (seasonal if available).
3. Skip items that are expiring within the configured threshold.
4. Include offers if they are on the needslist (always) or have probability
   above the configured threshold.
5. Output a CSV (`offers_out.csv`) listing offers sorted by clinic and
   probability.

Outputs
- `offers_out.csv`: columns include `clinic`, `sku_id`, `item`, `probability`,
  `suggested_quantity`, `available_in_stock`, `expiration_date`,
  `days_since_last_order`, `avg_reorder_interval`, `seasonal_factor`,
  `delivery_season`, `on_needslist`, `reason`.

## How to run

From PowerShell in the project root:

```powershell
# create & activate venv (local; the repo ignores venv/)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# MediMatch

Predictive offer generation for clinics from inventory and order history.

This repository contains a compact implementation of a unified probability
model that predicts whether a clinic will need a specific inventory item
and generates offers accordingly.

## Repository structure

- `model_with_test_data.py` - CSV-backed variant: the main script that
  reads local CSV files (`order_history.csv`, `inventory.csv`, optional
  `needslist.csv`) and writes `offers_out.csv`. This is useful for local
  testing and simple batch runs.
- `model_with_supabase.py` - Supabase-backed variant: the same core
  modeling logic adapted to read inputs from a Supabase database and
  write offers back to Supabase. Use this for production workflows that
  store data centrally. Requires `SUPABASE_URL` and `SUPABASE_ANON_KEY`
  environment variables.
- `test_supabase.py` - Lightweight integration/test helper for the
  Supabase workflow (example usage and sanity checks).
- `test_data/` - Example CSVs used by the CSV-backed script.
- `offers_out.csv` - Example output produced by the CSV script.
- `requirements.txt` - Lists external dependencies; includes `supabase`
  for the Supabase variant.
- `venv/` - Local Python virtual environment (ignored via `.gitignore`).

## Two model variants

There are two supported modes of operation in this repo:

1. CSV mode (`model_with_test_data.py`) — Reads and writes CSV files. This
   is ideal for local development, debugging, and running the model on
   exported datasets.
2. Supabase mode (`model_with_supabase.py`) — Connects to a Supabase
   instance, loads clinic/order/inventory data, generates offers for a
   clinic, and writes results back to a Supabase `offers` table. This is
   suitable for automated or production pipelines.

Both variants share the same modeling approach (reorder-interval timing,
consistency bonus, seasonality adjustments, and needslist boosts). The
Supabase variant contains helper functions to map database rows to the
same pandas-like structures consumed by the core model logic.

## High-level design

Inputs
- `order_history.csv` (required for CSV mode): historical order records. Must include
  the columns: `OUTBOUND RECIPIENT`, `ITEMIZED LIST`, `DATE REDISTRIBUTED`,
  `UNITS DISTRIBUTED`.
- `inventory.csv` (required for CSV mode): current inventory. Must include the columns:
  `GENERIC NAME`, `SKU ID`, `EXPIRATION_DATE`, `CASE QTY` (case quantity).
- `needslist.csv` (optional for CSV mode): explicit requests from clinics. Expected
  columns are `Clinic Name`, `Exact Item Name (Inventory)` and optional
  `NEEDS_ITEM` boolean-like flag.
- Supabase mode reads these same concepts from tables named like
  `order_history`, `inventory`, `needs_list`, and `clinics`.

Processing steps (simplified)
1. Load input data (CSV or Supabase).
2. For each clinic×item pair present in both order history and inventory,
   compute a predicted probability of need using the unified model:
   - Compute historical reorder intervals and timing relative to last order.
   - Classify the timing as Too Soon / Approaching / Overdue and assign
     a base probability.
   - Apply a consistency bonus for predictable patterns.
   - Apply seasonal multipliers computed from historical orders.
   - If on the needslist, add a transparent +25% boost and floor.
   - Cap and floor the final probability to sensible ranges.
   - Suggest a quantity using historical units distributed (seasonal if available).
3. Skip items that are expiring within the configured threshold.
4. Include offers if they are on the needslist (always) or have probability
   above the configured threshold.
5. Output offers to CSV (CSV mode) or save them to Supabase (Supabase mode).

Outputs
- `offers_out.csv` (CSV mode): columns include `clinic`, `sku_id`, `item`, `probability`,
  `suggested_quantity`, `available_in_stock`, `expiration_date`,
  `days_since_last_order`, `avg_reorder_interval`, `seasonal_factor`,
  `delivery_season`, `on_needslist`, `reason`.
- Supabase mode writes similar fields to a Supabase `offers` table.

## How to run

From PowerShell in the project root (CSV mode):

```powershell
# create & activate venv (local; the repo ignores venv/)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run CSV-backed model (uses ./test_data by default)
python model_with_test_data.py --order_history ./test_data/order_history.csv --inventory ./test_data/inventory.csv --needslist ./test_data/needslist.csv --out_csv offers_out.csv
```

Supabase mode (requires credentials in env vars):

```powershell
# set environment variables (PowerShell example)
$env:SUPABASE_URL = 'https://<your-instance>.supabase.co'
$env:SUPABASE_ANON_KEY = '<anon-key>'

# run Supabase-backed model (generate offers for a clinic)
python model_with_supabase.py --clinic-id <clinic-uuid> --delivery-days 3
```

See `test_supabase.py` for example usage and quick integration checks.

## Notes & suggestions

- The model is intentionally simple and transparent so it can be tuned.
- Consider adding unit tests that assert expected probabilities for
  contrived histories (happy path + edge cases like very sparse history).
- If you want the repository to not include the virtual environment, run
  these git commands to untrack an already committed `venv/` directory:

```powershell
# remove from index while keeping local files
git rm -r --cached venv
git commit -m "Stop tracking venv/"
```

## Contact
If you need help modifying the model or adding tests, open an issue or
reach out to the repo owner.
