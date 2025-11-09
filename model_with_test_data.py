"""
MediMatch offer generation model

This module contains the unified probability model and offer generation
pipeline used to predict which clinic×item combinations should be offered
from inventory. It's provided with lightweight test-data helpers and a
CLI entrypoint for running against CSV files.

High-level flow (input -> output):

- Inputs:
    - order_history.csv: historical orders with columns:
            OUTBOUND RECIPIENT, ITEMIZED LIST, DATE REDISTRIBUTED, UNITS DISTRIBUTED
    - inventory.csv: current inventory with columns:
            GENERIC NAME, SKU ID, EXPIRATION_DATE, CASE QTY
    - needslist.csv (optional): explicit clinic requests with columns:
            Clinic Name, Exact Item Name (Inventory), NEEDS_ITEM

- Process:
    1. Load CSVs
    2. For each clinic×item present in history and inventory, run
         `calculate_probability(...)` which:
             - computes reorder intervals from history
             - classifies timing (too soon / approaching / overdue)
             - applies consistency and seasonal adjustments
             - boosts items on the needslist
             - returns a probability and suggested quantity
    3. Filter out items expiring soon and those below the predicted
         probability threshold (unless they're on the needslist)
    4. Emit `offers_out.csv` containing offers sorted by clinic and
         probability

- Output: `offers_out.csv` with columns including:
    clinic, sku_id, item, probability, suggested_quantity, available_in_stock,
    expiration_date, days_since_last_order, avg_reorder_interval,
    seasonal_factor, delivery_season, on_needslist, reason

Design notes
- The probability calculation blends timing (intervals), seasonality
        (clinic×item seasonal multipliers), and explicit needslist boosts. The
        model is intentionally transparent and clamped to sensible floors/caps
        so downstream users can understand and tune behavior.

Variants
- This repository contains two variants of the pipeline:
    - CSV-backed (`model_with_test_data.py`) — reads/writes CSV files and is
        intended for local testing and batch runs.
    - Supabase-backed (`model_with_supabase.py`) — reads inputs from and
        writes outputs to a Supabase instance for production automation. The
        Supabase module contains helpers to translate database rows into the
        same pandas-compatible structures used here.

Usage (from repository root, PowerShell):

    # create a venv (not checked into git)
    python -m venv .venv;
    .\.venv\Scripts\Activate.ps1;
    pip install -r requirements.txt  # create file if you need one: pandas,numpy

    # run with test CSVs (defaults in CLI)
    python model_with_test_data.py --order_history ./test_data/order_history.csv --inventory ./test_data/inventory.csv --needslist ./test_data/needslist.csv

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys

# ----------------------------
# Season helpers
# ----------------------------
def get_season(date):
    m = date.month
    if m in [12, 1, 2]:
        return 'Winter'
    elif m in [3, 4, 5]:
        return 'Spring'
    elif m in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def calculate_seasonal_preference(order_history, clinic, item):
    """Return dict of season -> multiplier based on this clinic×item's history."""
    orders = order_history[
        (order_history['OUTBOUND RECIPIENT'] == clinic) &
        (order_history['ITEMIZED LIST'] == item)
    ].copy()

    if len(orders) < 4:
        return {'Winter': 1.0, 'Spring': 1.0, 'Summer': 1.0, 'Fall': 1.0}

    orders['season'] = orders['DATE REDISTRIBUTED'].apply(get_season)
    seasonal_counts = orders.groupby('season').size()
    total = len(orders)

    multipliers = {}
    for s in ['Winter', 'Spring', 'Summer', 'Fall']:
        actual_pct = (seasonal_counts.get(s, 0) / total) if total > 0 else 0.25
        multipliers[s] = actual_pct / 0.25  # deviation from uniform 25%
    return multipliers

# ----------------------------
# Expiration guard
# ----------------------------
def is_expiring_soon(expiration_date, current_date, days_threshold=30):
    if pd.isna(expiration_date) or expiration_date == 'N/A':
        return False
    try:
        exp = pd.to_datetime(expiration_date)
        curr = pd.to_datetime(current_date)
        return (exp - curr).days <= days_threshold
    except Exception:
        return False

# ----------------------------
# Unified Probability Model
# ----------------------------
def _safe_mean(series, default=50):
    s = pd.to_numeric(series, errors='coerce').dropna()
    return int(round(s.mean())) if len(s) else default

def calculate_probability(order_history, clinic, item, current_date, delivery_days=3, on_needslist=False):
    """
    UNIFIED probability calculation with needslist boost.
    
    PHILOSOPHY:
    - Calculate base probability (0.05-0.75) using timing + seasonality for ALL items
    - If on needslist, add +0.25 bonus to prioritize explicit requests
    - This ensures needslist items (0.30-1.00) always rank above predicted (0.05-0.75)
    
    CALCULATION STEPS:
    1. Analyze historical ordering patterns (intervals between orders)
    2. Assess current timing (are they due for reorder?)
    3. Apply seasonal adjustment (do they order more in this season?)
    4. Add needslist bonus if applicable
    5. Apply final caps/floors
    """
    orders = order_history[
        (order_history['OUTBOUND RECIPIENT'] == clinic) &
        (order_history['ITEMIZED LIST'] == item)
    ].copy()

    orders['DATE REDISTRIBUTED'] = pd.to_datetime(orders['DATE REDISTRIBUTED'])
    orders = orders.sort_values('DATE REDISTRIBUTED')

    # INSUFFICIENT DATA CASE: Use conservative defaults
    if len(orders) < 3:
        base_prob = 0.50 if on_needslist else 0.20
        return {
            'probability': base_prob,
            'reason': f"{'On needslist - ' if on_needslist else ''}Limited order history (<3 orders)",
            'suggested_quantity': _safe_mean(orders['UNITS DISTRIBUTED']),
            'days_since_last': None,
            'avg_interval': None,
            'seasonal_factor': 1.0,
            'delivery_season': get_season(pd.to_datetime(current_date) + timedelta(days=delivery_days)),
            'on_needslist': on_needslist
        }

    # HISTORICAL PATTERN ANALYSIS
    intervals = orders['DATE REDISTRIBUTED'].diff().dt.days.dropna()
    avg_interval = intervals.mean()
    min_interval = intervals.min()
    std_interval = intervals.std(ddof=0) if len(intervals) > 1 else 0.0

    last_order = orders['DATE REDISTRIBUTED'].max()
    current_dt = pd.to_datetime(current_date)
    days_since_last = (current_dt - last_order).days

    delivery_date = current_dt + timedelta(days=delivery_days)
    delivery_adjusted_days = days_since_last + delivery_days

    # TIMING ASSESSMENT: Calculate base probability from reorder patterns
    # Three scenarios based on where they are in their ordering cycle
    
    if delivery_adjusted_days < 0.8 * min_interval:
        # TOO SOON: Haven't even reached 80% of their shortest interval
        base = 0.10
        reason = f"Too soon (last {days_since_last}d, min interval {min_interval:.0f}d)"
        
    elif delivery_adjusted_days >= avg_interval:
        # OVERDUE: Past their average reorder time - likely need it
        overdue_factor = (delivery_adjusted_days - avg_interval) / max(avg_interval, 1e-9)
        base = 0.50 + 0.25 * min(overdue_factor, 1.0)  # 0.50 to 0.75
        reason = f"Due for reorder (avg {avg_interval:.0f}d, {days_since_last}d since last)"
        
    else:
        # APPROACHING: In the window between min and avg interval
        denom = max(avg_interval - min_interval, 1e-9)
        progress = (delivery_adjusted_days - min_interval) / denom
        base = 0.25 + 0.25 * np.clip(progress, 0.0, 1.0)  # 0.25 to 0.50
        reason = f"Approaching reorder window ({days_since_last}d since last, avg {avg_interval:.0f}d)"

    # CONSISTENCY BONUS: Reward predictable ordering patterns
    if avg_interval > 0 and (std_interval / avg_interval) < 0.3:
        base = min(base * 1.1, 0.65 if not on_needslist else 0.75)  # +10% bonus, different caps
        reason += " (consistent pattern)"

    # SEASONAL ADJUSTMENT: Modify based on historical seasonal demand
    seasonal_multipliers = calculate_seasonal_preference(order_history, clinic, item)
    delivery_season = get_season(delivery_date)
    seasonal_factor = seasonal_multipliers[delivery_season]

    adjusted = base * seasonal_factor
    
    # Apply seasonal caps/floors
    if seasonal_factor > 1.2:
        reason += f" (HIGH demand in {delivery_season})"
        adjusted = min(adjusted, 0.65 if not on_needslist else 0.75)  # Cap predicted lower
    elif seasonal_factor < 0.8:
        reason += f" (LOW demand in {delivery_season})"
        adjusted = max(adjusted, 0.05)
    else:
        adjusted = float(np.clip(adjusted, 0.05, 0.65 if not on_needslist else 0.75))

    # NEEDSLIST BOOST: Add transparent bonus for explicit requests
    if on_needslist:
        adjusted = min(adjusted + 0.25, 1.0)  # +25 point boost, cap at 100%
        adjusted = max(adjusted, 0.70)  # Floor at 70% for needslist items
        reason = f"On needslist (+25% boost) - {reason}"

    # QUANTITY SUGGESTION: Use seasonal data if available
    seasonal_orders = orders[orders['DATE REDISTRIBUTED'].apply(get_season) == delivery_season]
    if len(seasonal_orders) >= 2:
        suggested_qty = _safe_mean(seasonal_orders['UNITS DISTRIBUTED'])
    else:
        suggested_qty = _safe_mean(orders['UNITS DISTRIBUTED'])

    return {
        'probability': adjusted,
        'reason': reason,
        'suggested_quantity': suggested_qty,
        'days_since_last': days_since_last,
        'avg_interval': avg_interval,
        'seasonal_factor': seasonal_factor,
        'delivery_season': delivery_season,
        'on_needslist': on_needslist
    }

# ----------------------------
# Offer generation
# ----------------------------
def generate_offers(order_history, inventory, needslist=None, current_date='2025-11-08',
                   delivery_days=3, min_probability_predicted=0.40, expiration_threshold_days=30):
    """
    Generate ranked offer list for all clinics.
    
    STRATEGY:
    1. For each clinic×item with order history
    2. Calculate probability using unified method
    3. Add needslist boost if applicable
    4. Include in offers if:
       - On needslist (always include, regardless of probability)
       - NOT on needslist AND probability >= threshold (default 40%)
    5. Sort by probability (needslist items naturally rank highest)
    """

    order_history = order_history.copy()
    order_history['DATE REDISTRIBUTED'] = pd.to_datetime(order_history['DATE REDISTRIBUTED'])

    # Map inventory to expected columns and types
    inv = inventory.copy()
    inv = inv.rename(columns={
        'GENERIC NAME': 'ITEM',
        'CASE QTY': 'QUANTITY_AVAILABLE'
    })
    inv['QUANTITY_AVAILABLE'] = pd.to_numeric(inv['QUANTITY_AVAILABLE'], errors='coerce').fillna(0).astype(int)

    inventory_items = set(inv['ITEM'].values)

    all_offers = []

    # All clinic×item pairs that exist in history AND are in inventory
    pairs = (order_history.groupby(['OUTBOUND RECIPIENT', 'ITEMIZED LIST'])
             .size().reset_index(name='order_count'))
    pairs = pairs[pairs['ITEMIZED LIST'].isin(inventory_items)]

    for _, row in pairs.iterrows():
        clinic = row['OUTBOUND RECIPIENT']
        item = row['ITEMIZED LIST']

        item_inventory = inv[inv['ITEM'] == item].iloc[0]
        available_qty = int(item_inventory['QUANTITY_AVAILABLE'])
        expiration = item_inventory.get('EXPIRATION_DATE', 'N/A')
        sku_id = item_inventory.get('SKU ID', None)

        # Skip items expiring soon
        if is_expiring_soon(expiration, current_date, expiration_threshold_days):
            continue

        # Check if on needslist
        on_needslist = False
        if needslist is not None and len(needslist) > 0:
            needs_flag = needslist['NEEDS_ITEM'] if 'NEEDS_ITEM' in needslist.columns else True
            needs_mask = (
                (needslist['Clinic Name'] == clinic) &
                (needslist['Exact Item Name (Inventory)'] == item) &
                needs_flag
            )
            on_needslist = bool(needs_mask.any())

        # Calculate probability using unified method
        pred = calculate_probability(
            order_history, clinic, item, current_date, delivery_days, on_needslist
        )

        # Decision: Include this offer?
        include_offer = on_needslist or (pred['probability'] >= min_probability_predicted)

        if include_offer:
            all_offers.append({
                'clinic': clinic,
                'sku_id': sku_id,
                'item': item,
                'probability': pred['probability'],
                'suggested_quantity': min(int(pred['suggested_quantity']), available_qty),
                'available_in_stock': available_qty,
                'expiration_date': expiration,
                'days_since_last_order': pred['days_since_last'],
                'avg_reorder_interval': pred['avg_interval'],
                'seasonal_factor': pred['seasonal_factor'],
                'delivery_season': pred['delivery_season'],
                'on_needslist': on_needslist,
                'reason': pred['reason'],
            })

    offers_df = pd.DataFrame(all_offers)
    if not offers_df.empty:
        # Sort by clinic, then probability (needslist items naturally rank highest)
        offers_df = offers_df.sort_values(['clinic', 'probability'], ascending=[True, False])
    
    return offers_df

def generate_offers_for_clinic(order_history, inventory, clinic_name, needslist=None,
                               current_date='2025-11-08', delivery_days=3, expiration_threshold_days=30):
    """Generate offers for a specific clinic."""
    all_offers = generate_offers(
        order_history, inventory, needslist, current_date, delivery_days,
        min_probability_predicted=0.0, expiration_threshold_days=expiration_threshold_days
    )
    return all_offers[all_offers['clinic'] == clinic_name].reset_index(drop=True)

# ----------------------------
# CLI / Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate clinic-item offers using unified probability model.")
    parser.add_argument("--order_history", default="./test_data/order_history.csv", help="Path to order_history.csv")
    parser.add_argument("--inventory", default="./test_data/inventory.csv", help="Path to inventory.csv (GENERIC NAME, SKU ID, EXPIRATION_DATE, CASE QTY)")
    parser.add_argument("--needslist", default="./test_data/needslist.csv", help="Optional path to needslist.csv (Clinic Name, Exact Item Name (Inventory), NEEDS_ITEM)")
    parser.add_argument("--current_date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--delivery_days", type=int, default=3)
    parser.add_argument("--predicted_threshold", type=float, default=0.40)
    parser.add_argument("--expiration_threshold_days", type=int, default=30)
    parser.add_argument("--out_csv", default="offers_out.csv")
    args = parser.parse_args()

    # Load data
    oh = pd.read_csv(args.order_history)
    inv = pd.read_csv(args.inventory)

    # Load needslist if file exists; otherwise treat as None
    try:
        nl = pd.read_csv(args.needslist) if args.needslist else None
    except FileNotFoundError:
        nl = None

    # Basic column checks
    required_oh = {'OUTBOUND RECIPIENT','ITEMIZED LIST','DATE REDISTRIBUTED','UNITS DISTRIBUTED'}
    missing_oh = required_oh - set(oh.columns)
    if missing_oh:
        sys.exit(f"order_history.csv is missing columns: {missing_oh}")

    required_inv = {'GENERIC NAME','SKU ID','EXPIRATION_DATE','CASE QTY'}
    missing_inv = required_inv - set(inv.columns)
    if missing_inv:
        sys.exit(f"inventory.csv is missing columns: {missing_inv}")

    if nl is not None:
        required_nl = {'Clinic Name','Exact Item Name (Inventory)'}
        missing_nl = required_nl - set(nl.columns)
        if missing_nl:
            sys.exit(f"needslist.csv is missing columns: {missing_nl}")

    # Generate offers
    offers = generate_offers(
        order_history=oh,
        inventory=inv,
        needslist=nl,
        current_date=args.current_date,
        delivery_days=args.delivery_days,
        min_probability_predicted=args.predicted_threshold,
        expiration_threshold_days=args.expiration_threshold_days
    )

    # Save & print summary
    offers.to_csv(args.out_csv, index=False)
    print(f"\nSaved offers to: {args.out_csv}")
    if not offers.empty:
        print(f"Total offers: {len(offers)}")
        print(f"  Needslist: {offers['on_needslist'].sum()}")
        print(f"  Predicted: {(~offers['on_needslist']).sum()}")
        print("\nTop 20 offers:")
        cols_show = ['clinic','sku_id','item','probability','on_needslist','reason']
        cols_show = [c for c in cols_show if c in offers.columns]
        print(offers[cols_show].head(20).to_string(index=False))
    else:
        print("No offers generated with current constraints.")

if __name__ == "__main__":
    main()