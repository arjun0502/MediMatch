# recommend_offers.py
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
# Probability models
# ----------------------------
def _safe_mean(series, default=50):
    s = pd.to_numeric(series, errors='coerce').dropna()
    return int(round(s.mean())) if len(s) else default

def calculate_needslist_probability(order_history, clinic, item, current_date, delivery_days=3):
    orders = order_history[
        (order_history['OUTBOUND RECIPIENT'] == clinic) &
        (order_history['ITEMIZED LIST'] == item)
    ].copy()

    orders['DATE REDISTRIBUTED'] = pd.to_datetime(orders['DATE REDISTRIBUTED'])
    orders = orders.sort_values('DATE REDISTRIBUTED')

    if len(orders) < 3:
        return {
            'probability': 0.75,
            'reason': 'On needslist (limited order history)',
            'suggested_quantity': _safe_mean(orders['UNITS DISTRIBUTED']),
            'days_since_last': None,
            'avg_interval': None,
            'seasonal_factor': 1.0,
            'delivery_season': get_season(pd.to_datetime(current_date) + timedelta(days=delivery_days))
        }

    intervals = orders['DATE REDISTRIBUTED'].diff().dt.days.dropna()
    avg_interval = intervals.mean()

    last_order = orders['DATE REDISTRIBUTED'].max()
    current_dt = pd.to_datetime(current_date)
    days_since_last = (current_dt - last_order).days

    delivery_date = current_dt + timedelta(days=delivery_days)
    delivery_adjusted_days = days_since_last + delivery_days

    seasonal_multipliers = calculate_seasonal_preference(order_history, clinic, item)
    delivery_season = get_season(delivery_date)
    seasonal_factor = seasonal_multipliers[delivery_season]

    # Lenient window for needslist items
    if delivery_adjusted_days >= 0.8 * avg_interval:
        base = 0.85
        reason = f"On needslist, due for reorder (avg {avg_interval:.0f}d, {days_since_last}d since last)"
    else:
        progress = delivery_adjusted_days / max(0.8 * avg_interval, 1e-9)
        base = 0.65 + 0.20 * min(progress, 1.0)
        reason = f"On needslist ({days_since_last}d since last, avg {avg_interval:.0f}d)"

    adjusted = base * seasonal_factor
    if seasonal_factor > 1.2:
        reason += f" (HIGH demand in {delivery_season})"
        adjusted = min(adjusted, 0.95)
    elif seasonal_factor < 0.8:
        reason += f" (LOW demand in {delivery_season})"
        adjusted = max(adjusted, 0.60)
    else:
        adjusted = float(np.clip(adjusted, 0.60, 0.95))

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
        'delivery_season': delivery_season
    }

def calculate_predicted_probability(order_history, clinic, item, current_date, delivery_days=3):
    orders = order_history[
        (order_history['OUTBOUND RECIPIENT'] == clinic) &
        (order_history['ITEMIZED LIST'] == item)
    ].copy()

    orders['DATE REDISTRIBUTED'] = pd.to_datetime(orders['DATE REDISTRIBUTED'])
    orders = orders.sort_values('DATE REDISTRIBUTED')

    if len(orders) < 3:
        return {
            'probability': 0.20,
            'reason': 'Insufficient order history (<3 orders)',
            'suggested_quantity': _safe_mean(orders['UNITS DISTRIBUTED']),
            'days_since_last': None,
            'avg_interval': None,
            'seasonal_factor': 1.0,
            'delivery_season': get_season(pd.to_datetime(current_date) + timedelta(days=delivery_days))
        }

    intervals = orders['DATE REDISTRIBUTED'].diff().dt.days.dropna()
    avg_interval = intervals.mean()
    min_interval = intervals.min()
    std_interval = intervals.std(ddof=0) if len(intervals) > 1 else 0.0

    last_order = orders['DATE REDISTRIBUTED'].max()
    current_dt = pd.to_datetime(current_date)
    days_since_last = (current_dt - last_order).days

    delivery_date = current_dt + timedelta(days=delivery_days)
    delivery_adjusted_days = days_since_last + delivery_days

    seasonal_multipliers = calculate_seasonal_preference(order_history, clinic, item)
    delivery_season = get_season(delivery_date)
    seasonal_factor = seasonal_multipliers[delivery_season]

    # Strict timing logic (personalized to this clinic×item)
    if delivery_adjusted_days < 0.8 * min_interval:
        base = 0.10
        reason = f"Too soon (last {days_since_last}d, min interval {min_interval:.0f}d)"
    elif delivery_adjusted_days >= avg_interval:
        overdue_factor = (delivery_adjusted_days - avg_interval) / max(avg_interval, 1e-9)
        base = 0.60 + 0.30 * min(overdue_factor, 1.0)  # up to 0.90
        reason = f"Due for reorder (avg {avg_interval:.0f}d, {days_since_last}d since last)"
    else:
        denom = max(avg_interval - min_interval, 1e-9)
        progress = (delivery_adjusted_days - min_interval) / denom
        base = 0.25 + 0.35 * np.clip(progress, 0.0, 1.0)
        reason = f"Approaching reorder window ({days_since_last}d since last, avg {avg_interval:.0f}d)"

    # Consistency bonus
    if avg_interval > 0 and (std_interval / avg_interval) < 0.3:
        base = min(base * 1.1, 0.95)
        reason += " (consistent pattern)"

    # Seasonality
    adjusted = base * seasonal_factor
    if seasonal_factor > 1.2:
        reason += f" (HIGH demand in {delivery_season})"
        adjusted = min(adjusted, 0.95)
    elif seasonal_factor < 0.8:
        reason += f" (LOW demand in {delivery_season})"
        adjusted = max(adjusted, 0.05)
    else:
        adjusted = float(np.clip(adjusted, 0.05, 0.95))

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
        'delivery_season': delivery_season
    }

# ----------------------------
# Offer generation
# ----------------------------
def generate_offers(order_history, inventory, needslist=None, current_date='2025-11-08',
                   delivery_days=3, min_probability_predicted=0.40, expiration_threshold_days=30):
    """
    Strategy:
      1) NEEDSLIST items: always include (rank by probability)
      2) PREDICTED items: include only if probability >= threshold
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

    needslist_offers, predicted_offers = [], []

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

        if is_expiring_soon(expiration, current_date, expiration_threshold_days):
            continue

        # Is it on needslist?
        on_needslist = False
        if needslist is not None and len(needslist) > 0:
            # Treat missing NEEDS_ITEM as True
            needs_flag = needslist['NEEDS_ITEM'] if 'NEEDS_ITEM' in needslist.columns else True
            needs_mask = (
                (needslist['Clinic Name'] == clinic) &
                (needslist['Exact Item Name (Inventory)'] == item) &
                needs_flag
            )
            on_needslist = bool(needs_mask.any())

        if on_needslist:
            pred = calculate_needslist_probability(order_history, clinic, item, current_date, delivery_days)
            needslist_offers.append({
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
                'on_needslist': True,
                'reason': pred['reason'],
            })
        else:
            pred = calculate_predicted_probability(order_history, clinic, item, current_date, delivery_days)
            if pred['probability'] >= min_probability_predicted:
                predicted_offers.append({
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
                    'on_needslist': False,
                    'reason': pred['reason'],
                })

    needslist_df = pd.DataFrame(needslist_offers).sort_values('probability', ascending=False) if needslist_offers else pd.DataFrame()
    predicted_df = pd.DataFrame(predicted_offers).sort_values('probability', ascending=False) if predicted_offers else pd.DataFrame()
    offers_df = pd.concat([needslist_df, predicted_df], ignore_index=True).sort_values(['clinic', 'probability'], ascending=[True, False])
    return offers_df

def generate_offers_for_clinic(order_history, inventory, clinic_name, needslist=None,
                               current_date='2025-11-08', delivery_days=3, expiration_threshold_days=30):
    all_offers = generate_offers(
        order_history, inventory, needslist, current_date, delivery_days,
        min_probability_predicted=0.0, expiration_threshold_days=expiration_threshold_days
    )
    return all_offers[all_offers['clinic'] == clinic_name].reset_index(drop=True)

# ----------------------------
# CLI / Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate clinic-item offers.")
    parser.add_argument("--order_history", default="./data/order_history.csv", help="Path to order_history.csv")
    parser.add_argument("--inventory", default="./data/inventory.csv", help="Path to inventory.csv (GENERIC NAME, SKU ID, EXPIRATION_DATE, CASE QTY)")
    parser.add_argument("--needslist", default="./data/needslist.csv", help="Optional path to needslist.csv (Clinic Name, Exact Item Name (Inventory), NEEDS_ITEM)")
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
