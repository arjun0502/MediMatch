"""
MediMatch Supabase-backed offer generation model (Schema v2: clinic_name-based)

This module contains the unified probability model and a Supabase-backed
pipeline used to predict which clinic×item combinations should be
offered from inventory. It mirrors the CSV-backed variant but loads inputs
from Supabase tables and writes offers back to a Supabase `offers` table.

Schema highlights (new):
- order_history.clinic_name (text)
- needs_list.clinic_name (text)
- offers.clinic_name (text)
- inventory keyed by sku_id (text)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from supabase import create_client, Client
import argparse

# ----------------------------
# Configuration
# ----------------------------
def get_supabase_client() -> Client:
    """Get Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError(
            "Missing Supabase credentials. Please set:\n"
            "  SUPABASE_URL\n"
            "  SUPABASE_ANON_KEY"
        )
    return create_client(url, key)

# ----------------------------
# Clinic resolvers (ID -> name, partial name -> exact)
# ----------------------------
def get_clinic_name_by_id(sb: Client, clinic_id: str) -> str:
    """Resolve clinic_name from clinics.id (UUID)."""
    resp = sb.table("clinics").select("id,name").eq("id", clinic_id).limit(1).execute()
    if not resp.data:
        raise ValueError(f"No clinic found with id '{clinic_id}'")
    return resp.data[0]["name"]

def get_clinic_name_by_partial(sb: Client, clinic_name_query: str) -> str:
    """
    Resolve an exact clinic_name from a partial/case-insensitive query.
    If multiple match, prints options and returns the first match.
    """
    resp = (
        sb.table("clinics")
        .select("id,name")
        .ilike("name", f"%{clinic_name_query}%")
        .execute()
    )
    if not resp.data:
        raise ValueError(f"No clinic found matching '{clinic_name_query}'")
    if len(resp.data) > 1:
        print("Warning: multiple clinic matches found. Using first match:")
        for c in resp.data:
            print(f"  - {c['name']} ({c['id']})")
    return resp.data[0]["name"]

def resolve_clinic_name(sb: Client, clinic_id: Optional[str], clinic_name: Optional[str]) -> str:
    """Resolve to a concrete clinic_name string given either id or name query."""
    if clinic_name:
        # Normalize to exact stored name if possible (helpful against casing/whitespace)
        return get_clinic_name_by_partial(sb, clinic_name)
    if clinic_id:
        return get_clinic_name_by_id(sb, clinic_id)
    raise ValueError("You must provide either --clinic-id or --clinic-name")

# ----------------------------
# Season helpers
# ----------------------------
def get_season(date) -> str:
    if isinstance(date, str):
        date = pd.to_datetime(date)
    m = date.month
    if m in [12, 1, 2]:
        return 'Winter'
    elif m in [3, 4, 5]:
        return 'Spring'
    elif m in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def calculate_seasonal_preference(order_history_df: pd.DataFrame) -> Dict[str, float]:
    if len(order_history_df) < 4:
        return {'Winter': 1.0, 'Spring': 1.0, 'Summer': 1.0, 'Fall': 1.0}
    orders = order_history_df.copy()
    orders['season'] = pd.to_datetime(orders['accepted_at']).apply(get_season)
    seasonal_counts = orders.groupby('season').size()
    total = len(orders)
    multipliers = {}
    for s in ['Winter', 'Spring', 'Summer', 'Fall']:
        actual_pct = (seasonal_counts.get(s, 0) / total) if total > 0 else 0.25
        multipliers[s] = actual_pct / 0.25
    return multipliers

# ----------------------------
# Data Loading from Supabase (clinic_name-based)
# ----------------------------
def load_clinic_data(sb: Client, clinic_name: str) -> Dict:
    """
    Load data for a clinic using clinic_name.
    Returns dict with 'clinic_name', 'order_history', 'needs_list', 'inventory'
    """
    print(f"Loading data for clinic '{clinic_name}'...")

    # Optional: verify clinic exists (for better error messages)
    clinic_check = sb.table("clinics").select("id,name").ilike("name", clinic_name).execute()
    # If not found by exact ilike, try equality to avoid false positives:
    if not clinic_check.data:
        # Try exact (case-sensitive) match as a fallback
        clinic_check = sb.table("clinics").select("id,name").eq("name", clinic_name).execute()
    if not clinic_check.data:
        print("Note: clinic name not present in `clinics` table; proceeding using raw name.")

    # Order history for this clinic
    oh_resp = (
        sb.table("order_history")
        .select("*")
        .eq("clinic_name", clinic_name)
        .execute()
    )
    order_history = pd.DataFrame(oh_resp.data) if oh_resp.data else pd.DataFrame()

    # Needs list for this clinic
    nl_resp = (
        sb.table("needs_list")
        .select("*")
        .eq("clinic_name", clinic_name)
        .execute()
    )
    needs_list = pd.DataFrame(nl_resp.data) if nl_resp.data else pd.DataFrame()

    # Full inventory (needed for merging item names, expirations, qty)
    inv_resp = sb.table("inventory").select("*").execute()
    inventory = pd.DataFrame(inv_resp.data) if inv_resp.data else pd.DataFrame()

    print(f"  Order history: {len(order_history)} records")
    print(f"  Needs list:   {len(needs_list)} items")
    print(f"  Inventory:    {len(inventory)} items")

    return {
        "clinic_name": clinic_name,
        "order_history": order_history,
        "needs_list": needs_list,
        "inventory": inventory,
    }

# ----------------------------
# Probability Calculation
# ----------------------------
def _safe_mean(series, default=50):
    s = pd.to_numeric(series, errors='coerce').dropna()
    return int(round(s.mean())) if len(s) else default

def calculate_probability(
    order_history_df: pd.DataFrame,
    sku_id: str,
    current_date: str,
    delivery_days: int = 3,
    on_needslist: bool = False
) -> Dict:
    orders = order_history_df.copy()

    if len(orders) == 0 or len(orders) < 3:
        if on_needslist:
            base_prob = 0.75
            reason = "On needslist (+25% boost) - Limited order history (<3 orders)"
        else:
            base_prob = 0.20
            reason = "Limited order history (<3 orders)"
        return {
            'probability': base_prob,
            'reason': reason,
            'suggested_quantity': _safe_mean(orders['ordered_qty']) if len(orders) > 0 else 50,
            'days_since_last': None,
            'avg_interval': None,
            'seasonal_factor': 1.0,
            'delivery_season': get_season(pd.to_datetime(current_date) + timedelta(days=delivery_days)),
            'on_needslist': on_needslist
        }

    orders['accepted_at'] = pd.to_datetime(orders['accepted_at'])
    orders = orders.sort_values('accepted_at')

    intervals = orders['accepted_at'].diff().dt.days.dropna()
    avg_interval = intervals.mean()
    min_interval = intervals.min()
    std_interval = intervals.std(ddof=0) if len(intervals) > 1 else 0.0

    last_order = orders['accepted_at'].max()
    current_dt = pd.to_datetime(current_date)

    # Ensure both current_dt and last_order are timezone-naive
    current_dt = current_dt.tz_localize(None) if current_dt.tzinfo else current_dt
    last_order = last_order.tz_localize(None) if last_order.tzinfo else last_order

    days_since_last = (current_dt - last_order).days

    delivery_date = current_dt + timedelta(days=delivery_days)
    delivery_adjusted_days = days_since_last + delivery_days

    if delivery_adjusted_days < 0.8 * min_interval:
        base = 0.10
        reason = f"Too soon (last {days_since_last}d, min interval {min_interval:.0f}d)"
    elif delivery_adjusted_days >= avg_interval:
        overdue_factor = (delivery_adjusted_days - avg_interval) / max(avg_interval, 1e-9)
        base = 0.50 + 0.25 * min(overdue_factor, 1.0)
        reason = f"Due for reorder (avg {avg_interval:.0f}d, {days_since_last}d since last)"
    else:
        denom = max(avg_interval - min_interval, 1e-9)
        progress = (delivery_adjusted_days - min_interval) / denom
        base = 0.25 + 0.25 * np.clip(progress, 0.0, 1.0)
        reason = f"Approaching reorder window ({days_since_last}d since last, avg {avg_interval:.0f}d)"

    if avg_interval > 0 and (std_interval / avg_interval) < 0.3:
        base = min(base * 1.1, 0.65 if not on_needslist else 0.75)
        reason += " (consistent pattern)"

    seasonal_multipliers = calculate_seasonal_preference(orders)
    delivery_season = get_season(delivery_date)
    seasonal_factor = seasonal_multipliers[delivery_season]
    adjusted = base * seasonal_factor

    if seasonal_factor > 1.2:
        reason += f" (HIGH demand in {delivery_season})"
        adjusted = min(adjusted, 0.65 if not on_needslist else 0.75)
    elif seasonal_factor < 0.8:
        reason += f" (LOW demand in {delivery_season})"
        adjusted = max(adjusted, 0.05)
    else:
        adjusted = float(np.clip(adjusted, 0.05, 0.65 if not on_needslist else 0.75))

    if on_needslist:
        adjusted = min(adjusted + 0.25, 1.0)
        adjusted = max(adjusted, 0.70)
        reason = f"On needslist (+25% boost) - {reason}"

    seasonal_orders = orders[orders['accepted_at'].apply(get_season) == delivery_season]
    if len(seasonal_orders) >= 2:
        suggested_qty = _safe_mean(seasonal_orders['ordered_qty'])
    else:
        suggested_qty = _safe_mean(orders['ordered_qty'])

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
# Offer Generation (clinic_name-based)
# ----------------------------
def generate_offers_for_clinic(
    sb: Client,
    clinic_name: str,
    current_date: Optional[str] = None,
    delivery_days: int = 3,
    min_probability_predicted: float = 0.40,
    expiration_threshold_days: int = 30
) -> pd.DataFrame:
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
    current_dt = pd.to_datetime(current_date)

    data = load_clinic_data(sb, clinic_name)
    order_history = data['order_history']
    needs_list = data['needs_list']
    inventory = data['inventory']

    if inventory.empty:
        print("Warning: No inventory available")
        return pd.DataFrame()

    # expiration_date is NOT NULL in schema, but be defensive:
    inventory['expiration_date'] = pd.to_datetime(inventory['expiration_date'], errors='coerce')
    inventory = inventory.dropna(subset=['expiration_date'])

    # Filter out expiring items
    inventory = inventory[
        (inventory['expiration_date'] - current_dt).dt.days > expiration_threshold_days
    ]
    if inventory.empty:
        print("Warning: No inventory with sufficient shelf life")
        return pd.DataFrame()

    print(f"\nGenerating offers for '{clinic_name}'...")
    print(f"  Available inventory: {len(inventory)} items")

    # Get unique SKUs from order history and needs list
    historical_skus = set(order_history['sku_id'].unique()) if not order_history.empty else set()
    needs_skus = set(needs_list['sku_id'].unique()) if not needs_list.empty else set()
    inventory_skus = set(inventory['sku_id'].unique())

    candidate_skus = (historical_skus | needs_skus) & inventory_skus
    print(f"  Candidate items: {len(candidate_skus)}")

    offers: List[Dict] = []

    # Build quick lookup dicts for speed
    inv_by_sku = {row['sku_id']: row for _, row in inventory.iterrows()}

    for sku_id in candidate_skus:
        on_needslist = sku_id in needs_skus

        item_inv = inv_by_sku[sku_id]
        available_qty = int(item_inv['qty'])
        if available_qty <= 0:
            continue

        item_orders = order_history[order_history['sku_id'] == sku_id].copy()

        pred = calculate_probability(
            item_orders,
            sku_id,
            current_date,
            delivery_days,
            on_needslist
        )

        include_offer = on_needslist or (pred['probability'] >= min_probability_predicted)
        if not include_offer:
            continue

        if on_needslist:
            needs_item = needs_list[needs_list['sku_id'] == sku_id].iloc[0]
            suggested_qty = int(needs_item['qty'])
        else:
            suggested_qty = pred['suggested_quantity']

        offers.append({
            'clinic_name': clinic_name,
            'sku_id': sku_id,
            'item_name': item_inv['name'],
            'probability': float(pred['probability']),
            'suggested_quantity': min(int(suggested_qty), available_qty),
            'available_in_stock': available_qty,
            'expiration_date': pd.to_datetime(item_inv['expiration_date']).strftime('%Y-%m-%d'),
            'days_since_last_order': pred['days_since_last'],
            'avg_reorder_interval': pred['avg_interval'],
            'seasonal_factor': pred['seasonal_factor'],
            'delivery_season': pred['delivery_season'],
            'on_needslist': on_needslist,
            'reason': pred['reason'],
        })

    offers_df = pd.DataFrame(offers)
    if not offers_df.empty:
        offers_df = offers_df.sort_values('probability', ascending=False)
        print(f"\n✓ Generated {len(offers_df)} offers")
        print(f"  Needslist: {offers_df['on_needslist'].sum()}")
        print(f"  Predicted: {(~offers_df['on_needslist']).sum()}")
    else:
        print("\n✗ No offers generated")

    return offers_df

# ----------------------------
# Save to Supabase (offers.clinic_name)
# ----------------------------
def save_offers_to_supabase(sb: Client, offers_df: pd.DataFrame) -> int:
    if offers_df.empty:
        print("No offers to save")
        return 0

    print(f"\nSaving {len(offers_df)} offers to Supabase...")

    records = []
    now_iso = datetime.now().isoformat()
    for _, row in offers_df.iterrows():
        records.append({
            'clinic_name': row['clinic_name'],
            'sku_id': row['sku_id'],
            'name': row['item_name'],
            'offered_qty': int(row['suggested_quantity']),
            'probability': float(row['probability']),
            'reason': row['reason'],
            'created_at': now_iso
        })

    batch_size = 100
    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            _ = sb.table('offers').insert(batch).execute()
            total_inserted += len(batch)
            print(f"  Inserted batch {i//batch_size + 1}: {len(batch)} offers")
        except Exception as e:
            print(f"  Error inserting batch {i//batch_size + 1}: {e}")
            raise

    print(f"✓ Successfully saved {total_inserted} offers to database")
    return total_inserted

# ----------------------------
# Main entrypoint
# ----------------------------
def generate_and_save_offers(
    clinic_name: str,
    sb: Optional[Client] = None,
    current_date: Optional[str] = None,
    delivery_days: int = 3,
    min_probability_predicted: float = 0.40,
    expiration_threshold_days: int = 30,
    verbose: bool = True
) -> int:
    try:
        if sb is None:
            sb = get_supabase_client()

        offers_df = generate_offers_for_clinic(
            sb=sb,
            clinic_name=clinic_name,
            current_date=current_date,
            delivery_days=delivery_days,
            min_probability_predicted=min_probability_predicted,
            expiration_threshold_days=expiration_threshold_days
        )
        if offers_df.empty:
            return 0

        if verbose:
            print("\nTop 10 Offers:")
            print("-" * 80)
            display_cols = ['item_name', 'probability', 'suggested_quantity', 'on_needslist', 'reason']
            print(offers_df[display_cols].head(10).to_string(index=False))
            print("-" * 80)

        count = save_offers_to_supabase(sb, offers_df)
        return count

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate probability-based offers for a clinic and save to Supabase (clinic_name schema)"
    )
    parser.add_argument("--clinic-id", help="UUID of the clinic (will be resolved to clinic_name)")
    parser.add_argument("--clinic-name", help="Name (or partial name) of the clinic (case-insensitive)")
    parser.add_argument("--current-date", default=None, help="Current date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--delivery-days", type=int, default=3, help="Estimated delivery time in days (default: 3)")
    parser.add_argument("--predicted-threshold", type=float, default=0.40,
                        help="Minimum probability for predicted items (default: 0.40)")
    parser.add_argument("--expiration-threshold-days", type=int, default=30,
                        help="Don't offer items expiring within this many days (default: 30)")

    args = parser.parse_args()

    print("=" * 80)
    print("Medical Supply Offer Generator - Supabase Edition (clinic_name schema)")
    print("=" * 80)

    sb = get_supabase_client()
    clinic_name = resolve_clinic_name(sb, args.clinic_id, args.clinic_name)
    print(f"Resolved clinic -> '{clinic_name}'")

    count = generate_and_save_offers(
        clinic_name=clinic_name,
        sb=sb,
        current_date=args.current_date,
        delivery_days=args.delivery_days,
        min_probability_predicted=args.predicted_threshold,
        expiration_threshold_days=args.expiration_threshold_days
    )

    print("\n" + "=" * 80)
    print(f"✓ DONE: Created {count} offers for clinic '{clinic_name}'")
    print("=" * 80)

if __name__ == "__main__":
    main()
