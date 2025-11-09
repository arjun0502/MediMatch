"""
MediMatch Supabase-backed offer generation model

This module contains the unified probability model and a Supabase-backed
pipeline used to predict which clinic×item combinations should be
offered from inventory. It mirrors the CSV-backed variant
(`model_with_test_data.py`) but loads inputs from Supabase tables and
writes offers back to a Supabase `offers` table.

High-level flow (input -> output):

- Inputs (Supabase tables):
    - `clinics` (clinic metadata)
    - `order_history` (historical orders; expected fields include
        `clinic_id`, `sku_id`, `accepted_at`, `ordered_qty`)
    - `inventory` (items, SKU metadata, expiration dates, quantities)
    - `needs_list` (explicit clinic requests)

- Process:
    1. Connect to Supabase using `SUPABASE_URL` and `SUPABASE_ANON_KEY` from
         environment variables.
    2. Load clinic info, order history, needs list, and inventory for the
         target clinic.
    3. For each clinic×item present in history and inventory, run
         `calculate_probability(...)` which:
             - computes reorder intervals from history
             - classifies timing (too soon / approaching / overdue)
             - applies consistency and seasonal adjustments
             - boosts items on the needslist
             - returns a probability and suggested quantity
    4. Skip items expiring soon and those below the probability threshold
         (unless on the needslist).
    5. Save resulting offers to the Supabase `offers` table.

- Output: Supabase `offers` table rows with fields similar to the CSV
    output: `clinic_id`, `sku_id`, `offered_qty`, `probability`, `reason`,
    `created_at`, etc.

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
        writes outputs to a Supabase instance for production automation.

Usage (PowerShell):

    # set supabase credentials
    $env:SUPABASE_URL = 'https://<your-instance>.supabase.co'
    $env:SUPABASE_ANON_KEY = '<anon-key>'

    # run for a clinic by id
    python model_with_supabase.py --clinic-id <clinic-uuid>

"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import supabase
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

def get_clinic_id_by_name(supabase: Client, clinic_name: str) -> str:
    """
    Look up clinic ID from the clinics table using a case-insensitive, partial name match.

    If multiple clinics match, the function prints them and returns the first match.
    If you prefer to force the user to be specific, replace the multi-match case with a ValueError.
    """
    response = supabase.table("clinics") \
        .select("id, name") \
        .ilike("name", f"%{clinic_name}%") \
        .execute()

    if not response.data:
        raise ValueError(f"No clinic found matching name '{clinic_name}'")

    if len(response.data) > 1:
        print("Warning: multiple clinic matches found. Using first match:")
        for c in response.data:
            print(f"  - {c['name']} ({c['id']})")

        # If you want to force disambiguation:
        # raise ValueError(
        #     f"Multiple clinics match '{clinic_name}'. Please provide a more specific name or use --clinic-id."
        # )

    return response.data[0]["id"]

# ----------------------------
# Season helpers
# ----------------------------
def get_season(date) -> str:
    """Get season (Winter/Spring/Summer/Fall) from date."""
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
    """
    Calculate seasonal multipliers based on order history.

    Args:
        order_history_df: DataFrame with 'accepted_at' column

    Returns:
        Dict mapping season -> multiplier (e.g., {'Winter': 1.2, ...})
    """
    if len(order_history_df) < 4:
        return {'Winter': 1.0, 'Spring': 1.0, 'Summer': 1.0, 'Fall': 1.0}

    orders = order_history_df.copy()
    orders['season'] = pd.to_datetime(orders['accepted_at']).apply(get_season)
    seasonal_counts = orders.groupby('season').size()
    total = len(orders)

    multipliers = {}
    for s in ['Winter', 'Spring', 'Summer', 'Fall']:
        actual_pct = (seasonal_counts.get(s, 0) / total) if total > 0 else 0.25
        multipliers[s] = actual_pct / 0.25  # deviation from uniform 25%
    return multipliers

# ----------------------------
# Data Loading from Supabase
# ----------------------------
def load_clinic_data(supabase: Client, clinic_id: str) -> Dict:
    """
    Load all relevant data for a specific clinic from Supabase.

    Args:
        supabase: Supabase client
        clinic_id: UUID of the clinic

    Returns:
        Dict with 'clinic', 'order_history', 'needs_list', 'inventory'
    """
    print(f"Loading data for clinic {clinic_id}...")

    # Get clinic info
    clinic_response = supabase.table('clinics').select('*').eq('id', clinic_id).execute()
    if not clinic_response.data:
        raise ValueError(f"Clinic {clinic_id} not found")
    clinic = clinic_response.data[0]

    # Get order history for this clinic
    order_history_response = supabase.table('order_history') \
        .select('*, inventory(name)') \
        .eq('clinic_id', clinic_id) \
        .execute()
    order_history = pd.DataFrame(order_history_response.data)

    # Get needs list for this clinic
    needs_list_response = supabase.table('needs_list') \
        .select('*') \
        .eq('clinic_id', clinic_id) \
        .execute()
    needs_list = pd.DataFrame(needs_list_response.data)

    # Get all inventory
    inventory_response = supabase.table('inventory').select('*').execute()
    inventory = pd.DataFrame(inventory_response.data)

    print(f"  Clinic: {clinic.get('name', '<unknown>')}")
    print(f"  Order history: {len(order_history)} records")
    print(f"  Needs list: {len(needs_list)} items")
    print(f"  Inventory: {len(inventory)} items")

    return {
        'clinic': clinic,
        'order_history': order_history,
        'needs_list': needs_list,
        'inventory': inventory
    }

# ----------------------------
# Probability Calculation
# ----------------------------
def _safe_mean(series, default=50):
    """Calculate mean, handling empty series."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    return int(round(s.mean())) if len(s) else default

def calculate_probability(
    order_history_df: pd.DataFrame,
    sku_id: str,
    current_date: str,
    delivery_days: int = 3,
    on_needslist: bool = False
) -> Dict:
    """
    Calculate probability for a single clinic-item pair using unified method.

    Args:
        order_history_df: Historical orders for this clinic-item pair
        sku_id: SKU ID of the item
        current_date: Current date (YYYY-MM-DD)
        delivery_days: Estimated delivery time in days
        on_needslist: Whether this item is on the clinic's needs list

    Returns:
        Dict with 'probability', 'reason', 'suggested_quantity', etc.
    """
    orders = order_history_df.copy()

    if len(orders) == 0 or len(orders) < 3:
        # INSUFFICIENT DATA: Use conservative defaults
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

    # HISTORICAL PATTERN ANALYSIS
    orders['accepted_at'] = pd.to_datetime(orders['accepted_at'])
    orders = orders.sort_values('accepted_at')

    intervals = orders['accepted_at'].diff().dt.days.dropna()
    avg_interval = intervals.mean()
    min_interval = intervals.min()
    std_interval = intervals.std(ddof=0) if len(intervals) > 1 else 0.0

    last_order = orders['accepted_at'].max()
    current_dt = pd.to_datetime(current_date)
    days_since_last = (current_dt - last_order).days

    delivery_date = current_dt + timedelta(days=delivery_days)
    delivery_adjusted_days = days_since_last + delivery_days

    # TIMING ASSESSMENT
    if delivery_adjusted_days < 0.8 * min_interval:
        # TOO SOON
        base = 0.10
        reason = f"Too soon (last {days_since_last}d, min interval {min_interval:.0f}d)"
    elif delivery_adjusted_days >= avg_interval:
        # OVERDUE
        overdue_factor = (delivery_adjusted_days - avg_interval) / max(avg_interval, 1e-9)
        base = 0.50 + 0.25 * min(overdue_factor, 1.0)
        reason = f"Due for reorder (avg {avg_interval:.0f}d, {days_since_last}d since last)"
    else:
        # APPROACHING
        denom = max(avg_interval - min_interval, 1e-9)
        progress = (delivery_adjusted_days - min_interval) / denom
        base = 0.25 + 0.25 * np.clip(progress, 0.0, 1.0)
        reason = f"Approaching reorder window ({days_since_last}d since last, avg {avg_interval:.0f}d)"

    # CONSISTENCY BONUS
    if avg_interval > 0 and (std_interval / avg_interval) < 0.3:
        base = min(base * 1.1, 0.65 if not on_needslist else 0.75)
        reason += " (consistent pattern)"

    # SEASONAL ADJUSTMENT
    seasonal_multipliers = calculate_seasonal_preference(orders)
    delivery_season = get_season(delivery_date)
    seasonal_factor = seasonal_multipliers[delivery_season]

    adjusted = base * seasonal_factor

    # Apply seasonal caps/floors
    if seasonal_factor > 1.2:
        reason += f" (HIGH demand in {delivery_season})"
        adjusted = min(adjusted, 0.65 if not on_needslist else 0.75)
    elif seasonal_factor < 0.8:
        reason += f" (LOW demand in {delivery_season})"
        adjusted = max(adjusted, 0.05)
    else:
        adjusted = float(np.clip(adjusted, 0.05, 0.65 if not on_needslist else 0.75))

    # NEEDSLIST BOOST
    if on_needslist:
        adjusted = min(adjusted + 0.25, 1.0)
        adjusted = max(adjusted, 0.70)
        reason = f"On needslist (+25% boost) - {reason}"

    # QUANTITY SUGGESTION
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
# Offer Generation
# ----------------------------
def generate_offers_for_clinic(
    supabase: Client,
    clinic_id: str,
    current_date: Optional[str] = None,
    delivery_days: int = 3,
    min_probability_predicted: float = 0.40,
    expiration_threshold_days: int = 30
) -> pd.DataFrame:
    """
    Generate offers for a specific clinic.

    Args:
        supabase: Supabase client
        clinic_id: UUID of the clinic
        current_date: Current date (defaults to today)
        delivery_days: Estimated delivery time
        min_probability_predicted: Minimum probability to offer predicted items
        expiration_threshold_days: Don't offer items expiring within this many days

    Returns:
        DataFrame with offer recommendations
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")

    current_dt = pd.to_datetime(current_date)

    # Load data
    data = load_clinic_data(supabase, clinic_id)
    clinic = data['clinic']
    order_history = data['order_history']
    needs_list = data['needs_list']
    inventory = data['inventory']

    if inventory.empty:
        print("Warning: No inventory available")
        return pd.DataFrame()

    # Convert dates
    if 'expiration_date' in inventory.columns:
        inventory['expiration_date'] = pd.to_datetime(inventory['expiration_date'], errors='coerce')
        inventory = inventory.dropna(subset=['expiration_date'])
    else:
        # If your schema doesn’t always include expiration_date, skip expiration filtering
        inventory['expiration_date'] = pd.Timestamp.max

    # Filter out expiring items
    inventory = inventory[
        (inventory['expiration_date'] - current_dt).dt.days > expiration_threshold_days
    ]

    if inventory.empty:
        print("Warning: No inventory with sufficient shelf life")
        return pd.DataFrame()

    print(f"\nGenerating offers for {clinic.get('name', '<unknown>')}...")
    print(f"  Available inventory: {len(inventory)} items")

    # Get unique SKUs from order history and needs list
    historical_skus = set(order_history['sku_id'].unique()) if not order_history.empty else set()
    needs_skus = set(needs_list['sku_id'].unique()) if not needs_list.empty else set()
    inventory_skus = set(inventory['sku_id'].unique())

    # Combine: offer items that are either in history or on needs list, and are in inventory
    candidate_skus = (historical_skus | needs_skus) & inventory_skus

    print(f"  Candidate items: {len(candidate_skus)}")

    offers = []

    for sku_id in candidate_skus:
        # Check if on needs list
        on_needslist = sku_id in needs_skus

        # Get inventory info
        item_inv = inventory[inventory['sku_id'] == sku_id].iloc[0]
        available_qty = int(item_inv['qty'])

        if available_qty <= 0:
            continue

        # Get order history for this item
        item_orders = order_history[order_history['sku_id'] == sku_id].copy()

        # Calculate probability
        pred = calculate_probability(
            item_orders,
            sku_id,
            current_date,
            delivery_days,
            on_needslist
        )

        # Decision: Include this offer?
        include_offer = on_needslist or (pred['probability'] >= min_probability_predicted)

        if include_offer:
            # Get quantity from needs list if available, otherwise use suggested
            if on_needslist:
                needs_item = needs_list[needs_list['sku_id'] == sku_id].iloc[0]
                suggested_qty = int(needs_item['qty'])
            else:
                suggested_qty = pred['suggested_quantity']

            offers.append({
                'clinic_id': clinic_id,
                'clinic_name': clinic.get('name', ''),
                'sku_id': sku_id,
                'item_name': item_inv['name'],
                'probability': pred['probability'],
                'suggested_quantity': min(int(suggested_qty), available_qty),
                'available_in_stock': available_qty,
                'expiration_date': pd.to_datetime(item_inv['expiration_date']).strftime('%Y-%m-%d') \
                                    if not pd.isna(item_inv['expiration_date']) else None,
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
# Save to Supabase
# ----------------------------
def save_offers_to_supabase(supabase: Client, offers_df: pd.DataFrame) -> int:
    """
    Save offers to Supabase offers table.

    Args:
        supabase: Supabase client
        offers_df: DataFrame with offers

    Returns:
        Number of offers saved
    """
    if offers_df.empty:
        print("No offers to save")
        return 0

    print(f"\nSaving {len(offers_df)} offers to Supabase...")

    # Prepare records for insertion
    records = []
    now_iso = datetime.now().isoformat()
    for _, row in offers_df.iterrows():
        records.append({
            'clinic_id': row['clinic_id'],
            'sku_id': row['sku_id'],
            'offered_qty': int(row['suggested_quantity']),
            'probability': float(row['probability']),
            'reason': row['reason'],
            'created_at': now_iso
        })

    # Insert in batches to avoid payload size limits
    batch_size = 100
    total_inserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            _ = supabase.table('offers').insert(batch).execute()
            total_inserted += len(batch)
            print(f"  Inserted batch {i//batch_size + 1}: {len(batch)} offers")
        except Exception as e:
            print(f"  Error inserting batch {i//batch_size + 1}: {e}")
            raise

    print(f"✓ Successfully saved {total_inserted} offers to database")
    return total_inserted

# ----------------------------
# Main Function
# ----------------------------
def generate_and_save_offers(
    clinic_id: str,
    supabase: Optional[Client] = None,
    current_date: Optional[str] = None,
    delivery_days: int = 3,
    min_probability_predicted: float = 0.40,
    expiration_threshold_days: int = 30,
    verbose: bool = True
) -> int:
    """
    Generate offers for a clinic and save them to Supabase.

    This is the main entry point - it handles everything:
    1. Connects to Supabase
    2. Loads clinic data
    3. Generates offers with probabilities
    4. Writes offers to database

    Args:
        clinic_id: UUID of the clinic
        supabase: Optional Supabase client (creates one if not provided)
        current_date: Optional date (defaults to today)
        delivery_days: Estimated delivery time
        min_probability_predicted: Minimum probability for predicted items
        expiration_threshold_days: Don't offer items expiring soon
        verbose: Print progress messages

    Returns:
        Number of offers created
    """
    try:
        # Get Supabase client
        if supabase is None:
            supabase = get_supabase_client()

        # Generate offers
        offers_df = generate_offers_for_clinic(
            supabase=supabase,
            clinic_id=clinic_id,
            current_date=current_date,
            delivery_days=delivery_days,
            min_probability_predicted=min_probability_predicted,
            expiration_threshold_days=expiration_threshold_days
        )

        if offers_df.empty:
            return 0

        # Display top offers
        if verbose:
            print("\nTop 10 Offers:")
            print("-" * 80)
            display_cols = ['item_name', 'probability', 'suggested_quantity', 'on_needslist', 'reason']
            print(offers_df[display_cols].head(10).to_string(index=False))
            print("-" * 80)

        # Save to database
        count = save_offers_to_supabase(supabase, offers_df)

        return count

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

# ----------------------------
# CLI
# ----------------------------
def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate probability-based offers for a clinic and save to Supabase"
    )
    parser.add_argument(
        "--clinic-id",
        help="UUID of the clinic"
    )
    parser.add_argument(
        "--clinic-name",
        help="Name (or partial name) of the clinic (case-insensitive)"
    )
    parser.add_argument(
        "--current-date",
        default=None,
        help="Current date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--delivery-days",
        type=int,
        default=3,
        help="Estimated delivery time in days (default: 3)"
    )
    parser.add_argument(
        "--predicted-threshold",
        type=float,
        default=0.40,
        help="Minimum probability for predicted items (default: 0.40)"
    )
    parser.add_argument(
        "--expiration-threshold-days",
        type=int,
        default=30,
        help="Don't offer items expiring within this many days (default: 30)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Medical Supply Offer Generator - Supabase Edition")
    print("=" * 80)

    # Create client first so we can resolve name -> id if needed
    supabase = get_supabase_client()

    # Resolve clinic ID
    if args.clinic_id:
        clinic_id = args.clinic_id
    elif args.clinic_name:
        clinic_id = get_clinic_id_by_name(supabase, args.clinic_name)
        print(f"Resolved clinic name '{args.clinic_name}' -> {clinic_id}")
    else:
        raise ValueError("You must provide either --clinic-id or --clinic-name")

    count = generate_and_save_offers(
        clinic_id=clinic_id,
        supabase=supabase,
        current_date=args.current_date,
        delivery_days=args.delivery_days,
        min_probability_predicted=args.predicted_threshold,
        expiration_threshold_days=args.expiration_threshold_days
    )

    print("\n" + "=" * 80)
    print(f"✓ DONE: Created {count} offers for clinic {clinic_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()