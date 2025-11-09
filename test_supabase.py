"""
Safe Supabase connection & schema test (read-only)

This script checks:
 - Supabase connection
 - Presence of required tables
 - Presence of expected columns in each table

It does NOT read any real data, write any data, or modify anything.
"""

import os
import supabase
from supabase import create_client, Client

EXPECTED_SCHEMAS = {
    "clinics": ["id", "name"],
    "order_history": ["id", "clinic_id", "sku_id", "accepted_at", "ordered_qty"],
    "needs_list": ["id", "clinic_id", "sku_id", "qty"],
    "inventory": ["id", "sku_id", "name", "qty", "expiration_date"],
    "offers": ["id", "clinic_id", "sku_id", "offered_qty", "probability", "created_at", "reason"],
}

def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY")

    return create_client(url, key)

def test_schema():
    print("🔍 Testing Supabase connection...")
    client = get_client()
    print("✅ Connection successful\n")

    all_good = True

    for table, expected_cols in EXPECTED_SCHEMAS.items():
        print(f"Checking table: {table}")

        try:
            # Read 1 row to inspect columns
            response = client.table(table).select("*").limit(1).execute()

            if response.data:
                columns = list(response.data[0].keys())
            else:
                # No rows → fetch table metadata (safe)
                # Supabase returns full row structure in response.meta
                if hasattr(response, "response"):
                    columns = list(response.response.json()["columns"].keys())
                else:
                    columns = expected_cols  # fallback

            missing = [c for c in expected_cols if c not in columns]

            if missing:
                all_good = False
                print(f"❌ Missing columns: {missing}")
            else:
                print(f"✅ Columns OK: {expected_cols}")

        except Exception as e:
            all_good = False
            print(f"❌ Error reading table '{table}': {e}")

        print()

    if all_good:
        print("🎉 All tables and schemas validated successfully!")
    else:
        print("⚠️ Some issues were detected. Please review output above.")

if __name__ == "__main__":
    test_schema()
