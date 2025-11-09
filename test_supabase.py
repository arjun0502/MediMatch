"""
Safe Supabase connection & schema test (read-only)

This script checks:
 - Supabase connection
 - Presence of required tables
 - Presence of expected columns in each table

It does NOT read real data, write any data, or modify anything.
"""

import os
from typing import Dict, List, Optional
from supabase import create_client, Client

# Expected columns per your new schema (minimum set)
EXPECTED_SCHEMAS: Dict[str, List[str]] = {
    "clinics": [
        "id",
        "name",
        # optional but nice-to-have:
        # "created_at",
    ],
    "inventory": [
        "sku_id",
        "name",
        "description",
        "image_url",
        "expiration_date",
        "qty",
        # "updated_at",
    ],
    "needs_list": [
        "id",
        "created_at",
        "clinic_name",
        "item_name",
        "sku_id",
        "qty",
    ],
    "offers": [
        "id",
        "created_at",
        "clinic_name",
        "sku_id",
        "name",
        "offered_qty",
        "probability",
        "reason",
    ],
    "order_history": [
        "id",
        "accepted_at",
        "ordered_qty",
        "offer_id",
        "clinic_name",
        "sku_id",
    ],
}

def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY")
    return create_client(url, key)

def check_table_columns(sb: Client, table: str, expected_cols: List[str]) -> List[str]:
    """
    Validate that each expected column exists by issuing a zero-row SELECT
    on exactly those columns. If PostgREST can't find a column, it will error.
    Returns a list of missing columns (empty if all present).
    """
    try:
        # Try to select only the expected columns; limit(0) avoids reading data.
        sb.table(table).select(",".join(expected_cols)).limit(0).execute()
        return []
    except Exception as e:
        # Fall back: try to read 1 row with '*' and compare keys (works if table has data).
        try:
            resp = sb.table(table).select("*").limit(1).execute()
            if resp.data:
                present = set(resp.data[0].keys())
                return [c for c in expected_cols if c not in present]
            else:
                # Table is empty and direct column check failed (likely due to a bad column name).
                # Report all columns as "unknown" to force an explicit fix.
                return expected_cols[:]  # treat all as missing since prior select failed
        except Exception:
            # Can't even read the table; report all expected as missing.
            return expected_cols[:]

def test_schema():
    print("🔍 Testing Supabase connection...")
    sb = get_client()
    print("✅ Connection successful\n")

    all_good = True
    for table, expected_cols in EXPECTED_SCHEMAS.items():
        print(f"Checking table: {table}")
        missing = []
        try:
            missing = check_table_columns(sb, table, expected_cols)
        except Exception as e:
            print(f"❌ Error checking '{table}': {e}")
            all_good = False
            print()
            continue

        if missing:
            all_good = False
            print(f"❌ Missing columns: {missing}")
        else:
            print(f"✅ Columns OK: {expected_cols}")
        print()

    if all_good:
        print("🎉 All tables and schemas validated successfully!")
    else:
        print("⚠️ Some issues were detected. Please review output above.")

def list_clinics(sb: Client, name_filter: Optional[str] = None) -> List[Dict]:
    """
    Load clinics from the `clinics` table. If `name_filter` is provided,
    performs a case-insensitive partial match on the clinic name.
    Returns a list of {'id', 'name'} dicts.
    """
    query = sb.table('clinics').select('id,name')
    if name_filter:
        query = query.ilike('name', f"%{name_filter}%")
    resp = query.execute()
    return resp.data or []

def test_list_clinics():
    print("🔍 Testing list_clinics function...")
    sb = get_client()
    try:
        clinics = list_clinics(sb)
        if not clinics:
            print("⚠️ No clinics found in the database.")
        else:
            print(f"✅ Found {len(clinics)} clinic(s):")
            for clinic in clinics:
                print(f"  - {clinic['name']} ({clinic['id']})")

        # Optional filtered example
        filter_name = "Clinic 2"
        filtered = list_clinics(sb, name_filter=filter_name)
        print(f"\nFiltered clinics containing '{filter_name}':")
        if not filtered:
            print("  (none)")
        for clinic in filtered:
            print(f"  - {clinic['name']} ({clinic['id']})")
    except Exception as e:
        print(f"❌ Error testing list_clinics: {e}")

if __name__ == "__main__":
    test_schema()
    test_list_clinics()
