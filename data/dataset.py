# =============================================================================
# MICROFINANCE FRAUD DETECTION — REALISTIC DATASET GENERATOR (ENHANCED)
# =============================================================================
# Generates 5 relational tables with realistic fraud patterns.
# Fraud rate: ~3.8% (380 fraud loans out of ~10,200).
# Enhanced with:
#   - Feature noise (income, credit score, agent performance)
#   - Legitimate loans that mimic fraud (high delays, partial defaults)
#   - Cross-type fraud (identity + stacking, collusion + sudden default)
#   - Legitimate families sharing identifiers (graph noise)
# =============================================================================

import numpy as np
import pandas as pd
import hashlib
import random
import os
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Configuration & reproducibility
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUTPUT_DIR = "mfi_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Portfolio parameters
N_BORROWERS_LEGIT = 3000
N_AGENTS = 50
N_GROUPS = 300
N_LOANS_LEGIT = 9620          # baseline legitimate loans (will be slightly reduced later)

START_DATE = datetime(2021, 1, 1)
END_DATE   = datetime(2024, 6, 30)

TENURES = [12, 24, 36, 52]
INTEREST_LOW = 0.18
INTEREST_HIGH = 0.30
REGIONS = ["urban", "semi-urban", "rural"]

# Fraud counts (total 380)
N_GHOST      = 50
N_STACKING   = 20              # unique borrowers, each takes 3 loans → 60 loans
N_HIGH_DELAY = 80
N_SUDDEN     = 80
N_COLLUDE    = 80
N_ID_RINGS   = 10              # 10 rings × 3 borrowers → 30 loans

# Cross-type fraud counts (extra loans)
N_CROSS_ID_STACK = 5           # identity borrowers also do stacking (5 extra loans)
N_CROSS_COLL_DEF = 5           # collusion loans that also default

# Legitimate fraud-like loans (extra noise)
N_LEGIT_DEFAULT = 100          # legitimate borrowers who stop paying early
N_LEGIT_HIGH_DELAY = 150       # legitimate loans with high delays

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def rand_date(start=START_DATE, end=END_DATE):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def date_str(d):
    return d.strftime("%Y-%m-%d")

def make_phone():
    return f"9{random.randint(100_000_000, 999_999_999)}"

def short_hash(text, length=12):
    return hashlib.md5(text.encode()).hexdigest()[:length]

def clip_round(value, lo, hi, decimals=2):
    # works on both scalars and arrays
    return np.round(np.clip(value, lo, hi), decimals)

# Delay configurations
DELAY_CONFIGS = {
    "normal": dict(choices=[0,1,2,3,5,7,10,14], probs=[0.15,0.25,0.20,0.15,0.10,0.08,0.05,0.02]),
    "high": dict(choices=[3,5,7,10,14,21,28,35], probs=[0.05,0.10,0.20,0.25,0.20,0.12,0.05,0.03]),
    "mixed": dict(choices=[0,1,2,3,5,7,10,14,21], probs=[0.10,0.15,0.20,0.15,0.12,0.10,0.08,0.05,0.05]),
    "sudden": dict(choices=[0,1,2,3,5], probs=[0.20,0.35,0.30,0.10,0.05]),
}

def sample_delay(pattern):
    cfg = DELAY_CONFIGS.get(pattern, DELAY_CONFIGS["normal"])
    return int(np.random.choice(cfg["choices"], p=cfg["probs"]))

# -----------------------------------------------------------------------------
# Build base tables
# -----------------------------------------------------------------------------
print("Building agents...")
all_agent_ids = list(range(N_AGENTS))
corrupt_ids = random.sample(all_agent_ids, 6)   # 12% corrupt

agents_rows = []
for aid in all_agent_ids:
    is_corrupt = 1 if aid in corrupt_ids else 0
    perf = (clip_round(np.random.uniform(0.65, 0.90), 0, 1, 4) if is_corrupt
            else clip_round(np.random.uniform(0.50, 0.98), 0, 1, 4))
    agents_rows.append({
        "agent_id": aid,
        "agent_name": f"Agent_{aid:03d}",
        "branch": random.choice(["North","South","East","West","Central"]),
        "experience_years": random.randint(1, 15),
        "education_level": random.choice(["graduate","postgraduate","diploma"]),
        "performance_score": perf,
        "portfolio_size": 0,
        "is_corrupt": is_corrupt,
    })
agents_df = pd.DataFrame(agents_rows)

print("Building groups...")
groups_rows = []
for gid in range(N_GROUPS):
    region = np.random.choice(REGIONS, p=[0.33, 0.35, 0.32])
    groups_rows.append({
        "group_id": gid,
        "group_name": f"SHG_{gid:04d}",
        "region": region,
        "district": f"District_{random.randint(1, 30):02d}",
        "formation_date": date_str(rand_date(START_DATE, datetime(2022, 12, 31))),
        "group_size": random.randint(5, 20),
        "group_active": 1,
    })
groups_df = pd.DataFrame(groups_rows)

print("Building legitimate borrowers...")
borrowers_rows = []
for bid in range(N_BORROWERS_LEGIT):
    region = groups_df.loc[bid % N_GROUPS, "region"]
    phone = make_phone()
    addr = f"House_{random.randint(1, 9999)}_Street_{random.randint(1, 200)}"
    acc_dt = rand_date(START_DATE, datetime(2022, 6, 30))
    borrowers_rows.append({
        "borrower_id": bid,
        "full_name": f"Borrower_{bid:04d}",
        "age": random.randint(21, 62),
        "gender": np.random.choice(["F","M"], p=[0.74, 0.26]),
        "region": region,
        "district": f"District_{random.randint(1, 30):02d}",
        "group_id": bid % N_GROUPS,
        "agent_id": random.choice(all_agent_ids),
        "income_monthly": round(np.random.lognormal(mean=8.2, sigma=0.5), 2),
        "credit_score": clip_round(np.random.normal(620, 55), 300, 850, 1),
        "doc_score": round(np.random.beta(8, 2), 4),
        "past_defaults": int(np.random.choice([0,1,2,3], p=[0.88,0.08,0.03,0.01])),
        "repayment_behavior": np.random.choice(["good","average","risky"], p=[0.40, 0.38, 0.22]),
        "phone_hash": short_hash(phone),
        "address_hash": short_hash(addr),
        "account_creation_date": date_str(acc_dt),
        "kyc_verified": np.random.choice([1, 0], p=[0.95, 0.05]),
        "num_family_members": random.randint(1, 8),
        "existing_loans_count": 0,
    })
borrowers_df = pd.DataFrame(borrowers_rows)

# -----------------------------------------------------------------------------
# Add noise to features
# -----------------------------------------------------------------------------
print("Adding feature noise...")
borrowers_df["income_monthly"] = (
    borrowers_df["income_monthly"] * np.random.uniform(0.85, 1.15, size=len(borrowers_df))
).round(2)

borrowers_df["credit_score"] = clip_round(
    borrowers_df["credit_score"] + np.random.normal(0, 25, size=len(borrowers_df)),
    300, 850, 1
)

agents_df["performance_score"] = clip_round(
    agents_df["performance_score"] + np.random.normal(0, 0.03, size=len(agents_df)),
    0, 1, 4
)

# -----------------------------------------------------------------------------
# Add legitimate families sharing identifiers (graph noise)
# -----------------------------------------------------------------------------
print("Adding legitimate family sharing...")
n_family = 50
for _ in range(n_family):
    base_bid = random.choice(list(borrowers_df.index))
    phone_hash = borrowers_df.loc[base_bid, "phone_hash"]
    addr_hash = borrowers_df.loc[base_bid, "address_hash"]
    candidates = [b for b in borrowers_df.index if b != base_bid and
                  borrowers_df.loc[b, "phone_hash"] != phone_hash]
    if candidates:
        other_bid = random.choice(candidates)
        borrowers_df.loc[other_bid, "phone_hash"] = phone_hash
        borrowers_df.loc[other_bid, "address_hash"] = addr_hash

# -----------------------------------------------------------------------------
# Global containers for loans and transactions
# -----------------------------------------------------------------------------
loan_id_counter = 0
txn_id_counter = 0
all_loans = []
all_transactions = []

def add_loan(borrower_id, agent_id, group_id, loan_amount, tenure_weeks,
             interest_rate, disbursement_date, fraud_type, fraud_subtype,
             delay_pattern="normal", stop_at_week=None, partial_payment=True):
    global loan_id_counter, txn_id_counter
    lid = loan_id_counter
    loan_id_counter += 1
    is_fraud = 0 if fraud_type == "none" else 1
    emi = round(loan_amount * (1 + interest_rate) / tenure_weeks, 2)

    all_loans.append({
        "loan_id": lid,
        "borrower_id": borrower_id,
        "agent_id": agent_id,
        "group_id": group_id,
        "loan_amount": round(loan_amount, 2),
        "tenure_weeks": tenure_weeks,
        "interest_rate": round(interest_rate, 6),
        "emi_amount": emi,
        "loan_disbursement_date": date_str(disbursement_date),
        "loan_close_date": date_str(disbursement_date + timedelta(weeks=tenure_weeks)),
        "purpose": np.random.choice(["agriculture","business","education","home_improvement","medical"],
                                    p=[0.30, 0.35, 0.10, 0.15, 0.10]),
        "collateral": np.random.choice(["none","gold","property","vehicle"],
                                       p=[0.55, 0.20, 0.15, 0.10]),
        "fraud_type": fraud_type,
        "fraud_subtype": fraud_subtype,
        "is_fraud": is_fraud,
    })

    # Ghost loans: generate a few very small payments (to avoid zero transactions)
    if fraud_type == "ghost" and stop_at_week is None:
        for week in range(1, random.randint(2, 5)):
            txn_date = disbursement_date + timedelta(weeks=week)
            if txn_date > END_DATE:
                break
            delay = sample_delay("normal")
            amount = round(emi * np.random.uniform(0.05, 0.20), 2)
            all_transactions.append({
                "transaction_id": txn_id_counter,
                "loan_id": lid,
                "borrower_id": borrower_id,
                "week": week,
                "scheduled_date": date_str(txn_date),
                "actual_date": date_str(txn_date + timedelta(days=delay)),
                "emi_expected": emi,
                "amount_paid": amount,
                "delay_days": delay,
                "payment_mode": np.random.choice(["cash","bank_transfer","upi","cheque"],
                                                 p=[0.45, 0.25, 0.20, 0.10]),
                "payment_status": "paid",
            })
            txn_id_counter += 1
        return lid

    # Normal transaction generation
    for week in range(1, tenure_weeks + 1):
        txn_date = disbursement_date + timedelta(weeks=week)
        if txn_date > END_DATE:
            break
        if stop_at_week and week > stop_at_week:
            if partial_payment and week == stop_at_week + 1:
                amount = round(emi * np.random.uniform(0.2, 0.6), 2)
            else:
                break
        else:
            delay = sample_delay(delay_pattern)
            amount = round(emi * np.random.uniform(0.92, 1.08), 2)

        all_transactions.append({
            "transaction_id": txn_id_counter,
            "loan_id": lid,
            "borrower_id": borrower_id,
            "week": week,
            "scheduled_date": date_str(txn_date),
            "actual_date": date_str(txn_date + timedelta(days=delay)),
            "emi_expected": emi,
            "amount_paid": amount,
            "delay_days": delay,
            "payment_mode": np.random.choice(["cash","bank_transfer","upi","cheque"],
                                             p=[0.45, 0.25, 0.20, 0.10]),
            "payment_status": "paid",
        })
        txn_id_counter += 1
    return lid

# -----------------------------------------------------------------------------
# Generate legitimate loans (with fraud-like behavior added)
# -----------------------------------------------------------------------------
print("Generating legitimate loans (including fraud-like noise)...")
legit_pool = list(range(N_BORROWERS_LEGIT))
np.random.shuffle(legit_pool)

for i in range(N_LOANS_LEGIT):
    bid = legit_pool[i % N_BORROWERS_LEGIT]
    row = borrowers_df.loc[bid]
    gid = int(row["group_id"])
    aid = int(row["agent_id"])
    amt = clip_round(np.random.lognormal(mean=8.8, sigma=0.6), 1000, 80000, 2)
    tenure = random.choice(TENURES)
    rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
    disb = rand_date()

    # Add legitimate loans that mimic fraud
    delay_pattern = "normal"
    stop_at_week = None
    partial_payment = True

    if random.random() < 0.05:                     # 5% high delay
        delay_pattern = "high"
    elif random.random() < 0.10:                  # additional 10% mixed
        delay_pattern = "mixed"

    if random.random() < 0.02:                     # 2% legitimate default
        stop_at_week = int(tenure * np.random.uniform(0.4, 0.7))
        delay_pattern = "sudden"

    add_loan(bid, aid, gid, amt, tenure, rate, disb,
             "none", "none", delay_pattern, stop_at_week, partial_payment)

# -----------------------------------------------------------------------------
# Fraud Type 1: Ghost borrowers
# -----------------------------------------------------------------------------
print("Injecting ghost borrower fraud...")
ghost_agent_pool = corrupt_ids[:3]
ghost_bids = []
for i in range(N_GHOST):
    fake_bid = N_BORROWERS_LEGIT + i
    phone = make_phone()
    addr = f"FakeHouse_{random.randint(1, 999)}_FakeStreet_{random.randint(1, 50)}"
    acc_dt = rand_date(datetime(2023, 1, 1), datetime(2024, 3, 1))

    borrowers_df.loc[fake_bid] = {
        "borrower_id": fake_bid,
        "full_name": f"Ghost_{fake_bid:04d}",
        "age": random.randint(25, 45),
        "gender": "F",
        "region": "urban",
        "district": "District_01",
        "group_id": random.randint(0, N_GROUPS - 1),
        "agent_id": random.choice(ghost_agent_pool),
        "income_monthly": round(np.random.uniform(8000, 20000), 2),
        "credit_score": clip_round(np.random.uniform(700, 780), 300, 850, 1),
        "doc_score": round(np.random.beta(2, 8), 4),
        "past_defaults": 0,
        "repayment_behavior": "good",
        "phone_hash": short_hash(phone),
        "address_hash": short_hash(addr),
        "account_creation_date": date_str(acc_dt),
        "kyc_verified": 0,
        "num_family_members": random.randint(2, 5),
        "existing_loans_count": 0,
    }
    ghost_bids.append(fake_bid)

    aid = random.choice(ghost_agent_pool)
    gid = int(borrowers_df.loc[fake_bid, "group_id"])
    amt = round(np.random.uniform(15000, 70000), 2)
    tenure = random.choice([24, 36, 52])
    rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
    disb = rand_date(datetime(2022, 6, 1), datetime(2024, 3, 1))
    add_loan(fake_bid, aid, gid, amt, tenure, rate, disb,
             "ghost", "fabricated_identity", "ghost")

# -----------------------------------------------------------------------------
# Fraud Type 2: Loan stacking (2-3 concurrent loans)
# -----------------------------------------------------------------------------
print("Injecting loan stacking fraud...")
stacking_borrowers = random.sample(legit_pool[:800], N_STACKING)
for bid in stacking_borrowers:
    row = borrowers_df.loc[bid]
    gid = int(row["group_id"])
    anchor_date = rand_date(datetime(2022, 1, 1), datetime(2023, 6, 1))
    n_loans = random.randint(2, 3)
    for _ in range(n_loans):
        disb = anchor_date + timedelta(days=random.randint(0, 60))
        aid = random.choice(all_agent_ids)
        amt = round(np.random.uniform(5000, 35000), 2)
        tenure = random.choice([12, 24, 36])
        rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
        add_loan(bid, aid, gid, amt, tenure, rate, disb,
                 "stacking", "concurrent_loans", "mixed")

# -----------------------------------------------------------------------------
# Fraud Type 3: High delay
# -----------------------------------------------------------------------------
print("Injecting high-delay fraud...")
used_bids = set(stacking_borrowers)
hd_pool = [b for b in legit_pool if b not in used_bids][:N_HIGH_DELAY]
for bid in hd_pool:
    row = borrowers_df.loc[bid]
    gid = int(row["group_id"])
    aid = int(row["agent_id"])
    amt = round(np.random.uniform(3000, 40000), 2)
    tenure = random.choice(TENURES)
    rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
    disb = rand_date()
    add_loan(bid, aid, gid, amt, tenure, rate, disb,
             "high_delay", "chronic_late_payment", "high")
used_bids.update(set(hd_pool))

# -----------------------------------------------------------------------------
# Fraud Type 4: Sudden default
# -----------------------------------------------------------------------------
print("Injecting sudden-default fraud...")
sd_pool = [b for b in legit_pool if b not in used_bids][:N_SUDDEN]
for bid in sd_pool:
    row = borrowers_df.loc[bid]
    gid = int(row["group_id"])
    aid = int(row["agent_id"])
    amt = round(np.random.uniform(5000, 50000), 2)
    tenure = random.choice([24, 36, 52])
    rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
    disb = rand_date(START_DATE, datetime(2023, 6, 1))
    stop_w = int(tenure * np.random.uniform(0.4, 0.7))
    add_loan(bid, aid, gid, amt, tenure, rate, disb,
             "sudden_default", "early_cessation", "sudden",
             stop_at_week=stop_w, partial_payment=True)
used_bids.update(set(sd_pool))

# -----------------------------------------------------------------------------
# Fraud Type 5: Agent collusion
# -----------------------------------------------------------------------------
print("Injecting agent-collusion fraud...")
collude_pool = [b for b in legit_pool if b not in used_bids][:N_COLLUDE]
collude_agent_pool = corrupt_ids[3:]   # corrupt agents used for collusion
for bid in collude_pool:
    row = borrowers_df.loc[bid]
    gid = int(row["group_id"])
    # 80% of collusion loans go through corrupt agents, 20% through normal agents (noise)
    if random.random() < 0.8:
        aid = random.choice(collude_agent_pool)
    else:
        aid = random.choice(all_agent_ids)
    income = float(row["income_monthly"])
    amt = clip_round(np.random.uniform(income * 8, income * 18), 5000, 100000, 2)
    tenure = random.choice(TENURES)
    rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
    disb = rand_date()
    add_loan(bid, aid, gid, amt, tenure, rate, disb,
             "agent_collusion", "kickback_scheme", "normal")
used_bids.update(set(collude_pool))

# -----------------------------------------------------------------------------
# Fraud Type 6: Identity fraud (rings of 3, 80% shared identifiers)
# -----------------------------------------------------------------------------
print("Injecting identity fraud...")
ring_phone_hashes = [short_hash(make_phone()) for _ in range(N_ID_RINGS)]
ring_addr_hashes  = [short_hash(f"RingAddr_{i}_Lane1") for i in range(N_ID_RINGS)]
identity_bids = []
for ring_i in range(N_ID_RINGS):
    for member_j in range(3):
        fake_bid = N_BORROWERS_LEGIT + N_GHOST + ring_i * 3 + member_j
        acc_dt = rand_date(datetime(2023, 1, 1), datetime(2024, 6, 1))
        use_shared_phone = np.random.rand() < 0.8
        use_shared_addr  = np.random.rand() < 0.8
        phone = ring_phone_hashes[ring_i] if use_shared_phone else short_hash(make_phone())
        addr  = ring_addr_hashes[ring_i]  if use_shared_addr  else short_hash(f"OtherAddr_{fake_bid}")
        borrowers_df.loc[fake_bid] = {
            "borrower_id": fake_bid,
            "full_name": f"SynID_{fake_bid:04d}",
            "age": random.randint(22, 50),
            "gender": np.random.choice(["F","M"]),
            "region": "urban",
            "district": "District_05",
            "group_id": random.randint(0, N_GROUPS - 1),
            "agent_id": random.choice(all_agent_ids),
            "income_monthly": round(np.random.uniform(5000, 15000), 2),
            "credit_score": clip_round(np.random.uniform(680, 750), 300, 850, 1),
            "doc_score": round(np.random.beta(3, 7), 4),
            "past_defaults": 0,
            "repayment_behavior": "good",
            "phone_hash": phone,
            "address_hash": addr,
            "account_creation_date": date_str(acc_dt),
            "kyc_verified": 0,
            "num_family_members": random.randint(2, 4),
            "existing_loans_count": 0,
        }
        identity_bids.append(fake_bid)

        gid = int(borrowers_df.loc[fake_bid, "group_id"])
        aid = int(borrowers_df.loc[fake_bid, "agent_id"])
        amt = round(np.random.uniform(8000, 45000), 2)
        tenure = random.choice([12, 24, 36])
        rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
        disb = rand_date(datetime(2023, 1, 1), datetime(2024, 6, 1))
        add_loan(fake_bid, aid, gid, amt, tenure, rate, disb,
                 "identity_fraud", "synthetic_identity", "normal")

# -----------------------------------------------------------------------------
# Cross-type fraud: identity + stacking
# -----------------------------------------------------------------------------
print("Adding cross-type fraud (identity + stacking)...")
cross_bids = random.sample(identity_bids, min(N_CROSS_ID_STACK, len(identity_bids)))
for bid in cross_bids:
    row = borrowers_df.loc[bid]
    gid = int(row["group_id"])
    anchor_date = rand_date(datetime(2022, 1, 1), datetime(2023, 6, 1))
    n_loans = random.randint(2, 3)
    for _ in range(n_loans):
        disb = anchor_date + timedelta(days=random.randint(0, 60))
        aid = random.choice(all_agent_ids)
        amt = round(np.random.uniform(5000, 35000), 2)
        tenure = random.choice([12, 24, 36])
        rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
        add_loan(bid, aid, gid, amt, tenure, rate, disb,
                 "cross_identity_stacking", "mixed_ring", "mixed")

# -----------------------------------------------------------------------------
# Cross-type fraud: collusion + sudden default
# -----------------------------------------------------------------------------
print("Adding cross-type fraud (collusion + sudden default)...")
# Pick some borrowers that already have collusion loans
collusion_borrowers = [bid for bid in used_bids if any(loan["fraud_type"] == "agent_collusion" for loan in all_loans if loan["borrower_id"] == bid)]
collusion_borrowers = list(set(collusion_borrowers))[:N_CROSS_COLL_DEF]
for bid in collusion_borrowers:
    row = borrowers_df.loc[bid]
    gid = int(row["group_id"])
    # Ensure a corrupt agent (or mix)
    aid = random.choice(collude_agent_pool) if random.random() < 0.8 else random.choice(all_agent_ids)
    income = float(row["income_monthly"])
    amt = clip_round(np.random.uniform(income * 8, income * 18), 5000, 100000, 2)
    tenure = random.choice([24, 36, 52])
    rate = round(np.random.uniform(INTEREST_LOW, INTEREST_HIGH), 6)
    disb = rand_date(START_DATE, datetime(2023, 6, 1))
    stop_w = int(tenure * np.random.uniform(0.4, 0.7))
    add_loan(bid, aid, gid, amt, tenure, rate, disb,
             "cross_collusion_sudden", "kickback_default", "sudden",
             stop_at_week=stop_w, partial_payment=True)

# -----------------------------------------------------------------------------
# Compile final tables
# -----------------------------------------------------------------------------
print("Compiling final tables...")
loans_df = pd.DataFrame(all_loans)
txns_df = pd.DataFrame(all_transactions)

# Back-fill existing_loans_count on borrowers
loan_ct = loans_df.groupby("borrower_id").size().rename("existing_loans_count")
for bid, ct in loan_ct.items():
    if bid in borrowers_df.index:
        borrowers_df.loc[bid, "existing_loans_count"] = int(ct)

# Back-fill portfolio_size on agents
port = loans_df.groupby("agent_id").size().rename("portfolio_size")
for aid, ps in port.items():
    agents_df.loc[agents_df["agent_id"] == aid, "portfolio_size"] = int(ps)

borrowers_df = borrowers_df.reset_index(drop=True)
borrowers_df["borrower_id"] = borrowers_df["borrower_id"].astype(int)
loans_df["loan_id"] = loans_df["loan_id"].astype(int)
txns_df["loan_id"] = txns_df["loan_id"].astype(int)
txns_df["borrower_id"] = txns_df["borrower_id"].astype(int)

# -----------------------------------------------------------------------------
# Save to CSV
# -----------------------------------------------------------------------------
print("Saving CSV files...")
loans_df.to_csv(os.path.join(OUTPUT_DIR, "loans.csv"), index=False)
txns_df.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)
borrowers_df.to_csv(os.path.join(OUTPUT_DIR, "borrowers.csv"), index=False)
agents_df.to_csv(os.path.join(OUTPUT_DIR, "agents.csv"), index=False)
groups_df.to_csv(os.path.join(OUTPUT_DIR, "groups.csv"), index=False)

print("Dataset generation completed.")
print(f"Total loans: {len(loans_df)} | Fraud loans: {loans_df['is_fraud'].sum()} ({loans_df['is_fraud'].mean():.2%})")
print(f"Files saved to {OUTPUT_DIR}/")