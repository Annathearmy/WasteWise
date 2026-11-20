"""
WasteWise / WasteLoop MVP
-------------------------
User workflow: run_wasteloop()
Admin workflow: review_pending_feedback(), export_approved_feedback(),
                show_feedback_status(), show_category_progress(), check_database()

Importing this module does NOT mount Google Drive, load models, modify the database,
or start an interactive workflow. Models are loaded lazily when required.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

DEFAULT_PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
PROJECT_DIR = os.environ.get("WASTEWISE_PROJECT_PATH", DEFAULT_PROJECT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "Models")
DB_PATH = os.path.join(PROJECT_DIR, "Database", "wasteloop_mvp.db")
APPROVED_DATA_PATH = os.path.join(
    PROJECT_DIR, "Dataset", "approved_feedback_dataset.csv"
)
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "sentence_transformer")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "sentence_classifier.pkl")

CONFIDENCE_THRESHOLD = 0.55
APPROVED_COLUMNS = [
    "Phrase", "Class", "feedback_id", "original_text",
    "clarified_text", "approved_at"
]

embedding_model = None
embedding_classifier = None


def initialize_wasteloop(
    embedding_model_path: str = EMBEDDING_MODEL_PATH,
    classifier_path: str = CLASSIFIER_PATH,
    db_path: str = DB_PATH,
) -> tuple[Any, Any]:
    """Load model artifacts and verify that the configured database exists."""
    global embedding_model, embedding_classifier, DB_PATH

    DB_PATH = db_path
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"WasteLoop database not found: {DB_PATH}")
    if not os.path.isfile(classifier_path):
        raise FileNotFoundError(f"Classifier file not found: {classifier_path}")

    from src.shared_embedder import get_shared_embedder
    embedding_model = get_shared_embedder()
    embedding_classifier = joblib.load(classifier_path)

    with sqlite3.connect(DB_PATH) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise sqlite3.DatabaseError(f"Database integrity check failed: {integrity}")

    print("WasteLoop models loaded (shared ONNX runtime).")
    print(f"Classifier categories: {len(embedding_classifier.classes_)}")
    print(f"Database: {DB_PATH}")
    return embedding_model, embedding_classifier


def _ensure_models() -> None:
    if embedding_model is None or embedding_classifier is None:
        initialize_wasteloop()


def _display(data: Any) -> None:
    """Use notebook display when available; otherwise print."""
    try:
        from IPython.display import display
        display(data)
    except Exception:
        print(data)


def _feedback_columns(conn: sqlite3.Connection) -> set[str]:
    return {
        row[1] for row in conn.execute("PRAGMA table_info(user_feedback)")
    }


def _ensure_feedback_schema(conn: sqlite3.Connection) -> None:
    """Add review-related columns if the existing database is older."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT NOT NULL,
            clarified_text TEXT,
            model_prediction TEXT,
            model_score REAL,
            selected_category TEXT,
            suggested_category TEXT,
            feedback_type TEXT,
            review_status TEXT DEFAULT 'PENDING',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    existing = _feedback_columns(conn)
    additions = {
        "initial_prediction": "TEXT",
        "initial_score": "REAL",
        "reviewed_category": "TEXT",
        "review_notes": "TEXT",
        "reviewed_at": "TEXT",
    }
    for name, sql_type in additions.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE user_feedback ADD COLUMN {name} {sql_type}")


# ---------------------------------------------------------------------------
# Classification and user feedback
# ---------------------------------------------------------------------------

def _predict(text: str) -> tuple[Any, pd.DataFrame]:
    _ensure_models()
    vector = embedding_model.encode([text], normalize_embeddings=True)
    probabilities = embedding_classifier.predict_proba(vector)[0]
    top_indices = probabilities.argsort()[::-1][:10]
    top = pd.DataFrame({
        "category": embedding_classifier.classes_[top_indices],
        "score": probabilities[top_indices],
    })
    return probabilities, top


def save_user_feedback(
    original_text: str,
    clarified_text: Optional[str],
    model_prediction: Optional[str],
    model_score: Optional[float],
    selected_category: Optional[str] = None,
    suggested_category: Optional[str] = None,
    feedback_type: Optional[str] = None,
) -> int:
    """Save feedback; write both prediction-column pairs for schema compatibility."""
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_feedback_schema(conn)
        cursor = conn.execute("""
            INSERT INTO user_feedback (
                original_text, clarified_text, model_prediction, model_score,
                selected_category, suggested_category, feedback_type,
                initial_prediction, initial_score, review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
        """, (
            original_text, clarified_text, model_prediction, model_score,
            selected_category, suggested_category, feedback_type,
            model_prediction, model_score,
        ))
        return int(cursor.lastrowid)


def classify_with_clarification(text: str) -> dict:
    """Classify text; ask clarification and user category choice when confidence is low."""
    if not text or not text.strip():
        return {"predicted_class": None, "classification_status": "UNRESOLVED"}

    probabilities, top_results = _predict(text)
    best_index = int(probabilities.argmax())
    model_prediction = str(embedding_classifier.classes_[best_index])
    score = float(probabilities[best_index])

    print("Initial prediction:", model_prediction)
    print("Model score:", round(score, 4))

    if score >= CONFIDENCE_THRESHOLD:
        return {
            "predicted_class": model_prediction,
            "model_prediction": model_prediction,
            "model_score": score,
            "classification_status": "ACCEPTED",
            "classification_source": "MODEL",
            "updated_description": text,
        }

    clarification = input(
        "Please describe the item's material or packaging type: "
    ).strip()
    if not clarification:
        return {
            "predicted_class": None,
            "model_prediction": model_prediction,
            "model_score": score,
            "classification_status": "UNRESOLVED",
            "updated_description": text,
        }

    updated_text = f"{text} {clarification}".strip()
    probabilities, top_results = _predict(updated_text)
    best_index = int(probabilities.argmax())
    model_prediction = str(embedding_classifier.classes_[best_index])
    score = float(probabilities[best_index])

    if score >= CONFIDENCE_THRESHOLD:
        return {
            "predicted_class": model_prediction,
            "model_prediction": model_prediction,
            "model_score": score,
            "classification_status": "CLARIFIED",
            "classification_source": "MODEL",
            "updated_description": updated_text,
        }

    print("\nClassification remains uncertain. Select a category or suggest a new one:")
    _display(top_results.reset_index(drop=True))

    while True:
        choice = input("Enter category number (1-10), N for new category, or 0 to skip: ").strip().lower()
        if choice in {"n", "0"}:
            break
        if choice.isdigit() and 1 <= int(choice) <= len(top_results):
            break
        print("Invalid choice. Try again.")

    if choice == "n":
        suggestion = input("Enter the name of the new waste category: ").strip()
        if not suggestion:
            print("No suggestion entered; submission unresolved.")
            return {
                "predicted_class": None,
                "model_prediction": model_prediction,
                "model_score": score,
                "classification_status": "UNRESOLVED",
                "updated_description": updated_text,
            }
        feedback_id = save_user_feedback(
            text, updated_text, model_prediction, score,
            suggested_category=suggestion,
            feedback_type="NEW_CATEGORY_SUGGESTION",
        )
        print("New category suggestion saved for review. Feedback ID:", feedback_id)
        return {
            "predicted_class": None,
            "model_prediction": model_prediction,
            "model_score": score,
            "classification_status": "NEW_CATEGORY_PENDING",
            "updated_description": updated_text,
            "feedback_id": feedback_id,
        }

    if choice == "0":
        feedback_id = save_user_feedback(
            text, updated_text, model_prediction, score,
            feedback_type="UNRESOLVED",
        )
        print("Unresolved example saved. Feedback ID:", feedback_id)
        return {
            "predicted_class": None,
            "model_prediction": model_prediction,
            "model_score": score,
            "classification_status": "UNRESOLVED",
            "updated_description": updated_text,
            "feedback_id": feedback_id,
        }

    selected_category = str(top_results.iloc[int(choice) - 1]["category"])
    feedback_type = (
        "USER_CONFIRMATION" if selected_category == model_prediction
        else "USER_CORRECTION"
    )
    feedback_id = save_user_feedback(
        text, updated_text, model_prediction, score,
        selected_category=selected_category,
        feedback_type=feedback_type,
    )
    print("User selection saved. Feedback ID:", feedback_id)
    return {
        "predicted_class": selected_category,
        "model_prediction": model_prediction,
        "model_score": score,
        "classification_status": "USER_SELECTED",
        "classification_source": "USER",
        "updated_description": updated_text,
        "feedback_id": feedback_id,
    }


# ---------------------------------------------------------------------------
# Extraction and confirmation
# ---------------------------------------------------------------------------

CONDITION_KEYWORDS = [
    "empty", "clean", "dirty", "unwashed", "washed", "rinsed",
    "dry", "dried", "wet", "broken", "damaged", "used",
]
INTENT_KEYWORDS = {
    "recycle": ["recycle", "recycling"],
    "sell": ["sell", "selling"],
    "donate": ["donate", "donation", "give away"],
    "reuse": ["reuse", "upcycle", "repurpose"],
    "dispose": ["dispose", "discard", "throw away"],
}


def extract_information(text: str) -> dict:
    """Extract quantity, condition and recovery intention from the text."""
    working = text.lower()

    weight_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|g|grams?)\b",
        working,
    )
    count_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s+"
        r"(?:(?:[a-z][a-z0-9-]*)\s+){0,3}"
        r"(bottles?|cans?|jars?|packets?|pouches?|bags?|cups?|boxes?|"
        r"containers?|items?|pieces?|pcs)\b",
        working,
    )

    if weight_match:
        quantity = float(weight_match.group(1))
        quantity_unit = "kg" if weight_match.group(2).startswith("k") else "g"
    elif count_match:
        quantity = float(count_match.group(1))
        quantity_unit = "items"
    else:
        quantity, quantity_unit = None, None

    conditions = [
        keyword for keyword in CONDITION_KEYWORDS
        if re.search(r"\b" + re.escape(keyword) + r"\b", working)
    ]
    intents = [
        intent for intent, keywords in INTENT_KEYWORDS.items()
        if any(keyword in working for keyword in keywords)
    ]

    return {
        "quantity": quantity,
        "quantity_unit": quantity_unit,
        "condition": conditions,
        "condition_notes": text,
        "user_intent": intents,
    }


def _confirm_quantity(quantity: Optional[float], unit: Optional[str]) -> tuple[float, str]:
    while True:
        if quantity is not None:
            prompt = (
                f"I detected {quantity} {unit}. Press Enter to confirm "
                "or enter a correction (e.g. 3 items or 2 kg): "
            )
            answer = input(prompt).strip()
            if not answer:
                return float(quantity), str(unit)
        else:
            answer = input("How much waste do you have? (e.g. 5 items or 2 kg): ").strip()

        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|g|grams?|items?|pieces?|pcs|"
            r"bottles?|cans?|jars?|packets?|pouches?|bags?|cups?|boxes?|containers?)?",
            answer.lower(),
        )
        if match and float(match.group(1)) > 0:
            raw_unit = match.group(2) or "items"
            normalized_unit = (
                "kg" if raw_unit.startswith("k")
                else "g" if raw_unit in {"g", "gram", "grams"}
                else "items"
            )
            return float(match.group(1)), normalized_unit
        print("Please enter a valid positive quantity.")


def _collect_condition(existing: list[str], condition_notes: str) -> tuple[list[str], str]:
    if existing:
        return existing, condition_notes
    answer = input(
        "What is the condition of the waste? "
        "(empty and clean / unwashed / broken / other): "
    ).strip()
    conditions = [
        keyword for keyword in CONDITION_KEYWORDS
        if re.search(r"\b" + re.escape(keyword) + r"\b", answer.lower())
    ]
    if not conditions:
        conditions = ["unknown"]
    return conditions, (condition_notes + " " + answer).strip()


def _collect_intent(existing: list[str]) -> list[str]:
    if existing:
        return existing
    answer = input(
        "Would you like to recycle, sell, donate, reuse or consider any option? "
    ).strip().lower()
    if "any" in answer or "all" in answer:
        return ["any"]
    selected = [
        intent for intent, keywords in INTENT_KEYWORDS.items()
        if any(keyword in answer for keyword in keywords)
    ]
    return selected or ["any"]


# ---------------------------------------------------------------------------
# Recovery partner matching
# ---------------------------------------------------------------------------

def find_recovery_options(
    user_info: dict,
    include_demo_records: bool = True,
) -> pd.DataFrame:
    """
    Filter recovery options by exact waste class, city, PIN, quantity, condition,
    user intent and organisation status. This function does not rank candidates.
    """
    required = ["predicted_class", "city", "pincode"]
    missing = [key for key in required if not user_info.get(key)]
    if missing:
        raise ValueError("Missing required matching fields: " + ", ".join(missing))

    with sqlite3.connect(DB_PATH) as conn:
        candidates = pd.read_sql_query("""
            SELECT
                o.organization_name,
                o.partner_type,
                o.contact,
                o.verification_status,
                r.option_id,
                r.waste_class,
                r.recovery_method,
                r.minimum_quantity,
                r.quantity_unit,
                r.accepted_condition,
                r.preparation_instructions,
                r.city,
                r.pincode,
                r.pickup_available,
                r.dropoff_available
            FROM organizations o
            JOIN recovery_options r
                ON o.organization_id = r.organization_id
            WHERE r.waste_class = ?
              AND LOWER(r.city) = LOWER(?)
              AND r.pincode = ?
        """, conn, params=(
            user_info["predicted_class"],
            str(user_info["city"]).strip(),
            str(user_info["pincode"]).strip(),
        ))

    if candidates.empty:
        print("No organisations found for this waste category and location.")
        return candidates

    def quantity_eligible(row: pd.Series) -> bool:
        minimum = row["minimum_quantity"]
        if pd.isna(minimum):
            return True
        quantity = user_info.get("quantity")
        user_unit = user_info.get("quantity_unit")
        if quantity is None or user_unit is None:
            return False
        row_unit = "" if pd.isna(row["quantity_unit"]) else str(row["quantity_unit"]).lower().strip()
        user_unit = str(user_unit).lower().replace(" (inferred)", "").strip()
        return float(quantity) >= float(minimum) and user_unit == row_unit

    def condition_eligible(row: pd.Series) -> bool:
        required_condition = row["accepted_condition"]
        if pd.isna(required_condition) or not str(required_condition).strip():
            return True
        observed = {str(value).lower().strip() for value in user_info.get("condition", [])}
        notes = str(user_info.get("condition_notes", "")).lower()
        requirements = re.split(r"\s+and\s+|,", str(required_condition).lower())
        for requirement in requirements:
            requirement = requirement.strip()
            if not requirement:
                continue
            if requirement not in observed and not re.search(
                r"\b" + re.escape(requirement) + r"\b", notes
            ):
                return False
        return True

    preferences = user_info.get("user_intent") or []
    if preferences and "any" not in preferences and "unknown" not in preferences:
        candidates["intent_eligible"] = candidates["recovery_method"].isin(preferences)
    else:
        candidates["intent_eligible"] = True

    candidates["quantity_eligible"] = candidates.apply(quantity_eligible, axis=1)
    candidates["condition_eligible"] = candidates.apply(condition_eligible, axis=1)
    candidates["status_eligible"] = candidates["verification_status"].isin(["VERIFIED"])
    if include_demo_records:
        candidates["status_eligible"] |= candidates["verification_status"].isin(["DEMO_ONLY"])

    eligible = candidates[
        candidates["quantity_eligible"]
        & candidates["condition_eligible"]
        & candidates["intent_eligible"]
        & candidates["status_eligible"]
    ].copy()

    print("Potential organisations found:", len(candidates))
    print("Eligible recommendations:", len(eligible))
    return eligible


def display_recommendations(recommendations: pd.DataFrame) -> None:
    if recommendations is None or recommendations.empty:
        print("No suitable recovery organisations are currently listed.")
        return
    print("\nRecovery recommendations:")
    for _, partner in recommendations.iterrows():
        print("Organisation:", partner.get("organization_name"))
        print("Accepted waste:", partner.get("waste_class"))
        print("Recovery method:", partner.get("recovery_method"))
        print("Preparation:", partner.get("preparation_instructions"))
        print("Pickup available:", bool(partner.get("pickup_available")))
        print("Drop-off available:", bool(partner.get("dropoff_available")))
        print("Verification status:", partner.get("verification_status"))
        print()


# ---------------------------------------------------------------------------
# Main user-facing workflow
# ---------------------------------------------------------------------------

def run_wasteloop() -> tuple[Optional[dict], pd.DataFrame]:
    """Run the sequential interactive WasteLoop user workflow."""
    _ensure_models()
    text = input("Describe the waste you have: ").strip()
    if not text:
        print("Please enter a waste description.")
        return None, pd.DataFrame()

    city = input("Enter your city: ").strip()
    if not city:
        print("City is required.")
        return None, pd.DataFrame()

    while True:
        pincode = input("Enter your 6-digit PIN code: ").strip()
        if re.fullmatch(r"\d{6}", pincode):
            break
        print("Please enter a valid 6-digit PIN code.")

    classification = classify_with_clarification(text)
    if not classification.get("predicted_class"):
        print("Waste category could not be identified reliably.")
        print("Please provide more information before continuing.")
        return None, pd.DataFrame()

    predicted_class = classification["predicted_class"]
    model_score = classification.get("model_score")
    working_text = classification.get("updated_description", text)
    extracted = extract_information(working_text)

    quantity, unit = _confirm_quantity(
        extracted.get("quantity"), extracted.get("quantity_unit")
    )
    conditions, condition_notes = _collect_condition(
        extracted.get("condition", []), extracted.get("condition_notes", working_text)
    )
    intents = _collect_intent(extracted.get("user_intent", []))

    user_info = {
        "description": text,
        "updated_description": working_text,
        "predicted_class": predicted_class,
        "model_score": model_score,
        "quantity": quantity,
        "quantity_unit": unit,
        "condition": conditions,
        "condition_notes": condition_notes,
        "user_intent": intents,
        "city": city,
        "pincode": pincode,
        "classification_status": classification.get("classification_status"),
        "classification_source": classification.get("classification_source", "MODEL"),
    }

    print("\nConfirmed waste information:")
    _display(pd.DataFrame([user_info]))

    recommendations = find_recovery_options(user_info)
    display_recommendations(recommendations)
    return user_info, recommendations


# ---------------------------------------------------------------------------
# Admin-only functions. None are called by run_wasteloop() or at import time.
# ---------------------------------------------------------------------------

def review_pending_feedback() -> None:
    """Interactively approve, correct, reject, skip or quit pending submissions."""
    _ensure_models()
    Path(APPROVED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(APPROVED_DATA_PATH):
        approved_df = pd.read_csv(APPROVED_DATA_PATH)
    else:
        approved_df = pd.DataFrame(columns=APPROVED_COLUMNS)
    approved_df.to_csv(APPROVED_DATA_PATH, index=False)

    for col in APPROVED_COLUMNS:
        if col not in approved_df.columns:
            approved_df[col] = pd.NA
    approved_df = approved_df[APPROVED_COLUMNS]

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_feedback_schema(conn)
        pending = pd.read_sql_query("""
            SELECT * FROM user_feedback
            WHERE review_status = 'PENDING'
            ORDER BY feedback_id
        """, conn)

    if pending.empty:
        print("No pending submissions.")
        return

    valid_categories = {str(x).upper() for x in embedding_classifier.classes_}

    for _, record in pending.iterrows():
        fid = int(record["feedback_id"])
        print(f"\nFeedback ID: {fid}")
        print("Original description:", record.get("original_text"))
        print("Clarified description:", record.get("clarified_text"))
        print("Initial prediction:", record.get("initial_prediction"))
        print("Model score:", record.get("initial_score"))
        print("User selection:", record.get("selected_category"))
        print("Suggested category:", record.get("suggested_category"))

        decision = input("Approve (A), Correct (C), Reject (R), Skip (S), Quit (Q): ").strip().upper()
        if decision in {"", "Q"}:
            print("Review process stopped.")
            break
        if decision == "S":
            continue
        if decision == "R":
            notes = input("Reason for rejection: ").strip()
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE user_feedback
                    SET review_status='REJECTED', review_notes=?,
                        reviewed_at=CURRENT_TIMESTAMP
                    WHERE feedback_id=? AND review_status='PENDING'
                """, (notes, fid))
            print("Submission rejected.")
            continue
        if decision == "A":
            default = record.get("selected_category")
            if pd.isna(default) or not str(default).strip():
                default = record.get("suggested_category")
            if pd.isna(default) or not str(default).strip():
                default = ""
            category = input(f"Category to approve [{default}]: ").strip() or str(default)
        elif decision == "C":
            category = input("Enter corrected category: ").strip()
        else:
            print("Invalid choice; left pending.")
            continue

        category = category.strip().upper()
        if not category:
            print("Category cannot be empty; left pending.")
            continue
        if category not in valid_categories:
            print("Category is not in the trained classifier's classes; left pending.")
            print("New categories are not added to the model until retraining.")
            continue

        notes = input("Review notes (optional): ").strip()
        existing_ids = set(pd.to_numeric(approved_df["feedback_id"], errors="coerce").dropna().astype(int))
        if fid not in existing_ids:
            original = record.get("original_text")
            clarified = record.get("clarified_text")
            phrase = clarified if pd.notna(clarified) and str(clarified).strip() else original
            new_row = {
                "Phrase": phrase,
                "Class": category,
                "feedback_id": fid,
                "original_text": original,
                "clarified_text": clarified,
                "approved_at": pd.Timestamp.now().isoformat(),
            }
            approved_df = pd.concat([approved_df, pd.DataFrame([new_row])], ignore_index=True)
            approved_df = approved_df[APPROVED_COLUMNS]
            approved_df.to_csv(APPROVED_DATA_PATH, index=False)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE user_feedback
                SET review_status='APPROVED', reviewed_category=?,
                    review_notes=?, reviewed_at=CURRENT_TIMESTAMP
                WHERE feedback_id=? AND review_status='PENDING'
            """, (category, notes, fid))
        print("Approved category:", category)
        print("Approved feedback exported (or was already present).")

    print("\nReview session completed.")


def export_approved_feedback() -> pd.DataFrame:
    """Export approved feedback to the training-review CSV without duplicating IDs."""
    Path(APPROVED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(APPROVED_DATA_PATH):
        current = pd.read_csv(APPROVED_DATA_PATH)
    else:
        current = pd.DataFrame(columns=APPROVED_COLUMNS)
    for col in APPROVED_COLUMNS:
        if col not in current.columns:
            current[col] = pd.NA
    current = current[APPROVED_COLUMNS]

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_feedback_schema(conn)
        approved = pd.read_sql_query("""
            SELECT feedback_id, original_text, clarified_text,
                   reviewed_category, reviewed_at
            FROM user_feedback
            WHERE review_status='APPROVED'
            ORDER BY feedback_id
        """, conn)

    existing_ids = set(pd.to_numeric(current["feedback_id"], errors="coerce").dropna().astype(int))
    rows = []
    for _, row in approved.iterrows():
        fid = int(row["feedback_id"])
        if fid in existing_ids:
            continue
        phrase = row["clarified_text"] if pd.notna(row["clarified_text"]) and str(row["clarified_text"]).strip() else row["original_text"]
        rows.append({
            "Phrase": phrase,
            "Class": row["reviewed_category"],
            "feedback_id": fid,
            "original_text": row["original_text"],
            "clarified_text": row["clarified_text"],
            "approved_at": row["reviewed_at"] or pd.Timestamp.now().isoformat(),
        })
        existing_ids.add(fid)

    if rows:
        current = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
        current = current[APPROVED_COLUMNS]
        current.to_csv(APPROVED_DATA_PATH, index=False)
    print(f"Exported {len(rows)} new approved examples.")
    print("Approved dataset rows:", len(current))
    print("CSV path:", APPROVED_DATA_PATH)
    return current


def show_category_progress(min_examples: int = 50) -> Optional[pd.DataFrame]:
    if not os.path.exists(APPROVED_DATA_PATH):
        print("Approved feedback dataset does not exist yet.")
        return None
    approved_df = pd.read_csv(APPROVED_DATA_PATH)
    if approved_df.empty:
        print("No approved feedback examples available.")
        return None
    if "Phrase" not in approved_df or "Class" not in approved_df:
        raise ValueError("Approved CSV must contain Phrase and Class columns.")

    approved_df["normalised_phrase"] = (
        approved_df["Phrase"].fillna("").astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    summary = (
        approved_df.groupby("Class")
        .agg(
            approved_examples=("Phrase", "count"),
            unique_examples=("normalised_phrase", "nunique"),
        )
        .reset_index()
    )
    summary["examples_remaining"] = (
        min_examples - summary["unique_examples"]
    ).clip(lower=0)
    summary["ready_for_training_review"] = (
        summary["unique_examples"] >= min_examples
    )
    _display(summary)
    return summary


def show_feedback_status(min_examples: int = 50) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_feedback_schema(conn)
        status = pd.read_sql_query("""
            SELECT review_status, COUNT(*) AS submissions
            FROM user_feedback
            GROUP BY review_status
            ORDER BY review_status
        """, conn)
    print("Feedback review status:")
    _display(status)
    print("\nApproved category progress:")
    show_category_progress(min_examples=min_examples)
    return status


def check_database() -> dict:
    """Inspect database integrity, tables, counts, and foreign-key violations."""
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        tables = [
            row[0] for row in conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """).fetchall()
        ]
        counts = {}
        for table in tables:
            counts[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    result = {
        "integrity": integrity,
        "foreign_key_errors": len(fk_errors),
        "tables": tables,
        "row_counts": counts,
    }
    print("Database integrity:", result["integrity"])
    print("Foreign-key violations:", result["foreign_key_errors"])
    print("Tables:", result["tables"])
    print("Row counts:", result["row_counts"])
    return result


__all__ = [
    "initialize_wasteloop",
    "classify_with_clarification",
    "save_user_feedback",
    "extract_information",
    "find_recovery_options",
    "display_recommendations",
    "run_wasteloop",
    "review_pending_feedback",
    "export_approved_feedback",
    "show_feedback_status",
    "show_category_progress",
    "check_database",
]
