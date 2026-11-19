import os
import re
from flask import Flask, render_template, request, session, redirect, url_for
from dotenv import load_dotenv

load_dotenv()

# Ensure WASTEWISE_PROJECT_PATH points to the root of this project
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.environ.setdefault("WASTEWISE_PROJECT_PATH", PROJECT_ROOT)

from src import wasteloop
from src import rag

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "wastewise-local-dev-secret-key")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/ask", methods=["GET", "POST"])
def ask():
    question = ""
    result = None
    error = None

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            error = "Please enter a question about Indian waste-management regulations."
        else:
            try:
                result = rag.ask_wastewise(question)
            except Exception as exc:
                error = f"Unable to process your question right now: {exc}"

    return render_template("ask.html", question=question, result=result, error=error)


@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    if request.method == "GET" or request.form.get("action") == "reset":
        session.pop("analyzer_state", None)
        return render_template("analyzer.html", stage="initial", state={})

    action = request.form.get("action", "")
    state = session.get("analyzer_state", {})
    error = None

    # STEP 1: Initial description, city, and pincode
    if action == "start":
        description = request.form.get("description", "").strip()
        city = request.form.get("city", "").strip()
        pincode = request.form.get("pincode", "").strip()

        if not description:
            error = "Please enter a waste description."
        elif not city:
            error = "City is required."
        elif not re.fullmatch(r"\d{6}", pincode):
            error = "Please enter a valid 6-digit PIN code."

        if error:
            return render_template(
                "analyzer.html",
                stage="initial",
                state={"description": description, "city": city, "pincode": pincode},
                error=error,
            )

        wasteloop._ensure_models()
        probabilities, top_df = wasteloop._predict(description)
        best_index = int(probabilities.argmax())
        model_prediction = str(wasteloop.embedding_classifier.classes_[best_index])
        score = float(probabilities[best_index])

        state = {
            "description": description,
            "updated_description": description,
            "city": city,
            "pincode": pincode,
            "model_prediction": model_prediction,
            "model_score": score,
            "top_candidates": top_df.to_dict(orient="records"),
        }

        if score >= wasteloop.CONFIDENCE_THRESHOLD:
            state["predicted_class"] = model_prediction
            state["classification_status"] = "ACCEPTED"
            state["classification_source"] = "MODEL"
            return _proceed_to_extraction(state)

        state["stage"] = "clarify"
        session["analyzer_state"] = state
        return render_template("analyzer.html", stage="clarify", state=state)

    # STEP 2: Low-confidence clarification text
    elif action == "clarify":
        clarification = request.form.get("clarification", "").strip()
        text = state.get("description", "")

        if not clarification:
            state["classification_status"] = "UNRESOLVED"
            state["stage"] = "unresolved"
            session["analyzer_state"] = state
            return render_template("analyzer.html", stage="unresolved", state=state)

        updated_text = f"{text} {clarification}".strip()
        probabilities, top_df = wasteloop._predict(updated_text)
        best_index = int(probabilities.argmax())
        model_prediction = str(wasteloop.embedding_classifier.classes_[best_index])
        score = float(probabilities[best_index])

        state["updated_description"] = updated_text
        state["model_prediction"] = model_prediction
        state["model_score"] = score
        state["top_candidates"] = top_df.to_dict(orient="records")

        if score >= wasteloop.CONFIDENCE_THRESHOLD:
            state["predicted_class"] = model_prediction
            state["classification_status"] = "CLARIFIED"
            state["classification_source"] = "MODEL"
            return _proceed_to_extraction(state)

        state["stage"] = "select_category"
        session["analyzer_state"] = state
        return render_template("analyzer.html", stage="select_category", state=state)

    # NEW: Allow user to manually override category even when confidence is high
    elif action == "override_category":
        if not state.get("top_candidates") and state.get("updated_description"):
            _, top_df = wasteloop._predict(state["updated_description"])
            state["top_candidates"] = top_df.to_dict(orient="records")
        state["stage"] = "select_category"
        session["analyzer_state"] = state
        return render_template("analyzer.html", stage="select_category", state=state)

    # STEP 3: Manual top-10 category selection, new category suggestion, or skip
    elif action == "select_category":
        choice = request.form.get("choice", "").strip()
        suggestion = request.form.get("suggestion", "").strip()
        text = state.get("description", "")
        updated_text = state.get("updated_description", text)
        model_prediction = state.get("model_prediction")
        score = state.get("model_score")

        if choice == "new":
            if not suggestion:
                return render_template(
                    "analyzer.html",
                    stage="select_category",
                    state=state,
                    error="Please enter the name of the new waste category.",
                )
            feedback_id = wasteloop.save_user_feedback(
                text,
                updated_text,
                model_prediction,
                score,
                suggested_category=suggestion,
                feedback_type="NEW_CATEGORY_SUGGESTION",
            )
            state["classification_status"] = "NEW_CATEGORY_PENDING"
            state["feedback_id"] = feedback_id
            state["suggested_category"] = suggestion
            state["stage"] = "unresolved"
            session["analyzer_state"] = state
            return render_template("analyzer.html", stage="unresolved", state=state)

        elif choice == "skip" or not choice:
            feedback_id = wasteloop.save_user_feedback(
                text,
                updated_text,
                model_prediction,
                score,
                feedback_type="UNRESOLVED",
            )
            state["classification_status"] = "UNRESOLVED"
            state["feedback_id"] = feedback_id
            state["stage"] = "unresolved"
            session["analyzer_state"] = state
            return render_template("analyzer.html", stage="unresolved", state=state)

        else:
            selected_category = choice
            feedback_type = (
                "USER_CONFIRMATION"
                if selected_category == model_prediction
                else "USER_CORRECTION"
            )
            feedback_id = wasteloop.save_user_feedback(
                text,
                updated_text,
                model_prediction,
                score,
                selected_category=selected_category,
                feedback_type=feedback_type,
            )
            state["predicted_class"] = selected_category
            state["classification_status"] = "USER_SELECTED"
            state["classification_source"] = "USER"
            state["feedback_id"] = feedback_id
            return _proceed_to_extraction(state)

    # STEP 4: Confirm/fill extracted quantity, condition, and recovery intent
    elif action == "confirm_details":
        qty_raw = request.form.get("quantity", "").strip()
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms?|g|grams?|items?|pieces?|pcs|"
            r"bottles?|cans?|jars?|packets?|pouches?|bags?|cups?|boxes?|containers?)?",
            qty_raw.lower(),
        )
        if not match or float(match.group(1)) <= 0:
            return render_template(
                "analyzer.html",
                stage="confirm_details",
                state=state,
                error="Please enter a valid positive quantity (e.g. 5 items or 2 kg).",
            )

        raw_unit = match.group(2) or state.get("quantity_unit") or "items"
        normalized_unit = (
            "kg"
            if raw_unit.startswith("k")
            else "g"
            if raw_unit in {"g", "gram", "grams"}
            else "items"
        )
        quantity = float(match.group(1))

        # Condition handling matching _collect_condition
        existing_conditions = state.get("condition", [])
        condition_notes = state.get("condition_notes", state.get("updated_description", ""))
        if existing_conditions:
            conditions = existing_conditions
        else:
            cond_input = request.form.get("condition_input", "").strip()
            conditions = [
                kw
                for kw in wasteloop.CONDITION_KEYWORDS
                if re.search(r"\b" + re.escape(kw) + r"\b", cond_input.lower())
            ]
            if not conditions:
                conditions = ["unknown"]
            condition_notes = (condition_notes + " " + cond_input).strip()

        # Intent handling matching _collect_intent
        existing_intents = state.get("user_intent", [])
        if existing_intents:
            intents = existing_intents
        else:
            intent_input = request.form.get("intent_input", "").strip().lower()
            if "any" in intent_input or "all" in intent_input or not intent_input:
                intents = ["any"]
            else:
                selected = [
                    intent
                    for intent, keywords in wasteloop.INTENT_KEYWORDS.items()
                    if any(kw in intent_input for kw in keywords)
                ]
                intents = selected or ["any"]

        user_info = {
            "description": state["description"],
            "updated_description": state["updated_description"],
            "predicted_class": state["predicted_class"],
            "model_score": state.get("model_score"),
            "quantity": quantity,
            "quantity_unit": normalized_unit,
            "condition": conditions,
            "condition_notes": condition_notes,
            "user_intent": intents,
            "city": state["city"],
            "pincode": state["pincode"],
            "classification_status": state.get("classification_status"),
            "classification_source": state.get("classification_source", "MODEL"),
        }

        recommendations_df = wasteloop.find_recovery_options(user_info)
        recommendations = (
            recommendations_df.to_dict(orient="records")
            if recommendations_df is not None and not recommendations_df.empty
            else []
        )

        state["user_info"] = user_info
        state["recommendations"] = recommendations
        state["stage"] = "results"
        session["analyzer_state"] = state
        return render_template("analyzer.html", stage="results", state=state)

    return redirect(url_for("analyzer"))


def _proceed_to_extraction(state: dict):
    working_text = state.get("updated_description", state.get("description", ""))
    extracted = wasteloop.extract_information(working_text)

    state["quantity"] = extracted.get("quantity")
    state["quantity_unit"] = extracted.get("quantity_unit")
    state["condition"] = extracted.get("condition", [])
    state["condition_notes"] = extracted.get("condition_notes", working_text)
    state["user_intent"] = extracted.get("user_intent", [])
    state["stage"] = "confirm_details"

    session["analyzer_state"] = state
    return render_template("analyzer.html", stage="confirm_details", state=state)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)