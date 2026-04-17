# streamlit_app.py
# TrueOffer — Streamlit UI

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Readable label mappings for categorical features
# ---------------------------------------------------------------------------

QUALITY_LABELS = {
    "Ex": "Excellent", "Gd": "Good", "TA": "Average", "Fa": "Fair",
}
QUALITY_CODES = {v: k for k, v in QUALITY_LABELS.items()}

NEIGHBORHOOD_LABELS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem",
    "BrDale": "Briardale",            "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",         "CollgCr": "College Creek",
    "Crawfor": "Crawford",            "Edwards": "Edwards",
    "Gilbert": "Gilbert",             "IDOTRR": "Iowa DOT & Rail Road",
    "MeadowV": "Meadow Village",      "Mitchel": "Mitchell",
    "NAmes": "North Ames",            "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",     "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames",       "OldTown": "Old Town",
    "SWISU": "South & West of ISU",   "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",         "Somerst": "Somerset",
    "StoneBr": "Stone Brook",         "Timber": "Timberland",
    "Veenker": "Veenker",
}
NEIGHBORHOOD_CODES = {v: k for k, v in NEIGHBORHOOD_LABELS.items()}

# ---------------------------------------------------------------------------
# Feature metadata (options use readable labels; codes resolved before API call)
# ---------------------------------------------------------------------------

FEATURE_META = {
    "OverallQual":  {"label": "Overall Quality (1–10)", "type": "int",    "min": 1,    "max": 10},
    "GrLivArea":    {"label": "Living Area (sq ft)",     "type": "float",  "min": 300,  "max": 6000},
    "GarageCars":   {"label": "Garage Spaces",           "type": "int",    "min": 0,    "max": 5},
    "TotalBsmtSF":  {"label": "Basement Area (sq ft)",   "type": "float",  "min": 0,    "max": 6000},
    "1stFlrSF":     {"label": "1st Floor Area (sq ft)",  "type": "float",  "min": 300,  "max": 5000},
    "FullBath":     {"label": "Full Bathrooms",          "type": "int",    "min": 0,    "max": 6},
    "YearBuilt":    {"label": "Year Built",              "type": "int",    "min": 1872, "max": 2010},
    "ExterQual":    {"label": "Exterior Quality",        "type": "select", "options": list(QUALITY_LABELS.values())},
    "KitchenQual":  {"label": "Kitchen Quality",         "type": "select", "options": list(QUALITY_LABELS.values())},
    "Neighborhood": {"label": "Neighborhood",            "type": "select", "options": list(NEIGHBORHOOD_LABELS.values())},
}

FEATURE_DEFAULTS = {
    "OverallQual":  6,       "GrLivArea":   1465.0,
    "GarageCars":   2,       "TotalBsmtSF": 997.5,
    "1stFlrSF":     1095.0,  "FullBath":    2,
    "YearBuilt":    1972,
    "ExterQual":    QUALITY_LABELS["TA"],
    "KitchenQual":  QUALITY_LABELS["TA"],
    "Neighborhood": NEIGHBORHOOD_LABELS["NAmes"],
}

def _to_api_value(feat: str, display_val):
    """Convert readable label back to API code for categorical features."""
    if feat in ("ExterQual", "KitchenQual"):
        return QUALITY_CODES.get(display_val, display_val)
    if feat == "Neighborhood":
        return NEIGHBORHOOD_CODES.get(display_val, display_val)
    return display_val

# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------

def _call_api(message: str, assumed_overrides: dict | None = None) -> dict:
    try:
        resp = requests.post(
            "http://localhost:8001/analyze",
            json={"message": message, "assumed_overrides": assumed_overrides},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Start FastAPI with: uvicorn main:app --reload --port 8001"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out after 30 seconds."}
    if resp.status_code != 200:
        return {"error": f"Backend error {resp.status_code}: {resp.text[:200]}"}
    return resp.json()

# ---------------------------------------------------------------------------
# Widget builder
# ---------------------------------------------------------------------------

def _widget_for(feat: str, container):
    meta = FEATURE_META.get(feat, {})
    label = meta.get("label", feat)
    ftype = meta.get("type", "float")
    default = FEATURE_DEFAULTS.get(feat)
    if ftype == "select":
        opts = meta["options"]
        idx = opts.index(default) if default in opts else 0
        return container.selectbox(label, opts, index=idx, key=f"override_{feat}")
    elif ftype == "int":
        return container.number_input(
            label, min_value=meta["min"], max_value=meta["max"],
            value=int(default or meta["min"]), step=1, key=f"override_{feat}",
        )
    else:
        return container.number_input(
            label, min_value=float(meta["min"]), max_value=float(meta["max"]),
            value=float(default or meta["min"]), step=1.0, key=f"override_{feat}",
        )

# ---------------------------------------------------------------------------
# Page config & header
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TrueOffer", layout="wide")
st.title("TrueOffer — Is This Price Fair?")
st.caption("Describe a property in plain English to get a price verdict, or ask a market question.")

# ---------------------------------------------------------------------------
# TOP ROW: input left | assumed features right
# ---------------------------------------------------------------------------

input_col, features_col = st.columns([1, 1])

with input_col:
    st.subheader("Describe the Property")

    with st.expander("💡 What to include in your description"):
        st.markdown(
            "The more detail you provide, the fewer features will be assumed:\n"
            "- **Size** — living area, basement area, 1st floor sq ft\n"
            "- **Bedrooms & bathrooms** — full bathrooms count\n"
            "- **Garage** — number of garage spaces\n"
            "- **Quality** — overall quality (1–10), exterior & kitchen finish\n"
            "- **Location** — neighborhood name (e.g. North Ames, Northridge Heights)\n"
            "- **Age** — year built\n\n"
            "**Example:** *3-bedroom house in Northridge Heights, built in 2003, "
            "2200 sqft living area, 2 garage spaces, excellent kitchen*"
        )

    with st.expander("💬 Ask a market question instead"):
        st.markdown(
            "You can also ask for market insights:\n"
            "- *What are the most expensive neighborhoods in Ames?*\n"
            "- *Which neighborhoods offer the best value for money?*\n"
            "- *How does overall quality affect home prices?*\n"
            "- *What is the typical price range for homes in North Ames?*\n"
            "- *How much does a garage add to a home's value?*"
        )

    st.text_area(
        "Your message",
        key="user_message",
        height=140,
        placeholder=(
            "e.g. 3-bedroom house in North Ames, built in 1995, "
            "1800 sqft living area, 2 garage spaces, good kitchen"
        ),
    )

    btn_col, msg_col = st.columns([1, 2])
    with btn_col:
        analyze_clicked = st.button("Analyze", type="primary")
    with msg_col:
        if (
            "result" in st.session_state
            and st.session_state["result"].get("type") == "prediction"
            and not st.session_state.get("assumed_features")
        ):
            st.caption("✅ All features extracted — no assumptions made.")

    if analyze_clicked:
        if not st.session_state.user_message.strip():
            st.warning("Please enter a property description.")
        else:
            result = _call_api(st.session_state.user_message)
            st.session_state["result"] = result
            st.session_state["submitted_message"] = st.session_state.user_message
            st.session_state["assumed_features"] = (
                result["data"].get("assumed_features", [])
                if result.get("type") == "prediction" else []
            )

with features_col:
    assumed = st.session_state.get("assumed_features", [])
    if assumed:
        st.subheader("Correct Assumed Values")
        st.caption(
            f"{len(assumed)} feature(s) were assumed by the AI — adjust for a refined prediction."
        )
        overrides = {}
        with st.form("assumed_overrides_form"):
            left, right = st.columns(2)
            for i, feat in enumerate(assumed):
                col = left if i % 2 == 0 else right
                display_val = _widget_for(feat, col)
                overrides[feat] = _to_api_value(feat, display_val)
            if st.form_submit_button("Re-run with My Values", type="primary"):
                result = _call_api(st.session_state["submitted_message"], assumed_overrides=overrides)
                st.session_state["result"] = result
                st.session_state["assumed_features"] = (
                    result["data"].get("assumed_features", [])
                    if result.get("type") == "prediction" else []
                )
                st.rerun()

# ---------------------------------------------------------------------------
# BOTTOM: result
# ---------------------------------------------------------------------------

st.divider()

if "result" not in st.session_state:
    st.info("Enter a description above and click Analyze.")
else:
    result = st.session_state["result"]
    if "error" in result:
        st.error(result["error"])
    elif result["type"] == "prediction":
        data = result["data"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", data["verdict"].upper())
        c2.metric("Predicted Price", f"${data['predicted_price']:,.0f}")
        delta = data["price_delta"]
        c3.metric("vs. Median", f"+${delta:,.0f}" if delta >= 0 else f"-${abs(delta):,.0f}")
        if data.get("explanation"):
            st.write(data["explanation"])
    elif result["type"] == "insight":
        st.subheader("Market Insight")
        import re
        answer = re.sub(r"`([^`]+)`", r"\1", result["data"]["answer"])
        st.write(answer)
        if result["data"].get("sources"):
            st.caption("Sources: " + ", ".join(result["data"]["sources"]))

if __name__ == "__main__":
    pass
