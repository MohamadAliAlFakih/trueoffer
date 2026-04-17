# streamlit_app.py
# TrueOffer — two-panel Streamlit UI
# Left panel: property description input + assumed features editor
# Right panel: prediction verdict / market insight result

import json

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Feature metadata for the assumed-features editor (Task 2)
# ---------------------------------------------------------------------------

FEATURE_META = {
    "OverallQual":  {"label": "Overall Quality (1-10)",  "type": "int",    "min": 1,    "max": 10},
    "GrLivArea":    {"label": "Living Area (sq ft)",      "type": "float",  "min": 300,  "max": 6000},
    "GarageCars":   {"label": "Garage Spaces",            "type": "int",    "min": 0,    "max": 5},
    "TotalBsmtSF":  {"label": "Basement Area (sq ft)",    "type": "float",  "min": 0,    "max": 6000},
    "1stFlrSF":     {"label": "1st Floor Area (sq ft)",   "type": "float",  "min": 300,  "max": 5000},
    "FullBath":     {"label": "Full Bathrooms",           "type": "int",    "min": 0,    "max": 6},
    "YearBuilt":    {"label": "Year Built",               "type": "int",    "min": 1872, "max": 2010},
    "ExterQual":    {"label": "Exterior Quality",         "type": "select", "options": ["Ex", "Gd", "TA", "Fa"]},
    "KitchenQual":  {"label": "Kitchen Quality",          "type": "select", "options": ["Ex", "Gd", "TA", "Fa"]},
    "Neighborhood": {"label": "Neighborhood",             "type": "select", "options": [
        "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards",
        "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill", "NridgHt",
        "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
    ]},
}

FEATURE_DEFAULTS = {
    "OverallQual":  6,
    "GrLivArea":    1465.0,
    "GarageCars":   2,
    "TotalBsmtSF":  997.5,
    "1stFlrSF":     1095.0,
    "FullBath":     2,
    "YearBuilt":    1972,
    "ExterQual":    "TA",
    "KitchenQual":  "TA",
    "Neighborhood": "NAmes",
}

# ---------------------------------------------------------------------------
# Helper: call the FastAPI backend
# ---------------------------------------------------------------------------

def _call_api(message: str, assumed_overrides: dict | None = None) -> dict:
    """POST to /analyze. Returns the response dict or an {"error": ...} dict."""
    try:
        resp = requests.post("http://localhost:8001/analyze", json={"message": message, "assumed_overrides": assumed_overrides}, timeout=30)
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Start FastAPI with: uvicorn main:app --reload"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out after 30 seconds."}

    if resp.status_code != 200:
        return {"error": f"Backend error {resp.status_code}: {resp.text[:200]}"}

    return resp.json()


# ---------------------------------------------------------------------------
# Right-panel renderers
# ---------------------------------------------------------------------------

def _render_insight(data: dict) -> None:
    st.subheader("Market Insight")
    st.write(data["answer"])
    if data.get("sources"):
        st.caption("Sources: " + ", ".join(data["sources"]))


def _render_prediction(data: dict) -> None:
    verdict_color = {"fair": "green", "high": "red", "low": "orange"}.get(data["verdict"], "gray")
    st.markdown(f"**Verdict:** :{verdict_color}[{data['verdict'].upper()}]")
    st.metric("Predicted Price", f"${data['predicted_price']:,.0f}")
    delta_label = (
        f"+${data['price_delta']:,.0f}"
        if data["price_delta"] >= 0
        else f"-${abs(data['price_delta']):,.0f}"
    )
    st.caption(f"vs. dataset median: {delta_label}")

    assumed_count = len(data.get("assumed_features", []))
    if assumed_count > 0:
        st.info(
            f"{assumed_count} feature(s) were assumed (not mentioned in your description). "
            "Correct them on the left for a refined price."
        )

    st.write(data.get("explanation", ""))

    # Store assumed_features list so the editor can render on the left
    st.session_state["assumed_features"] = data.get("assumed_features", [])


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TrueOffer", layout="wide")
st.title("TrueOffer — Is This Price Fair?")
st.caption("Describe the property in plain English. We'll tell you if the price is fair.")

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1, 1])

# ---- LEFT PANEL ----
with left_col:
    st.header("Describe the Property")
    st.text_area(
        "Your message",
        key="user_message",
        height=140,
        placeholder=(
            "e.g. 4-bedroom house in NAmes neighborhood, built in 1995, "
            "2000 sqft living area, 2 garage spaces"
        ),
    )

    if st.button("Analyze", type="primary"):
        if not st.session_state.user_message.strip():
            st.warning("Please enter a property description.")
        else:
            result = _call_api(st.session_state.user_message, assumed_overrides=None)
            st.session_state["result"] = result
            st.session_state["submitted_message"] = st.session_state.user_message
            # Reset assumed_features from prior run
            if result.get("type") == "prediction":
                st.session_state["assumed_features"] = result["data"].get("assumed_features", [])
            else:
                st.session_state["assumed_features"] = []

    # ---- ASSUMED FEATURES EDITOR ----
    assumed = st.session_state.get("assumed_features", [])
    if assumed:
        st.divider()
        st.subheader("Correct Assumed Values")
        st.caption(
            f"{len(assumed)} feature(s) were assumed by the AI. "
            "Adjust them below for a refined prediction."
        )
        overrides = {}
        with st.form("assumed_overrides_form"):
            for feat in assumed:
                meta = FEATURE_META.get(feat, {})
                label = meta.get("label", feat)
                ftype = meta.get("type", "float")
                current_default = FEATURE_DEFAULTS.get(feat)
                if ftype == "select":
                    opts = meta["options"]
                    idx = opts.index(current_default) if current_default in opts else 0
                    overrides[feat] = st.selectbox(label, opts, index=idx, key=f"override_{feat}")
                elif ftype == "int":
                    overrides[feat] = st.number_input(
                        label,
                        min_value=meta.get("min", 0),
                        max_value=meta.get("max", 9999),
                        value=int(current_default or meta.get("min", 0)),
                        step=1,
                        key=f"override_{feat}",
                    )
                else:
                    overrides[feat] = st.number_input(
                        label,
                        min_value=float(meta.get("min", 0)),
                        max_value=float(meta.get("max", 9999)),
                        value=float(current_default or meta.get("min", 0.0)),
                        step=1.0,
                        key=f"override_{feat}",
                    )
            submitted = st.form_submit_button("Re-run with My Values", type="primary")
            if submitted:
                result = _call_api(st.session_state["submitted_message"], assumed_overrides=overrides)
                st.session_state["result"] = result
                # Update assumed_features for next render cycle
                if result.get("type") == "prediction":
                    st.session_state["assumed_features"] = result["data"].get("assumed_features", [])
                st.rerun()

# ---- RIGHT PANEL ----
with right_col:
    st.header("Result")
    if "result" not in st.session_state:
        st.info("Enter a description on the left and click Analyze.")
    else:
        result = st.session_state["result"]
        if "error" in result:
            st.error(result["error"])
        elif result.get("type") == "prediction":
            _render_prediction(result["data"])
        elif result.get("type") == "insight":
            _render_insight(result["data"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pass  # streamlit run streamlit_app.py
