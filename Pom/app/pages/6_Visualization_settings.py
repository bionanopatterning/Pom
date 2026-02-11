import streamlit as st
import json
import os
from Pom.app.util import load_data
from Pom.core.tools import get_feature_library, save_feature_library
st.set_page_config(page_title="Render settings", layout="wide")

st.title("Visualization settings")

st.subheader("Feature library")
st.write("Edit how each feature is rendered in the browser when using 'pom render'.")

df = load_data()
feature_names = list(df.columns)
library = get_feature_library()

for feat in feature_names:
    if feat not in library:
        library[feat] = {
            "visible": True,
            "color": "#ff0000",
            "sigma": 1.0,  # Å
            "dust": 0.0,  # Å^3
            "threshold": 0.5,  # 0..1
        }

if "library" not in st.session_state:
    st.session_state.library = library

lib = st.session_state.library

"  "
"  "

with st.columns([1, 6, 1])[1]:
    for feat in sorted(feature_names):
        row = st.columns([2, 1.5, 2, 2.5, 2.5, 3.5])

        row[0].markdown(
            f"<div style='margin-top: 0.6em; font-weight: 500;'>{feat}</div>",
            unsafe_allow_html=True,
        )

        color = row[1].color_picker(
            "Color",
            value=lib[feat]["color"],
            key=f"color_{feat}",
            label_visibility="collapsed",
        )

        visible = row[2].checkbox(
            "Visible",
            value=lib[feat]["visible"],
            key=f"visible_{feat}"
        )

        sigma = row[3].slider(
            "Sigma",
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            value=float(lib[feat]["sigma"]),
            key=f"sigma_{feat}",
            label_visibility="collapsed",
            format="Sigma = %.1f Å",
        )

        dust = row[4].slider(
            "Dust",
            min_value=0.0,
            max_value=5_000_000.0,
            step=10_000.0,
            value=float(lib[feat]["dust"]),
            key=f"dust_{feat}",
            label_visibility="collapsed",
            format="Dust = %.0f Å³",
        )

        threshold = row[5].slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(lib[feat]["threshold"]),
            key=f"threshold_{feat}",
            label_visibility="collapsed",
            format="Threshold = %.2f",
        )

        # Write back
        lib[feat]["color"] = color
        lib[feat]["visible"] = visible
        lib[feat]["sigma"] = sigma
        lib[feat]["dust"] = dust
        lib[feat]["threshold"] = threshold

save_feature_library(lib)
# ---------------------------------------------------------------------
# Image compositions
# ---------------------------------------------------------------------


"  "
"  "
"  "
"  "



st.subheader("Image compositions")
st.write("Define which features to render with 'pom render', and how the resulting images are named in the browser. Note the name **thumbnail** defines the image used on the tomogram detail page and that you can change its composition.")

COMP_PATH = os.path.join("pom", "image_compositions.json")

# Helper functions
def load_compositions(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_compositions(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Load compositions
compositions = load_compositions(COMP_PATH)

if "compositions" not in st.session_state:
    st.session_state.compositions = compositions

comps = st.session_state.compositions

# Available options: features + rank1..rank5
rank_options = [f"rank{i}" for i in range(1, 6)]
all_options = sorted(feature_names + rank_options)

with st.columns([1, 6, 1])[1]:
    if not comps:
        st.info("No compositions defined yet.")
    else:
        for name in sorted(comps.keys()):
            with st.container(border=True):
                cols = st.columns([2, 6, 1])

                # Name
                cols[0].markdown(
                    f"<div style='margin-top: 0.5em; font-weight: 600;'>pom/images/{name}</div>",
                    unsafe_allow_html=True,
                )

                valid_defaults = [f for f in comps[name] if f in all_options]

                selected = cols[1].multiselect(
                    "Features",
                    options=all_options,
                    default=valid_defaults,
                    key=f"comp_feats_{name}",
                    label_visibility="collapsed",
                )

                if valid_defaults != comps[name]:
                    comps[name] = valid_defaults
                    save_compositions(COMP_PATH, comps)

                # Delete button
                if cols[2].button(":material/delete:", key=f"delete_{name}"):
                    del comps[name]
                    save_compositions(COMP_PATH, comps)
                    st.rerun()

                # Write back any edits
                if selected != comps[name]:
                    comps[name] = selected
                    save_compositions(COMP_PATH, comps)

    with st.expander("Add new composition", expanded=False):
        new_name = st.text_input("Composition name", key="new_comp_name")
        new_feats = st.multiselect("Select features", options=all_options, key="new_comp_feats")

        if st.button("Add composition"):
            if not new_name:
                st.warning("Please provide a name.")
            elif new_name in comps:
                st.warning("A composition with this name already exists.")
            else:
                comps[new_name] = new_feats
                save_compositions(COMP_PATH, comps)
                st.success(f"Added composition '{new_name}'")
                st.rerun()
