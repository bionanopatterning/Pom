import streamlit as st
import pandas as pd
import os
from PIL import Image
import numpy as np
import glob
import json
from Pom.app.util import load_data, get_image
from Pom.core.tools import get_tomogram_by_name

st.set_page_config(
    page_title="Tomogram details",
    layout='wide'
)

os.makedirs(os.path.join("pom", "subsets"), exist_ok=True)

def read_subset(subset):
    subset_path = os.path.join("pom", "subsets", f"{subset}.txt")
    if os.path.exists(subset_path):
        return [t for t in open(subset_path).read().splitlines() if t.strip()]
    else:
        return []

def add_to_subset(subset, tomo):
    subset_path = os.path.join("pom", "subsets", f"{subset}.txt")
    if not os.path.exists(subset_path):
        with open(subset_path, 'w') as f:
            f.write(f'{tomo}\n')
    else:
        tomos = read_subset(subset)
        if tomo not in tomos:
            tomos.append(tomo)
            with open(subset_path, 'w') as f:
                f.write('\n'.join(tomos) + '\n')

def remove_from_subset(subset, tomo):
    subset_path = os.path.join("pom", "subsets", f"{subset}.txt")
    if os.path.exists(subset_path):
        tomos = read_subset(subset)
        if tomo in tomos:
            tomos.remove(tomo)
            with open(subset_path, 'w') as f:
                f.write('\n'.join(tomos) + '\n')

def create_subset():
    name = st.session_state.new_subset_name.strip()
    if not name:
        return
    add_to_subset(name, tomo_name)

df = load_data()

# Load available compositions
compositions_path = os.path.join('pom', 'image_compositions.json')
if os.path.exists(compositions_path):
    with open(compositions_path, 'r') as f:
        available_compositions = list(json.load(f).keys())
else:
    available_compositions = []

# Set default composition display
if 'composition_display' not in st.session_state:
    st.session_state.composition_display = 'thumbnail' if 'thumbnail' in available_compositions else (available_compositions[0] if available_compositions else 'thumbnail')

def open_in_ais(tomo_name):
    from Pom.core.tools import get_tomogram_by_name
    cmd_path = os.path.join(os.path.expanduser("~"), ".Ais", "pom_to_ais.cmd")
    with open(cmd_path, 'a') as f:
        mrc_path = os.path.abspath(get_tomogram_by_name(tomo_name))
        scns_path = mrc_path.replace('.mrc', '.scns')
        if os.path.exists(scns_path):
            f.write(f"open\t{scns_path}\n")
        else:
            f.write(f"open\t{mrc_path}\n")

# Query params
tomo_name = df.index[0]
if "tomo_id" in st.query_params:
    tomo_name = st.query_params["tomo_id"]


tomo_subsets = [os.path.splitext(os.path.basename(j))[0] for j in glob.glob(os.path.join("pom", "subsets", "*.txt"))]

tomo_names = df.index.tolist()
_, column_base, _ = st.columns([1, 15, 1])

with column_base:
    # Navigation and title
    _, c1, c2, c3, _ = st.columns([5, 1, 8, 1, 5])
    with c1:
        if st.button(":material/Keyboard_Arrow_Left:"):
            idx = tomo_names.index(tomo_name)
            idx = (idx - 1) % len(tomo_names)
            tomo_name = tomo_names[idx]
            st.query_params["tomo_id"] = tomo_name
    with c3:
        if st.button(":material/Keyboard_Arrow_Right:"):
            idx = tomo_names.index(tomo_name)
            idx = (idx + 1) % len(tomo_names)
            tomo_name = tomo_names[idx]
            st.query_params["tomo_id"] = tomo_name
    with c2:
        tomo_title_field = st.markdown(f'<div style="text-align: center;font-size: 30px;margin-bottom: 0; margin-top: 0;"><b>{tomo_name}</b></div>', unsafe_allow_html=True)

    " "
    # Ais link and subsets
    file_found = os.path.exists(get_tomogram_by_name(tomo_name))
    if file_found:
        columns = st.columns([1.2, 5, 1.5], vertical_alignment="bottom")
        if columns[0].button("Open in Ais", type="primary", width="stretch"):
            open_in_ais(tomo_name)
    else:
        columns = st.columns([0, 5, 2], vertical_alignment="bottom")

    with columns[1]:
        in_subsets = []
        for subset in tomo_subsets:
            subset_tomos = read_subset(subset)
            if tomo_name in subset_tomos:
                in_subsets.append(subset)

        new_subsets = st.multiselect("Include in tomogram subsets", options=tomo_subsets, default=in_subsets, key=f'subset_select_{tomo_name}')

        if in_subsets != new_subsets:
            for subset in tomo_subsets:
                if subset in new_subsets:
                    add_to_subset(subset, tomo_name)
                else:
                    remove_from_subset(subset, tomo_name)

    with columns[2]:
        st.text_input(
            "Create new subset",
            key="new_subset_name",
            placeholder="Subset name",
            on_change=create_subset
        )

    cols = st.columns([1, 1])
    with cols[0]:
        img = get_image(tomo_name, "density").transpose(Image.FLIP_TOP_BOTTOM)
        st.image(img, caption='Central slice', width="stretch")
    with cols[1]:
        img = get_image(tomo_name, st.session_state.composition_display)
        st.image(img, caption=st.session_state.composition_display, width="stretch")
        if available_compositions:
            st.selectbox(
                "Composition",
                available_compositions,
                key="composition_display",
                label_visibility="collapsed"
            )

    st.text("")

    features = df.loc[tomo_name].sort_values(ascending=False).index.tolist()

    n_imgs_per_row = 5
    while features != []:
        n_cols = min(len(features), n_imgs_per_row)
        col_features = features[:n_cols]
        features = features[n_cols:]
        for o, c in zip(col_features, st.columns(n_imgs_per_row)):
            with c:
                volume_fraction = df.loc[tomo_name, o]
                st.text(f"{o} ({volume_fraction:.1f}%)")
                st.image(get_image(tomo_name, o).transpose(Image.FLIP_TOP_BOTTOM), width="stretch")

