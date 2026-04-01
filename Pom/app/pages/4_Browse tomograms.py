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

def _tomo_name_from_entry(entry):
    """Return the bare tomogram name from a subset entry (full path or plain name)."""
    return os.path.splitext(os.path.basename(entry))[0]

def _migrate_subset_to_full_paths(subset_path, entries):
    # backwards compatibility (260331): rewrite plain tomo names to full paths on first read
    needs_migration = any(os.sep not in e and '/' not in e for e in entries)
    if not needs_migration:
        return entries
    migrated = []
    for e in entries:
        if os.sep not in e and '/' not in e:
            migrated.append(get_tomogram_by_name(e) or e)
        else:
            migrated.append(e)
    with open(subset_path, 'w') as f:
        f.write('\n'.join(migrated) + '\n')
    return migrated

def read_subset(subset):
    subset_path = os.path.join("pom", "subsets", f"{subset}.txt")
    if os.path.exists(subset_path):
        entries = [t for t in open(subset_path).read().splitlines() if t.strip()]
        return _migrate_subset_to_full_paths(subset_path, entries)
    else:
        return []

def add_to_subset(subset, tomo):
    from Pom.core.tools import get_tomogram_by_name
    subset_path = os.path.join("pom", "subsets", f"{subset}.txt")
    full_path = get_tomogram_by_name(tomo) or tomo
    tomos = read_subset(subset)
    if tomo not in [_tomo_name_from_entry(t) for t in tomos]:
        tomos.append(full_path)
        with open(subset_path, 'w') as f:
            f.write('\n'.join(tomos) + '\n')

def remove_from_subset(subset, tomo):
    subset_path = os.path.join("pom", "subsets", f"{subset}.txt")
    if os.path.exists(subset_path):
        tomos = read_subset(subset)
        tomos = [t for t in tomos if _tomo_name_from_entry(t) != tomo]
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
        aislink_path = mrc_path.replace('.mrc', '.aislink')
        if os.path.exists(aislink_path):
            mrc_path = f'Z:/compu_projects/easymode/volumes_cryocare/{open(aislink_path).read().strip()}'
        scns_path = mrc_path.replace('.mrc', '.scns')
        if os.path.exists(scns_path):
            f.write(f"open\t{scns_path}\n")
        else:
            f.write(f"open\t{mrc_path}\n")

# Query params
tomo_name = df.index[0]
if "tomo_id" in st.query_params:
    tomo_name = st.query_params["tomo_id"]


tomo_subsets = sorted([os.path.splitext(os.path.basename(j))[0] for j in glob.glob(os.path.join("pom", "subsets", "*.txt"))])

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
    tomo_file = get_tomogram_by_name(tomo_name)
    file_found = tomo_file and os.path.exists(tomo_file)
    if file_found:
        columns = st.columns([1.2, 5, 1.5], vertical_alignment="bottom")
        if columns[0].button("Open in Ais", type="primary", width="stretch"):
            open_in_ais(tomo_name)
    else:
        columns = st.columns([0.01, 5, 2], vertical_alignment="bottom")

    with columns[1]:
        in_subsets = []
        for subset in tomo_subsets:
            subset_tomos = read_subset(subset)
            if tomo_name in [_tomo_name_from_entry(t) for t in subset_tomos]:
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

    row_data = df.loc[tomo_name]
    volume_features = [f for f in row_data.index if not f.startswith('particle_')]
    particle_features = [f for f in row_data.index if f.startswith('particle_')]

    volume_features = sorted(volume_features, key=lambda f: row_data[f], reverse=True)

    n_imgs_per_row = 5
    while volume_features:
        n_cols = min(len(volume_features), n_imgs_per_row)
        col_features = volume_features[:n_cols]
        volume_features = volume_features[n_cols:]
        for o, c in zip(col_features, st.columns(n_imgs_per_row)):
            with c:
                volume_fraction = row_data[o]
                st.text(f"{o} ({volume_fraction:.1f}%)")
                st.image(get_image(tomo_name, o).transpose(Image.FLIP_TOP_BOTTOM), width="stretch")

    if particle_features:
        st.text("")
        st.markdown("**Particles**")
        for f in particle_features:
            count = int(row_data[f]) if not pd.isna(row_data[f]) else 0
            label = f.removeprefix('particle_')
            st.text(f"{label}: {count}")