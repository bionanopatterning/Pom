import streamlit as st
import json
from Pom.core.config import parse_feature_library, FeatureLibraryFeature
import pandas as pd
import os
from PIL import Image
import numpy as np
import glob
import copy
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Tomogram details",
    layout='wide'
)

if "show_density" not in st.session_state:
    st.session_state.show_density       = True
if "show_macromolecules" not in st.session_state:
    st.session_state.show_macromolecules = True
if "show_organelles" not in st.session_state:
    st.session_state.show_organelles    = True

with open("project_configuration.json", 'r') as f:
    project_configuration = json.load(f)


def get_image(tomo, image, projection=False):
    image_dir = image.split("_")[0]
    if projection:
        img_path = os.path.join(project_configuration["root"], project_configuration["image_dir"], f"{image_dir}_projection", f"{tomo}_{image}.png")
    else:
        img_path = os.path.join(project_configuration["root"], project_configuration["image_dir"], image_dir, f"{tomo}_{image}.png")
    if os.path.exists(img_path):
        return Image.open(img_path)
    else:
        return Image.fromarray(np.zeros((128, 128)), mode='L')


def recolor(color, style=0):
    if style == 0:
        return (np.array(color) / 2.0 + 0.5)
    if style == 1:
        return (np.array(color) / 8 + 0.875)
    else:
        return color

def create_subset():
    name = st.session_state.new_subset_name.strip()
    if not name:
        return
    subset_path = os.path.join(project_configuration["root"],
                               "subsets",
                               f"{name}.json")
    if not os.path.exists(subset_path):                      # idempotency
        with open(subset_path, "w") as f:
            json.dump({"tomos": [tomo_name]}, f)


feature_library = parse_feature_library("feature_library.txt")


@st.cache_data
def load_data():
    cache_df = pd.read_excel(os.path.join(project_configuration["root"], "summary.xlsx"), index_col=0)
    cache_df = cache_df.dropna(axis=0)
    to_drop = list()
    for f in project_configuration["macromolecules"]:
        if f in cache_df.columns:
            to_drop.append(f)
    cache_df = cache_df.drop(columns=to_drop)
    cache_rank_df = cache_df.rank(axis=0, ascending=False)

    return cache_df, cache_rank_df


df, rank_df = load_data()


def rank_distance_series(tomo_name, rank_df):
    m_ranks = rank_df.loc[tomo_name]
    distances = rank_df.apply(lambda row: np.sum((row - m_ranks)**2), axis=1)
    sorted_distances = distances.sort_values()
    return sorted_distances


def open_in_ais(tomo_name):
    cmd_path = os.path.join(os.path.expanduser("~"), ".Ais", "pom_to_ais.cmd")
    tomo_dir = os.path.join(project_configuration["root"], project_configuration["tomogram_dir"])
    with open(cmd_path, 'a') as f:
        base = os.path.abspath(os.path.join(tomo_dir, f"{tomo_name}"))
        if os.path.exists(base+".scns"):
            f.write(f"open\t{base+'.scns'}\n")
        else:
            f.write(f"open\t{base + '.mrc'}\n")

# Query params
tomo_name = df.index[0]
if "tomo_id" in st.query_params:
    tomo_name = st.query_params["tomo_id"]

if not os.path.exists(os.path.join(project_configuration["root"], "subsets")):
    os.makedirs(os.path.join(project_configuration["root"], "subsets"))

tomo_subsets = [os.path.splitext(os.path.basename(j))[0] for j in glob.glob(os.path.join(project_configuration["root"], "subsets", "*.json"))]

tomo_names = df.index.tolist()
_, column_base, _ = st.columns([1, 15, 1])
with column_base:
    c1, c2, c3 = st.columns([1, 19, 1])
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
    file_found = os.path.exists(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"{tomo_name}.mrc"))
    if file_found:
        columns = st.columns([1.2, 5, 1.5], vertical_alignment="bottom")
        if columns[0].button("Open in Ais", type="primary", use_container_width=True):
            open_in_ais(tomo_name)
    else:
        columns = st.columns([0, 5, 2], vertical_alignment="bottom")

    with columns[1]:
        in_subsets = []
        for subset in tomo_subsets:
            with open(os.path.join(project_configuration["root"], "subsets", f"{subset}.json"), 'r') as f:
                subset_tomos = json.load(f)['tomos']
            if tomo_name in subset_tomos:
                in_subsets.append(subset)

        new_subsets = st.multiselect("Include in tomogram subsets", options=tomo_subsets, default=in_subsets, key=f'subset_select_{tomo_name}')

        if in_subsets != new_subsets:
            for subset in tomo_subsets:
                if subset in new_subsets:
                    with open(os.path.join(project_configuration["root"], "subsets", f"{subset}.json"), 'r') as f:
                        tomos = json.load(f)['tomos']
                        if tomo_name not in tomos:
                            tomos.append(tomo_name)
                    with open(os.path.join(project_configuration["root"], "subsets", f"{subset}.json"), 'w') as f:
                        json.dump({'tomos': tomos}, f)
                else:
                    with open(os.path.join(project_configuration["root"], "subsets", f"{subset}.json"), 'r') as f:
                        tomos = json.load(f)['tomos']
                        if tomo_name in tomos:
                            tomos.remove(tomo_name)
                    with open(os.path.join(project_configuration["root"], "subsets", f"{subset}.json"), 'w') as f:
                        json.dump({'tomos': tomos}, f)

    with columns[2]:
        st.text_input(
            "Create new subset",
            key="new_subset_name",
            placeholder="Subset name",
            on_change=create_subset
        )


    cb1, cb2, cb3 = st.columns([1, 1, 1])
    with cb1:
        st.checkbox("Show density", key="show_density")
    with cb2:
        st.checkbox("Show macromolecules", key="show_macromolecules")
    with cb3:
        st.checkbox("Show organelles", key="show_organelles")

    blocks = []
    if st.session_state.show_density:
        blocks.append(("density", "Density (central slice)", True))
    if st.session_state.show_macromolecules:
        blocks.append(("Macromolecules", "Macromolecules", False))
    if st.session_state.show_organelles:
        blocks.append(("Top3", "Top 3 organelles", False))

    if blocks:  # at least one box ticked
        cols = st.columns([5] * len(blocks))
        for col, (img_tag, caption, flip) in zip(cols, blocks):
            with col:
                img = get_image(tomo_name, img_tag)
                if flip: img = img.transpose(Image.FLIP_TOP_BOTTOM)
                st.image(img, caption=caption, use_container_width=True)


    if len(df) > 5:
        ranked_distance_series = rank_distance_series(tomo_name, rank_df)
        c1, c2 = st.columns([5, 5])
        with c1:
            st.markdown(
                f'<div style="text-align: center;margin-bottom: -15px; font-size: 14px;"><b>Most similar tomograms</b></div>',
                unsafe_allow_html=True)
            for j in range(3):
                t_name = ranked_distance_series.index[1 + j]
                t_link = f"/Browse_tomograms?tomo_id={t_name}"
                st.markdown(
                    f"<p style='text-align: center; margin-bottom: -20px;font-size: 12px;'><a href='{t_link}'>{t_name}</a></p>",
                    unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div style="text-align: center;margin-top: 5px; margin-bottom: -15px; font-size: 14px;"><b>Most dissimilar tomograms:</b></div>',
                unsafe_allow_html=True)
            for j in range(3):
                t_name = ranked_distance_series.index[-(j + 1)]
                t_link = f"/Browse_tomograms?tomo_id={t_name}"
                st.markdown(
                    f"<p style='text-align: center; margin-bottom: -20px;font-size: 12px;'><a href='{t_link}'>{t_name}</a></p>",
                    unsafe_allow_html=True)

    st.text("")
    ontologies = df.loc[tomo_name].sort_values(ascending=False).index.tolist()
    for o in project_configuration["macromolecules"] + project_configuration["soft_ignore_in_summary"]:
        if o in ontologies:
            ontologies.remove(o)
    for o in project_configuration["soft_ignore_in_summary"]:
        if o not in ontologies:
            ontologies.append(o)

    n_imgs_per_row = 5
    while ontologies != []:
        n_cols = min(len(ontologies), n_imgs_per_row)
        col_ontologies = ontologies[:n_cols]
        ontologies = ontologies[n_cols:]
        for o, c in zip(col_ontologies, st.columns(n_imgs_per_row)):
            with c:
                st.text(f"{o}")
                st.image(get_image(tomo_name, o, projection=True).transpose(Image.FLIP_TOP_BOTTOM), use_container_width=True)
                st.image(get_image(tomo_name, f"{o}_side", projection=True).transpose(Image.FLIP_TOP_BOTTOM), use_container_width=True)


