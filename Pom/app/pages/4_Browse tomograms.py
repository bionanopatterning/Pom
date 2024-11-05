import streamlit as st
import json
from Pom.core.config import parse_feature_library, FeatureLibraryFeature
import pandas as pd
import os
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Tomogram details",
    layout='wide'
)

with open("project_configuration.json", 'r') as f:
    project_configuration = json.load(f)


def get_image(tomo, image, projection=False):
    image_dir = image.split("_")[0]
    if projection:
        image_dir += "_projection"
    img_path = os.path.join(project_configuration["image_dir"], image_dir, f"{tomo}_{image}.png")
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


feature_library = parse_feature_library("feature_library.txt")

@st.cache_data
def load_data():
    cache_df = pd.read_excel(os.path.join(project_configuration["root"], "summary.xlsx"), index_col=0)
    cache_df = cache_df.dropna(axis=0)
    to_drop = list()
    for f in project_configuration["macromolecules"] + ["Thickness (nm)", "Thickness error (nm)"]:
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

# Query params
tomo_name = df.index[0]
if "tomo_id" in st.query_params:
    tomo_name = st.query_params["tomo_id"]


tomo_names = df.index.tolist()
_, column_base, _ = st.columns([3, 15, 3])
with column_base:
    c1, c2, c3 = st.columns([1, 19, 1])
    with c1:
        if st.button("<"):
            idx = tomo_names.index(tomo_name)
            idx = (idx - 1) % len(tomo_names)
            tomo_name = tomo_names[idx]
            st.query_params["tomo_id"] = tomo_name
    with c2:
        tomo_title_field = st.empty()
    with c3:
        if st.button("\>"):
            idx = tomo_names.index(tomo_name)
            idx = (idx + 1) % len(tomo_names)
            tomo_name = tomo_names[idx]
            st.query_params["tomo_id"] = tomo_name

    tomo_title_field = st.markdown(f'<div style="text-align: center;font-size: 30px;"><b>{tomo_name}\n\n</b></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([5, 5, 5])
    with c1:
        st.image(get_image(tomo_name, "density").transpose(Image.FLIP_TOP_BOTTOM), use_column_width=True,
                 caption="Density (central slice)")
    with c2:
        st.image(get_image(tomo_name, "Macromolecules"), use_column_width=True, caption="Macromolecules")
    with c3:
        st.image(get_image(tomo_name, "Top3"), use_column_width=True, caption="Top 3 ontologies")

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
                st.image(get_image(tomo_name, o, projection=True).transpose(Image.FLIP_TOP_BOTTOM), use_column_width=True)
                st.image(get_image(tomo_name, f"{o}_side", projection=True).transpose(Image.FLIP_TOP_BOTTOM), use_column_width=True)


