import streamlit as st
import json
from Pom.core.render import parse_feature_library, FeatureLibraryFeature
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

with open("C:/Users/mgflast/PycharmProjects/ontoseg/project_configuration.json", 'r') as f:
    project_configuration = json.load(f)


def get_image(tomo, image):
    img_path = os.path.join(project_configuration["root"], project_configuration["image_dir"], image, f"{tomo}_bin2_{image}.png")
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


feature_library = parse_feature_library("C:/Users/mgflast/.Ais/feature_library.txt")
df = pd.read_excel(os.path.join(project_configuration["root"], "summary.xlsx"), index_col=0)
df = df.dropna(axis=0)

# Query params
tomo_name=df.index[0]
if "tomo_id" in st.query_params:
    tomo_name = st.query_params["tomo_id"]


st.markdown(f'<div style="text-align: center;font-size: 30px;"><b>{tomo_name}\n\n</b></div>', unsafe_allow_html=True)
st.text("")
_, column_base, _ = st.columns([3, 15, 3])
with column_base:
    c1, c2, c3 = st.columns([5, 5, 5])
    with c1:
        st.image(get_image(tomo_name, "density").transpose(Image.FLIP_TOP_BOTTOM), use_column_width=True,
                 caption="Density (central slice)")
    with c2:
        st.image(get_image(tomo_name, "Macromolecules"), use_column_width=True, caption="Macromolecules")
    with c3:
        st.image(get_image(tomo_name, "Top3"), use_column_width=True, caption="Top 3 ontologies")

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
                st.image(get_image(tomo_name, o).transpose(Image.FLIP_TOP_BOTTOM), use_column_width=True)
                st.image(get_image(tomo_name, f"{o}_side").transpose(Image.FLIP_TOP_BOTTOM), use_column_width=True)



# def tomo_pie_plot(tomo_name, df):
#     df = copy.deepcopy(df[df.index == tomo_name])
#     pie_data = dict()
#     for k in df.columns:
#         if k in project_configuration["ontologies"] + ["Unknown"]:
#             pie_data[k] = df[k].sum()
#     pie_data = sorted(pie_data.items(), key=lambda x: x[1], reverse=True)
#     rearranged_pie_data = []
#     for i in range(len(pie_data) // 2):
#         rearranged_pie_data.append(pie_data[i])
#         rearranged_pie_data.append(pie_data[-(i + 1)])
#     if len(pie_data) % 2 == 0:
#         rearranged_pie_data.append(pie_data[len(pie_data) // 2])
#     # Unpack the rearranged data into separate lists
#     pie_data = dict()
#     for l, v in rearranged_pie_data:
#         pie_data[l] = v
#
#     pie_colours = [recolor(feature_library[label].colour) for label in pie_data]
#
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(
#         pie_data.values(),  # Values for pie chart
#         labels=None,
#         colors=pie_colours,  # Custom colors for each slice
#         startangle=90,  # Start angle for better alignment
#         explode=[0.0] * len(pie_data),
#         textprops={'fontsize': 6}
#     )
#     # Equal aspect ratio ensures the pie is drawn as a circle.
#     ax.axis('equal')
#     return fig
