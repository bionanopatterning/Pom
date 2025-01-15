import streamlit as st
import copy
import pandas as pd
from Pom.core.config import *
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Introduction",
    layout='wide'
)

df = pd.read_excel(os.path.join(project_configuration["root"], "summary.xlsx"), index_col=0)
df = df.dropna(axis=0)


def ontology_summary(df, ontology):
    if not ontology in df:
        return
    o_values = df[ontology]
    best_n = 10
    st.header(f"{ontology}")
    # get highest and lowest valued images for this.
    o_in_top3 = df.apply(lambda row: ontology in row.nlargest(5).index, axis=1).sum()
    st.markdown(f"**{ontology.capitalize()}** is a top 5  component in **{o_in_top3}** tomograms, with the total {ontology.lower()} volume in the full dataset equivalent to approximately **{df[ontology].sum() / 100.0:.0f}** original tomogram volumes.")

    sorted_df = o_values.sort_values(ascending=False)
    tomo_top = sorted_df.head(best_n).sample(n=1).index[0]
    tomo_bot = sorted_df.tail(best_n).sample(n=1).index[0]
    c1, c2 = st.columns(2)
    with c1:
        top_caption_link = f"/Browse_tomograms?tomo_id={tomo_top}"
        st.image(get_image(tomo_top, "density"), caption=f"High {ontology} content:", use_column_width=True)
        st.markdown(f"<p style='text-align: center;margin-top: -20px;font-size: 13px;'><a href='{top_caption_link}'>{tomo_top}</a></p>", unsafe_allow_html=True)
    with c2:
        bot_caption_link = f"/Browse_tomograms?tomo_id={tomo_bot}"
        st.image(get_image(tomo_bot, "density"), caption=f"Low {ontology} content:", use_column_width=True)
        st.markdown(f"<p style='text-align: center;margin-top: -20px;font-size: 13px;'><a href='{bot_caption_link}'>{tomo_bot}</a></p>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([0.2, 0.6, 0.2])
with c2:
    st.title("Introduction")
    macromolecules = project_configuration["macromolecules"]
    if "Density" in macromolecules:
        macromolecules.remove("Density")

    ontologies = [o for o in project_configuration["ontologies"] if o != "_"]
    n_volumes = len(df) * (len(macromolecules) + len(ontologies) + 2)

    st.text(f"{len(df)} tomograms, {len(macromolecules)} macromolecules, {len(ontologies) + 1} organelles, {n_volumes} volumes.")

    st.markdown(f"All segmentations, visualizations, and other data shared on these pages, including this summary report, were generated using Pom - a _work-in-progress_ Python cli module for making sense of large datasets.")

    st.text("")
    st.markdown(f'<div style="text-align: center;"><b>At a glance: ensemble composition of {len(df)} tomograms</b></div>', unsafe_allow_html=True)
    st.text("")
    pie_data = dict()
    for k in df.columns:
        if k in project_configuration["ontologies"] + ["Unknown"] and k not in project_configuration["soft_ignore_in_summary"]:
            pie_data[k] = df[k].sum()
    pie_data = sorted(pie_data.items(), key= lambda x: x[1], reverse=True)

    rearranged_pie_data = []
    for i in range(len(pie_data) // 2):
        rearranged_pie_data.append(pie_data[i])
        rearranged_pie_data.append(pie_data[-(i + 1)])
    if len(pie_data) % 2 == 1:
        rearranged_pie_data.append(pie_data[len(pie_data) // 2])
    # Unpack the rearranged data into separate lists
    pie_data = dict()
    for l, v in rearranged_pie_data:
        pie_data[l] = v


    pie_colours = [recolor(feature_library[label].colour) for label in pie_data]
    fig, ax = plt.subplots()
    ax.pie(
        pie_data.values(),  # Values for pie chart
        labels=pie_data.keys(),  # Labels for pie chart
        colors=pie_colours,  # Custom colors for each slice
        autopct='%1.1f%%',  # Show percentage
        startangle=0,  # Start angle for better alignment
        explode=[0.0] * len(pie_data),
        textprops={'fontsize': 6}
    )
    # Equal aspect ratio ensures the pie is drawn as a circle.
    ax.axis('equal')

    # Display the pie chart in Streamlit
    st.pyplot(fig)

    ontologies = [o for o in project_configuration["ontologies"] if o != "_"]
    if "Void" in ontologies:
        ontologies.remove("Void")
        ontologies.append("Void")

    ontologies.append("Unknown")
    for o in ontologies:
        ontology_summary(df, o)

