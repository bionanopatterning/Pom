import streamlit as st
import matplotlib.pyplot as plt
from Pom.app.util import *
from Pom.core.tools import get_feature_library
import numpy as np
from matplotlib.colors import to_rgb


st.set_page_config(
    page_title="Introduction",
    layout='wide'
)


df = load_data()

def feature_summary(df, feature):
    if feature not in df.columns:
        return
    feature_values = df[feature]
    best_n = 10
    st.header(f"{feature}")

    feature_in_top5 = df.apply(lambda row: feature in row.nlargest(5).index, axis=1).sum()
    st.markdown(f"**{feature}** is a top 5 component in **{feature_in_top5}** tomograms, with total volume equivalent to approximately **{df[feature].sum() / 100.0:.0f}** original tomogram volumes.")

    sorted_values = feature_values.sort_values(ascending=False)
    tomo_high = sorted_values.head(best_n).sample(n=1).index[0]
    tomo_low = sorted_values.tail(best_n).sample(n=1).index[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**High {feature} content:**")
        img_high = get_image(tomo_high, "density")
        if img_high:
            st.image(img_high, width="stretch")
        high_link = f"/Browse_tomograms?tomo_id={tomo_high}"
        st.markdown(f"[{tomo_high}]({high_link})")
    with c2:
        st.markdown(f"**Low {feature} content:**")
        img_low = get_image(tomo_low, "density")
        if img_low:
            st.image(img_low, width="stretch")
        low_link = f"/Browse_tomograms?tomo_id={tomo_low}"
        st.markdown(f"[{tomo_low}]({low_link})")

c1, c2, c3 = st.columns([0.2, 0.6, 0.2])
with c2:
    st.title("Dataset Overview")

    all_features = list(df.columns)
    n_features = len(all_features)
    n_tomograms = len(df)

    st.text(f"{n_tomograms} tomograms, {n_features} features")

    st.markdown("Dataset summary and visualizations generated using Pom.")

    st.text("")
    st.markdown(f'<div style="text-align: center;"><b>Feature composition across {n_tomograms} tomograms</b></div>', unsafe_allow_html=True)
    st.text("")

    pie_data = {feature: df[feature].sum() for feature in all_features}
    pie_data = dict(sorted(pie_data.items(), key=lambda x: x[1], reverse=True))

    # Get colors from feature library
    feature_library = get_feature_library()
    pie_colors = [feature_library.get(feature, {}).get('color', '#808080') for feature in pie_data.keys()]
    pie_colors = [np.array(to_rgb(c)) / 3 + 2 / 3 for c in pie_colors]

    fig, ax = plt.subplots()
    ax.pie(
        pie_data.values(),
        labels=pie_data.keys(),
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=0,
        textprops={'fontsize': 8}
    )
    ax.axis('equal')
    st.pyplot(fig)

    features = list(pie_data.keys())
    for feature in features:
        feature_summary(df, feature)

