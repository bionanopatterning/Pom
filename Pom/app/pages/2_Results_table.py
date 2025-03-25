import streamlit as st
import json
import os
import copy
from PIL import Image
import pandas as pd
import numpy as np
from Pom.core.config import parse_feature_library, FeatureLibraryFeature
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from matplotlib import colors

st.set_page_config(
    page_title="Results table",
    layout='wide'
)

with open("project_configuration.json", 'r') as f:
    project_configuration = json.load(f)


feature_library = parse_feature_library(os.path.join(os.path.expanduser("~"), ".Ais", "feature_library.txt"))
df = pd.read_excel(os.path.join(project_configuration["root"], "summary.xlsx"), index_col=0)
df = df.dropna(axis=0)


with open("project_configuration.json", 'r') as f:
    project_configuration = json.load(f)



def get_image(tomo, image):
    image_dir = image.split("_")[0]
    img_path = os.path.join(project_configuration["image_dir"], image_dir, f"{tomo}_{image}.png")
    if os.path.exists(img_path):
        return Image.open(img_path)
    else:
        return Image.fromarray(np.zeros((128, 128)), mode='L')

def recolor(color, style=0):
    c = color
    if style == 0:
        c =  np.array(color) / 2.0 + 0.5
    if style == 1:
        c = np.array(color) / 8 + 0.875
    c = np.clip(c, 0, 1)
    return c

copy_df = copy.deepcopy(df)
copy_df = copy_df.reset_index()
copy_df.rename(columns={'tomogram': 'Tomogram'}, inplace=True)
copy_df = copy_df.round(1)
columns = list(copy_df.columns)
for o in project_configuration["soft_ignore_in_summary"] + project_configuration["macromolecules"]:
    if o in columns:
        columns.remove(o)
        columns.append(o)
copy_df = copy_df[columns]


column_colors = dict()
for k in copy_df.columns:
    if k not in feature_library:
        feature_library[k] = FeatureLibraryFeature()
        feature_library[k].title = k
    column_colors[k] = recolor(feature_library[k].colour, 1)


st.title("Dataset summary")
n_ontologies = len(project_configuration["ontologies"]) + 1
n_macromolecules = len(project_configuration["macromolecules"])
if "Density" in project_configuration["macromolecules"]:
    n_macromolecules -= 1
st.markdown(f"The table below lists measurements of the fraction of a tomogram's volume occupied by each of **{n_ontologies} ontology** segmentations and **{n_macromolecules} macromolecule** segmentations.")
st.markdown(f"Click a **header** element to sort by that feature, or a **tomogram name** to inspect that volume.")

search_query = st.text_input("Search Tomogram by name")
if search_query:
    filtered_df = copy_df[copy_df['Tomogram'].str.contains(search_query, case=False, na=False)]
else:
    filtered_df = copy_df


numerical_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

# Create a layout with columns to make sliders more compact
with st.expander(label="Filters"):
    slider_filters = {}
    n_sliders_per_row = 6  # Control how many sliders per row
    cols = st.columns(n_sliders_per_row)

    for idx, col in enumerate(numerical_columns):
        min_val = float(copy_df[col].min())
        max_val = float(copy_df[col].max())
        if max_val <= min_val:
            max_val += 1.0
        col_idx = idx % n_sliders_per_row  # Choose column for slider
        with cols[col_idx]:  # Add slider in the appropriate column
            slider_filters[col] = st.slider(f"{col}", min_val, max_val, (min_val, max_val))

    # Apply the filters to the DataFrame
    for col, (min_val, max_val) in slider_filters.items():
        filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]

# Create AgGrid options
gb = GridOptionsBuilder.from_dataframe(filtered_df)
gb.configure_pagination(enabled=True, paginationPageSize=50)
gb.configure_selection('single', use_checkbox=False)  # Allow single row selection
gb.configure_column('Tomogram', header_name="Tomogram", pinned=True,
                    cellRenderer=JsCode(
                        """
                        class UrlCellRenderer {
                          init(params) {
                            this.eGui = document.createElement('a');
                            this.eGui.innerText = params.value;
                            this.eGui.setAttribute('href', '/Browse_tomograms?tomo_id=' + params.value);
                            this.eGui.setAttribute('style', "text-decoration:none");
                            this.eGui.setAttribute('target', "_blank");
                          }
                          getGui() {
                            return this.eGui;
                          }
                        }
                        """
                    ),
                    )

for k in column_colors:
    if k == "Tomogram":
        gb.configure_column(k, cellStyle={'backgroundColor': colors.to_hex((0.98, 0.98, 0.98))})
    else:
        gb.configure_column(k, cellStyle={'backgroundColor': colors.to_hex(column_colors[k])})

grid_options = gb.build()


# Use AgGrid to display the DataFrame
grid_response = AgGrid(
    filtered_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=900,
    theme="streamlit",
    allow_unsafe_jscode=True,  # Allow HTML rendering for clickable links
)


st.markdown(
    """
    <style>
    .ag-header {
        background-color: #f0f0f0 !important;
        color: black !important;
        font-size: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)