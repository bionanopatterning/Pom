import streamlit as st
import copy
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from Pom.app.util import load_data
from Pom.core.tools import get_feature_library
from matplotlib import colors

st.set_page_config(
    page_title="Results table",
    layout='wide'
)

df = load_data()

copy_df = copy.deepcopy(df)
copy_df = copy_df.reset_index()
copy_df.rename(columns={'tomogram': 'Tomogram'}, inplace=True)
copy_df = copy_df.round(3)

st.title("Dataset Summary")
st.markdown(f"The table below lists measurements of the fraction of a tomogram's volume occupied by each of the segmented features.")
st.markdown(f"Click a **header** to sort by that feature, or a **tomogram name** to inspect that volume.")

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
gb.configure_pagination(enabled=True, paginationPageSize=200)
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

# Apply feature colors to cells (lightened by averaging with white)
def lighten_color(hex_color):
    """Convert hex color to RGB, average with white, return as RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16) / 255.0, int(hex_color[2:4], 16) / 255.0, int(hex_color[4:6], 16) / 255.0
    # Average with white (1.0, 1.0, 1.0)
    r_light = (r + 1.0) / 2.0
    g_light = (g + 1.0) / 2.0
    b_light = (b + 1.0) / 2.0
    return (r_light, g_light, b_light)

feature_library = get_feature_library()
for col in numerical_columns:
    if col in feature_library and 'color' in feature_library[col]:
        hex_color = feature_library[col]['color']
        rgb_color = lighten_color(hex_color)
        gb.configure_column(
            col,
            cellStyle={'backgroundColor': colors.to_hex(rgb_color)}
        )

grid_options = gb.build()


# Use AgGrid to display the DataFrame
grid_response = AgGrid(
    filtered_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=2700,
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