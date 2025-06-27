import streamlit as st
import json
import pandas as pd
import os
from PIL import Image
import numpy as np
import glob

# ----------------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------------

st.set_page_config(
    page_title="Tomogram Gallery",
    layout="wide",
)

# ----------------------------------------------------------------------------------
# Load configuration & helpers
# ----------------------------------------------------------------------------------

with open("project_configuration.json", "r") as f:
    project_configuration = json.load(f)


def get_image(tomo: str, image: str):
    """Return a PIL image for a given tomogram name and display option."""
    image_tag = image.split(" projection")[0]
    image_dir = image_tag if "projection" not in image else f"{image_tag}_projection"
    img_path = os.path.join(
        project_configuration["image_dir"], image_dir, f"{tomo}_{image_tag}.png"
    )

    if os.path.exists(img_path):
        out_img = Image.open(img_path)
        if "density" in image or "projection" in image:
            out_img = out_img.transpose(Image.FLIP_TOP_BOTTOM)
        return out_img

    # fallback: return an empty placeholder
    return Image.fromarray(np.zeros((128, 128), dtype=np.uint8), mode="L")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache summary.xlsx as a DataFrame."""
    cache_df = pd.read_excel(
        os.path.join(project_configuration["root"], "summary.xlsx"), index_col=0
    )
    return cache_df.dropna(axis=0)


# ----------------------------------------------------------------------------------
# Load data & prepare session state
# ----------------------------------------------------------------------------------

df = load_data()

tomo_subsets = [
    os.path.splitext(os.path.basename(j))[0]
    for j in glob.glob(os.path.join(project_configuration["root"], "subsets", "*.json"))
]

# Initialise Streamlit session state defaults
st.session_state.setdefault("page_num", 0)
st.session_state.setdefault("search_query", "")
st.session_state.setdefault("display_option", "density")
st.session_state.setdefault("n_cols", 5)
st.session_state.setdefault("subset", "all")
# NEW: sorting controls
st.session_state.setdefault("sort_column", "None")
st.session_state.setdefault("sort_ascending", False)


def reset_page_number():
    """Utility to jump back to the first page when filters change."""
    st.session_state.page_num = 0


# ----------------------------------------------------------------------------------
# Sidebar / toolbar controls
# ----------------------------------------------------------------------------------

st.title("Tomogram Gallery")
st.write(
    "Browse through the collection of tomograms. Use the controls below to search, filter, and sort the gallery."
)

controls = st.columns([3.5, 1, 1.2, 1.2, 0.8, 0.8], vertical_alignment="bottom")

# Search
with controls[0]:
    st.text_input(
        "Search tomograms",
        value=st.session_state.search_query,
        key="search_query",
        on_change=reset_page_number,
    )

# Subset selector
with controls[1]:
    st.selectbox(
        "Subset",
        ["all"] + tomo_subsets,
        key="subset",
        on_change=reset_page_number,
    )

# Display option (density / projections)
with controls[2]:
    options = project_configuration["gallery_categories"] + [
        f"{o} projection" for o in project_configuration["ontologies"] + ["Unknown"]
    ]
    st.selectbox("Display option", options, key="display_option")

# Sort column selector (NEW)
with controls[3]:
    st.selectbox(
        "Sort by",
        ["None"] + list(df.columns),
        key="sort_column",
        on_change=reset_page_number,
    )

# Ascending toggle (bottom aligned)
with controls[4]:
    st.markdown("<div style='margin-top: 24px'>", unsafe_allow_html=True)
    st.toggle(
        "ascending",
        key="sort_ascending",
        on_change=reset_page_number,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Columns selector (same row)
with controls[5]:
    st.number_input(
        "Columns",
        min_value=1,
        max_value=1000,
        step=1,
        key="n_cols",
    )

# ----------------------------------------------------------------------------------
# Filter & sort tomogram list
# ----------------------------------------------------------------------------------

tomogram_names = df.index.tolist()

# Apply search filter
if st.session_state.search_query:
    tomogram_names = [
        name
        for name in tomogram_names
        if st.session_state.search_query.lower() in name.lower()
    ]

# Apply subset filter
if st.session_state.subset != "all":
    subset_json = os.path.join(
        project_configuration["root"], "subsets", f"{st.session_state.subset}.json"
    )
    with open(subset_json, "r") as f:
        subset_tomos = json.load(f)["tomos"]
    tomogram_names = [name for name in tomogram_names if name in subset_tomos]

# Apply sorting (NEW)
if (
    st.session_state.sort_column != "None"
    and st.session_state.sort_column in df.columns
    and len(tomogram_names) > 0
):
    tomogram_names = (
        df.loc[tomogram_names]
        .sort_values(
            by=st.session_state.sort_column,
            ascending=st.session_state.sort_ascending,
            kind="mergesort",  # stable sort preserves existing order when equal
        )
        .index.tolist()
    )

# ----------------------------------------------------------------------------------
# Gallery rendering & pagination
# ----------------------------------------------------------------------------------

if not tomogram_names:
    st.info("No tomograms found matching the current filters.")
    st.stop()

# Pagination calculations
n_cols = st.session_state.n_cols
n_rows = 4
per_page = n_cols * n_rows
total_pages = (len(tomogram_names) - 1) // per_page + 1

# Keep page_num within bounds
total_pages = max(total_pages, 1)
st.session_state.page_num = min(st.session_state.page_num, total_pages - 1)
st.session_state.page_num = max(st.session_state.page_num, 0)

# Determine slice of tomograms to display
start = st.session_state.page_num * per_page
end = start + per_page
tomograms_page = tomogram_names[start:end]

# Display images grid
for idx in range(0, len(tomograms_page), n_cols):
    row_tomos = tomograms_page[idx : idx + n_cols]
    cols = st.columns(n_cols)
    for col, tomo_name in zip(cols, row_tomos):
        with col:
            # Clickable tomogram name
            st.markdown(
                f"<div style='text-align: center; font-size:14px; margin-top:5px;'>"
                f"<a href='/Browse_tomograms?tomo_id={tomo_name}' style='text-decoration: none; color: inherit;'>"
                f"{tomo_name}</a></div>",
                unsafe_allow_html=True,
            )
            # Image
            st.image(get_image(tomo_name, st.session_state.display_option), use_container_width=True)

# ----------------------------------------------------------------------------------
# Pagination buttons
# ----------------------------------------------------------------------------------

def first_page():
    st.session_state.page_num = 0

def prev_page():
    if st.session_state.page_num > 0:
        st.session_state.page_num -= 1

def next_page():
    if st.session_state.page_num < total_pages - 1:
        st.session_state.page_num += 1

def last_page():
    st.session_state.page_num = total_pages - 1

pag_cols = st.columns([9, 1, 1, 3, 1, 1, 9])

pag_cols[1].button(":material/First_Page:", on_click=first_page, type="primary")
pag_cols[2].button(":material/Keyboard_Arrow_Left:", on_click=prev_page, type="primary")
pag_cols[3].markdown(
    f"<div style='text-align: center;'>Page {st.session_state.page_num + 1} of {total_pages}</div>",
    unsafe_allow_html=True,
)
pag_cols[4].button(":material/Keyboard_Arrow_Right:", on_click=next_page, type="primary")
pag_cols[5].button(":material/Last_Page:", on_click=last_page, type="primary")