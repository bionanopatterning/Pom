import streamlit as st
import os
import glob
import json
from Pom.app.util import load_data, get_image

st.set_page_config(
    page_title="Tomogram Gallery",
    layout="wide",
)

df = load_data()

tomo_subsets = [os.path.splitext(os.path.basename(j))[0] for j in glob.glob(os.path.join("pom", "subsets", "*.txt"))]

# Load compositions
compositions_path = os.path.join('pom', 'image_compositions.json')
if os.path.exists(compositions_path):
    with open(compositions_path, 'r') as f:
        compositions = list(json.load(f).keys())
else:
    compositions = []

st.session_state.setdefault("page_num", 0)
st.session_state.setdefault("search_query", "")
st.session_state.setdefault("display_option", "density")
st.session_state.setdefault("n_cols", 5)
st.session_state.setdefault("subset", "all")
st.session_state.setdefault("sort_column", "None")
st.session_state.setdefault("sort_ascending", False)


def reset_page_number():
    """Utility to jump back to the first page when filters change."""
    st.session_state.page_num = 0

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

# Display option
with controls[2]:
    options = ['density'] + compositions + [f'{o}_projection' for o in list(df.columns)]
    st.selectbox("Display option", options, key="display_option")

# Sort column selector (NEW)
with controls[3]:
    st.selectbox(
        "Sort by",
        ["None"] + list(df.columns),
        key="sort_column",
        on_change=reset_page_number,
    )

# Ascending toggle
with controls[4]:
    st.toggle(
        "ascending",
        key="sort_ascending",
        on_change=reset_page_number,
    )

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
    subset_txt = os.path.join("pom", "subsets", f"{st.session_state.subset}.txt")
    with open(subset_txt, "r") as f:
        subset_tomos = [line.strip() for line in f if line.strip()]
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
            st.image(get_image(tomo_name, st.session_state.display_option), width="stretch")

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