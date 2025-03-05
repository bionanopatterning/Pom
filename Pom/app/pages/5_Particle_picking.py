import streamlit as st
import json
import os
import glob
import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
import mrcfile
from Pom.core.config import parse_feature_library, FeatureLibraryFeature
from scipy.ndimage import gaussian_filter
CONTEXT_BAR_HEIGHT = 5

st.set_page_config(
    page_title="Particle picking",
    layout='wide'
)

""
st.markdown("""
<style>
div[data-testid="stButtonGroup"] button {
    width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)


with open("project_configuration.json", 'r') as f:
    project_configuration = json.load(f)

def save_job(job_config):
    # save required files
    job_path = os.path.join(project_configuration["root"], "capp", job_config["job_name"], "config.json")
    os.makedirs(os.path.dirname(job_path), exist_ok=True)
    with open(job_path, 'w') as json_file:
        json.dump(job_config, json_file, indent=2)

    st.query_params["job_name"] = job_config["job_name"]
    time.sleep(1)
    st.rerun()

def new_job():
    job_config = dict()
    c1, c2 = st.columns([2, 3])

    c1.subheader("Base settings")
    job_config["job_name"] = c1.text_input("Job name", value="New CAPP job")
    available_targets = set([os.path.splitext(os.path.basename(f))[0].split("__")[1] for f in glob.glob(os.path.join(project_configuration["root"], project_configuration["macromolecule_dir"], f"*__*.mrc"))])

    job_config["target"] = c1.selectbox("Target", options=available_targets)

    with c2:
        st.subheader("Context settings")
        job_config["context_elements"] = st.multiselect("Features to include", project_configuration["ontologies"] + ["Unknown"], project_configuration["ontologies"] + ["Unknown"])

    st.divider()

    if st.columns([3, 1, 3])[1].button("Save job", use_container_width=True, type="primary"):
        save_job(job_config)


def crop_from_tomo(particle_data, s, feature_library):
    x = particle_data['X']
    y = particle_data['Y']
    z = particle_data['Z']
    tomo = particle_data['tomo']
    tomo_path = os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"{tomo}.mrc")

    image = copy(mrcfile.mmap(tomo_path).data[z, :, :])
    J, K = image.shape
    j = y
    k = x


    _j = min(j + s//2, J)
    _j = max(_j, s)
    _k = min(k + s//2, K)
    _k = max(_k, s)

    image = image[_j-s:_j, _k-s:_k]
    image -= np.mean(image)
    image /= np.std(image)
    image = (image + 2.0) / 4.0
    image = gaussian_filter(image, 0.8)

    context_elements = [e for e in list(particle_data.index) if e not in ['X', 'Y', 'Z', 'tomo']]
    context_img = np.zeros((CONTEXT_BAR_HEIGHT, s, 3))
    j = 0
    for c in context_elements:
        c_val = particle_data[c]
        context_img[:, int(j * s):(int((j + c_val) * s)), :] = feature_library[c].colour
        j += c_val
    combined_img = np.zeros((s + CONTEXT_BAR_HEIGHT, s, 3))
    combined_img[:s, :, 0] = image[::-1]
    combined_img[:s, :, 1] = image[::-1]
    combined_img[:s, :, 2] = image[::-1]
    combined_img[s:, :, :] = context_img
    combined_img = np.clip(combined_img, 0.0, 1.0)
    fig, ax = plt.subplots()
    _x = x - (_k - s)
    _y = y - (_j - s)
    ax.imshow(combined_img, cmap='gray')
    ax.axis('off')
    ax.scatter(_x, s - _y, marker='o', s=4000, facecolors='none', edgecolors='red', linewidth=2)
    plt.close(fig)
    return fig


def open_in_ais(tomo_name, slice=128):
    cmd_path = os.path.join(os.path.expanduser("~"), ".Ais", "pom_to_ais.cmd")
    tomo_dir = os.path.join(project_configuration["root"], project_configuration["tomogram_dir"])
    with open(cmd_path, 'a') as f:
        base = os.path.abspath(os.path.join(tomo_dir, f"{tomo_name}"))
        if os.path.exists(base+".scns"):
            f.write(f"open\t{base+'.scns'}\tslice\t{slice}\n")
        else:
            f.write(f"open\t{base + '.mrc'}\tslice\t{slice}\n")


def view_particles(job_name):
    N_ROWS = 5
    N_COLS = 6
    IMG_SIZE = 128

    # load particle data.
    data_path = os.path.join(project_configuration["root"], "capp", job_name, "all_particles.tsv")
    if not os.path.exists(data_path):
        st.markdown("<p style='text-align: center; font-size: 20px;'>No particles found (yet).</p>",
                        unsafe_allow_html=True)
        return

    with open(os.path.join(project_configuration["root"], "capp", job_name, f"config.json"), 'r') as f:
        job_config = json.load(f)

    feature_library = parse_feature_library(os.path.join(os.path.expanduser("~"), ".Ais", "feature_library.txt"))
    if "Unknown" not in feature_library:
        feature_library["Unknown"] = FeatureLibraryFeature()
        feature_library["Unknown"].colour = (0.6, 0.3, 0.6)
    for f in job_config["context_elements"]:
        if f not in feature_library:
            feature_library[f] = FeatureLibraryFeature()

    all_particles = pd.read_table(data_path, header=0)

    if "particle_page" not in st.session_state:
        st.session_state.particle_page = 1

    n_pages = int(np.ceil(len(all_particles) / (N_COLS * N_ROWS)))
    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")

    with c1:
        filter_value = st.text_input("Filter tomograms", value="")
        all_particles = all_particles[all_particles.iloc[:, -1].str.contains(filter_value, na=False)]

    with c2:
        _c = st.columns([1, 1, 4, 1, 1])
        if _c[0].button(":material/First_Page:", type="primary"):
            st.session_state.particle_page = 1
        if _c[1].button(":material/Keyboard_Arrow_Left:", type="primary"):
            st.session_state.particle_page = max(st.session_state.particle_page - 1, 1)
        if _c[3].button(":material/Keyboard_Arrow_Right:", type="primary"):
            st.session_state.particle_page = min(st.session_state.particle_page + 1, n_pages)
        if _c[4].button(":material/Last_Page:", type="primary"):
            st.session_state.particle_page = n_pages
        _c[2].markdown(
            f"<p style='text-align: center; padding: 3px 0;font-size: 20px;'>Page {st.session_state.particle_page} of {n_pages}</p>",
            unsafe_allow_html=True
        )

    with c3:

        _c = st.columns([3, 1], vertical_alignment='bottom')

        with _c[0]:
            sort_options = ["Tomo name"] + [e for e in list(all_particles.columns.values) if e in job_config["context_elements"]]
            sort_option = st.selectbox("Sorting options", options=sort_options, label_visibility="collapsed")

        with _c[1]:
            sort_order_options = [":material/Arrow_Upward:", ":material/Arrow_Downward:"]
            sort_order = st.pills("sort_df", label_visibility='hidden', options=sort_order_options, selection_mode='single', default=sort_order_options[0])

        if sort_option == sort_options[0]:
            all_particles = all_particles.sort_values(by=all_particles.columns[-1], ascending=sort_order==sort_order_options[0])
        else:
            all_particles = all_particles.sort_values(by=sort_option, ascending=sort_order==sort_order_options[1])


    def display_particle(particle_data, uid):
        img = crop_from_tomo(particle_data, IMG_SIZE, feature_library)
        tomo = particle_data[-1]
        st.pyplot(img, use_container_width=True)

        if st.button("open in Ais", use_container_width=True, key=f"{uid}btn"):
            open_in_ais(tomo, slice=particle_data['Z'])

        st.markdown(
            f"<p style='text-align: center; font-size: 12px; margin: 0px;'>{tomo}</p>",
            unsafe_allow_html=True
        )


        " "


    idx_offset = N_COLS * N_ROWS * (st.session_state.particle_page - 1)
    for j in range(min(N_ROWS, int(np.ceil((len(all_particles) - idx_offset) / N_COLS)))):
        for k, c in enumerate(st.columns(N_COLS, border=False)):
            idx = idx_offset + j * N_COLS + k
            if idx < len(all_particles):
                with c:
                    particle_data = all_particles.iloc[idx]
                    display_particle(particle_data, uid=idx)




def view_job(job_name):
    with open(os.path.join(project_configuration["root"], "capp", job_name, f"config.json"), 'r') as f:
        job_config = json.load(f)

    with st.expander(f"Processing instructions", expanded=True):
        st.markdown(f"**Pick particles with Ais:**")
        # c1, c2 = st.columns([3, 2], vertical_alignment='bottom')
        # with c1:
        c = st.columns([3, 1, 1, 1, 1])
        threshold = c[1].number_input("Threshold", min_value=1, max_value=255, step=1, value=128)
        spacing = c[2].number_input("Spacing (px)", min_value=1, step=1, value=10)
        size = c[3].number_input("Volume (voxels)", min_value=1, step=1, value=100)
        margin = c[4].number_input("Margin (px)", min_value=1, step=1, value=16)

        n_proc_use = multiprocessing.cpu_count() // 2
        st.code(f"ais pick -d {os.path.join(project_configuration['root'], project_configuration['macromolecule_dir'])} -t {job_config['target']} -m {margin} -size-px {size} -spacing-px {spacing} -threshold {threshold} -p {n_proc_use} -ou {os.path.join(project_configuration['root'], 'capp', job_name, 'coordinates')}")

        st.markdown(f"**Sample context with Pom:**")
        c = st.columns([6, 1])
        window_size = c[1].number_input("Window size (px)", min_value=1, step=1, max_value=margin*2, value=margin*2)
        st.code(f"pom capp -c {job_name} -w {window_size} -p {n_proc_use}")

    with st.expander(f"Detected particles", expanded=True):
        view_particles(job_name)

available_jobs = [os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(project_configuration["root"], "capp", "*", "config.json"))]
available_jobs = ["Create new job"] + available_jobs

if "selected_job" not in st.session_state:
    st.session_state.selected_job = "Create new job"
    selected_job = st.session_state.selected_job
else:
    selected_job = available_jobs[0]

def redirect():
    redirect_target = st.session_state.selected_job
    time.sleep(0.5)
    st.query_params["job_name"] = redirect_target

if "job_name" in st.query_params:
    selected_job = st.query_params["job_name"]

selected_job_idx = 0
if selected_job in available_jobs:
    selected_job_idx = available_jobs.index(selected_job)

c1, c2, c3 = st.columns([5, 1, 2])
with c1:
    st.header("Context-aware particle picking")
with c3:
    selected_job = st.selectbox("Select job", options=available_jobs, key="selected_job", on_change=redirect())
st.divider()

if selected_job == "Create new job":
    new_job()
else:
    view_job(selected_job)