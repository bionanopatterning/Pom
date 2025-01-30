import Pommie.typedefs
import streamlit as st
import numpy as np
import json
import glob
import os
import Pommie
import random
import uuid
import mrcfile
from copy import copy
import time

st.set_page_config(
    page_title="Template matching",
    layout='wide'
)



st.markdown(
    """
    <style>
    [data-testid="stSliderTickBarMax"] {
        display: none !important; /* Completely hide the element */
        visibility: hidden !important; /* Alternative: Make it invisible but keep its space */
    }
    
    [data-testid="stSliderTickBarMin"] {
        display: none !important; /* Completely hide the element */
        visibility: hidden !important; /* Alternative: Make it invisible but keep its space */
    }
    
    [data-testid="stSliderTickBar"] {
        height: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Target segmented-control buttons by attribute */
    button[kind="segmented_control"] {
        height: 40px !important;       /* or any desired height */
        line-height: 40px !important;  /* line-height often needs matching */
    }
    </style>
    """,
    unsafe_allow_html=True
)

TEMPLATE_PREVIEW_COLS = 16
TEMPLATE_PREVIEW_ROWS = 2

with open("project_configuration.json", 'r') as f:
    project_configuration = json.load(f)


class AreaFilter:
    def __init__(self):
        self.id = uuid.uuid4()
        self.o = "..."
        self.threshold = 0.5
        self.edge = False
        self.edge_in = 10.0
        self.edge_out = 0.0
        self.logic = "include"
        self.mask = None
        self.active = True

    def __eq__(self, other):
        if isinstance(other, AreaFilter):
            return self.id == other.id
        return False

    def to_dict(self):
        filter_dict = {}
        filter_dict["feature"] = self.o
        filter_dict["threshold"] = self.threshold
        filter_dict["edge"] = self.edge
        filter_dict["edge_in"] = self.edge_in
        filter_dict["edge_out"] = self.edge_out
        filter_dict["logic"] = self.logic
        return filter_dict


def generate_template_previews(job_config, n_samples=18):
    Pommie.compute.initialize()

    template = Pommie.typedefs.Particle.from_path(job_config["template_path"])
    template = template.bin(job_config["template_binning"])
    template.data -= np.mean(template.data)
    template.data /= np.std(template.data)
    template.data = (template.data + 2.0) / 4.0
    template = Pommie.compute.gaussian_filter([template], sigma=job_config["template_blur"])[0]
    spherical_mask = Pommie.typedefs.Mask(template)
    spherical_mask.spherical(radius_px=spherical_mask.n//2)
    template.data *= spherical_mask.data


    template_mask = Pommie.typedefs.Particle.from_path(job_config["template_mask_path"])
    template_mask = template_mask.bin(job_config["template_binning"])
    template_mask.data *= spherical_mask.data

    polar_min_rad = (job_config["transform_polar_min"]) * np.pi / 180.0
    polar_max_rad = (job_config["transform_polar_max"]) * np.pi / 180.0
    transforms = Pommie.typedefs.Transform.sample_unit_sphere(n_samples=job_config["transform_n"],
                                                              polar_lims=(polar_min_rad, polar_max_rad))
    transforms = random.sample(transforms, min(len(transforms), n_samples))
    templates = template.resample(transforms)
    template_masks = template_mask.resample(transforms)
    n = template.n

    image_pairs = list()
    for j in range(len(transforms)):
        template_j_2d = templates[j].data[n//2, :, :]
        template_mask_j_2d = template_masks[j].data[n//2, :, :]
        image_pairs.append((template_j_2d, template_mask_j_2d))
    return image_pairs


def generate_mask_preview(job_config, tomo_idx):
    # grab a random tomo central slice
    tomos = glob.glob(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], "*.mrc"))
    tomo = tomos[tomo_idx]
    print(tomo)
    tomo_name = os.path.splitext(os.path.basename(tomo))[0]

    slice = mrcfile.mmap(tomo).data
    slice_pxd = copy(slice[slice.shape[0]//2, :, :])
    slice_pxd -= np.mean(slice_pxd)
    slice_pxd /= np.std(slice_pxd)
    slice_pxd = (slice_pxd + 2.0) / 4.0

    stride = job_config["stride"]
    available_features = project_configuration["ontologies"] + ["Unknown"]
    for f in st.session_state.area_filters:
        if f.o not in available_features:
            continue
        if not f.active:
            continue

        segmentation_path = os.path.join(project_configuration["root"], project_configuration["output_dir"], f"{tomo_name}__{f.o}.mrc")

        vol = mrcfile.mmap(segmentation_path)
        pxs = vol.voxel_size.x / 10.0
        img = vol.data[vol.data.shape[0]//2, :, :][np.newaxis, :, :]
        img = Pommie.typedefs.Volume.from_array(img)
        print()
        if not f.edge:
            img.threshold(f.threshold)
        else:
            img.to_shell_mask(f.threshold, int(f.edge_out / pxs), int(f.edge_in / pxs))

        img.unbin(2)  # mind: with np.newaxis axis0 this results in shape (2, n, n)
        f.mask = img.data[0, :, :].astype(np.uint8)

    mask = np.zeros_like(slice_pxd)
    for f in st.session_state.area_filters:
        if not f.active:
            continue
        if f.logic == "include":
            print("include")
            mask = np.logical_or(mask, f.mask)
    for f in st.session_state.area_filters:
        if not f.active:
            continue
        if f.logic == "exclude":
            print("exclude")
            mask = np.logical_and(mask, np.logical_not(f.mask))

    mask = mask * 255
    for j in range(stride-1):
        mask[1+j::stride, :] = 0
        mask[:, 1+j::stride] = 0
    return slice_pxd, mask


def save_job(job_config):
    # write area selection setup to job config
    job_config["selection_criteria"] = list()
    for f in st.session_state.area_filters:
        if f.active:
            job_config["selection_criteria"].append(f.to_dict())

    # save required files
    job_path = os.path.join(project_configuration["root"], "astm", job_config["job_name"], "config.json")
    os.makedirs(os.path.dirname(job_path), exist_ok=True)
    with open(job_path, 'w') as json_file:
        json.dump(job_config, json_file, indent=2)

    st.query_params["job_name"] = job_config["job_name"]
    time.sleep(1)
    st.rerun()

def new_job():
    job_config = dict()
    c1, c2, c3 = st.columns([4, 4, 4])

    c1.subheader("Base settings")
    job_config["job_name"] = c1.text_input("Job name", value="...")
    job_config["stride"] = c1.number_input("Stride", value=1, min_value=1)

    c2.subheader("Transform")
    job_config["transform_n"] = c2.number_input("Number of transforms", value=500, min_value=1, max_value=500)
    job_config["transform_polar_min"] = c2.number_input("Polar angle start", value=-90, min_value=-90, max_value=90)
    job_config["transform_polar_max"] = c2.number_input("Polar angle stop", value=90, min_value=-90, max_value=90)

    c3.subheader("Template")
    job_config["template_path"] = c3.text_input("Template path")
    job_config["template_mask_path"] = c3.text_input("Template mask path")
    job_config["template_binning"] = c3.number_input("Bin factor", value=1, min_value=1)
    job_config["template_blur"] = c3.number_input("Template blur (Å)", value=20.0, step=1.0)

    if c3.columns([2, 3 * 0.75, 2])[1].button("Preview templates", use_container_width=True, type="primary"):
        with st.expander("Template previews (random subset)", expanded=True):
            with st.spinner("Generating previews..."):
                previews = generate_template_previews(job_config, n_samples=TEMPLATE_PREVIEW_ROWS * TEMPLATE_PREVIEW_COLS)

            i = 0
            for k in range(TEMPLATE_PREVIEW_ROWS):
                for j, c in enumerate(st.columns(TEMPLATE_PREVIEW_COLS)):
                    if i >= len(previews):
                        break
                    with c:
                        img_pair = previews[i]
                        st.image(img_pair[0], clamp=True, use_container_width=True)
                        st.image(img_pair[1], clamp=True, use_container_width=True)

                    i+=1
                if k < TEMPLATE_PREVIEW_ROWS - 1:
                    st.divider()
    else:
        st.divider()

    if "area_filters" not in st.session_state:
        st.session_state.area_filters = []

    st.subheader("Area selection")
    for f in st.session_state.area_filters:
        c0, c0b, c1, c2, c3, c4, c5, c6 = st.columns([0.2, 0.8, 1.4, 1.3, 1.0, 3.0, 0.25, 0.25], vertical_alignment="bottom")
        with c0b:
            f.active = st.toggle("Active", value=True, key=f"{f.id}active")
        with c1:
            f.o = st.selectbox("Feature", options=project_configuration["ontologies"] + ["Unknown"], key=f"{f.id}selectbox")
        with c2:
            #f.threshold = st.number_input("Threshold", value=0.5, min_value=0.0, max_value=1.0, key=f"{f.id}threshold", format="%0.5f")
            f.threshold = st.slider("Threshold", value=0.5, min_value=0.0, max_value=1.0, key=f"{f.id}threshold")
        with c3:
            f.logic = st.segmented_control("Inclusion mode\n", options=["include", "exclude"], default="include", key=f"{f.id}inclusion")
        with c4:
            _c0, _c1, _c2, _c3 = st.columns([0.25, 1, 2, 2], vertical_alignment="bottom")
            f.edge = _c1.toggle("Edge", value=False, key=f"{f.id}edge")
            f.edge_in = _c2.number_input("Inside (nm)", value=10.0, step=5.0, key=f"{f.id}edge_in", disabled=not f.edge)
            f.edge_out = _c3.number_input("Outside (nm)", value=10.0, step=5.0, key=f"{f.id}edge_out", disabled=not f.edge)
        with c6:
            if st.button(":material/Close:", key=f"{f.id}close", type="tertiary"):
                st.session_state.area_filters.remove(f)
                st.rerun()
    "\n"
    if st.columns([3, 1.0, 3])[1].button("Add criterion", use_container_width=True, type="secondary"):
        st.session_state.area_filters.append(AreaFilter())
        st.rerun()

    "\n"

    if len(st.session_state.area_filters) > 0:
        with st.columns([1, 4, 1])[1]:
            with st.expander("Slice & mask preview", expanded=True):
                n_tomos = len(glob.glob(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"*.mrc")))
                if "tomo_idx" not in st.session_state:
                    st.session_state.tomo_idx = np.random.randint(0, n_tomos - 1, 1)[0]
                tomo_idx = st.slider("Test tomogram index", min_value=0, max_value=n_tomos - 1, value=st.session_state.tomo_idx, step=1)
                if tomo_idx != st.session_state.tomo_idx:
                    st.session_state.tomo_idx = tomo_idx
                preview_slice, preview_mask = generate_mask_preview(job_config, tomo_idx)
                c1, c2 = st.columns([2, 2])
                with c1:
                    st.image(preview_slice, clamp=True, use_container_width=True)
                with c2:
                    st.image(preview_mask, clamp=True, use_container_width=True)

                # select tomogram to perview

    st.divider()

    if st.columns([3, 1, 3])[1].button("Save job", use_container_width=True, type="primary"):
        save_job(job_config)






def view_job(job_name):
    st.text(f"Start or continue template matching:")
    st.code(f"pom astm run -c {job_name}")
    st.text(f"Detect particles via matching score:")
    st.code(f"pom astm pick -c {job_name}")
    # how to run job info
    # results?
    # progress?

def clear_job_query():
    st.query_params = dict()


available_jobs = [os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(project_configuration["root"], "astm", "*", "config.json"))]
available_jobs = ["Create new job"] + available_jobs

selected_job = available_jobs[0]
if "job_name" in st.query_params:
    selected_job = st.query_params["job_name"]

selected_job_idx = 0
if selected_job in available_jobs:
    selected_job_idx = available_jobs.index(selected_job)

c1, c2, c3 = st.columns([4, 3, 1])
with c1:
    st.header("Area-selective template matching")
with c3:
    selected_job = st.selectbox("Select job", options=available_jobs, on_change=clear_job_query, index=selected_job_idx)
st.divider()

if selected_job == "Create new job":
    new_job()
else:
    view_job(selected_job)
