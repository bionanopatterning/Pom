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
import pandas as pd
from copy import copy
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import multiprocessing

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


def crop_from_tomo(tomo, x, y, z, s):
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


    fig, ax = plt.subplots()
    _x = x - (_k - s)
    _y = y - (_j - s)
    ax.imshow(image[::-1], cmap='gray')
    ax.axis('off')
    ax.scatter(_x, s - _y, marker='o', s=4000, facecolors='none', edgecolors='red', linewidth=2)
    plt.close(fig)
    return fig



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


def open_in_ais(tomo_name, slice=128):
    cmd_path = os.path.join(os.path.expanduser("~"), ".Ais", "pom_to_ais.cmd")
    tomo_dir = os.path.join(project_configuration["root"], project_configuration["tomogram_dir"])
    with open(cmd_path, 'a') as f:
        base = os.path.abspath(os.path.join(tomo_dir, f"{tomo_name}"))
        if os.path.exists(base+".scns"):
            f.write(f"open\t{base+'.scns'}\tslice\t{slice}\n")
        else:
            f.write(f"open\t{base + '.mrc'}\tslice\t{slice}\n")


def generate_template_previews(job_config, n_samples=18):
    Pommie.compute.initialize()

    template = Pommie.typedefs.Particle.from_path(job_config["template_path"])
    template = template.bin(job_config["template_binning"])
    template.data -= np.mean(template.data)
    template.data /= np.std(template.data)
    template.data = (template.data + 2.0) / 4.0
    if job_config["template_blur"] > 0:
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


def generate_mask_preview(job_config, selected_tomo):
    # grab a random tomo central slice
    tomo_path = os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"{selected_tomo}.mrc")
    slice = mrcfile.mmap(tomo_path).data
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

        segmentation_path = os.path.join(project_configuration["root"], project_configuration["output_dir"], f"{selected_tomo}__{f.o}.mrc")

        vol = mrcfile.mmap(segmentation_path)
        pxs = vol.voxel_size.x / 10.0
        img = vol.data[vol.data.shape[0]//2, :, :][np.newaxis, :, :]
        img = Pommie.typedefs.Volume.from_array(img)
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
            mask = np.logical_or(mask, f.mask)
    for f in st.session_state.area_filters:
        if not f.active:
            continue
        if f.logic == "exclude":
            mask = np.logical_and(mask, np.logical_not(f.mask))

    mask = mask * 255
    for j in range(stride-1):
        mask[1+j::stride, :] = 0
        mask[:, 1+j::stride] = 0
    return slice_pxd, mask


def generate_result_previews(job_name, tomo, score_contrast_lims=(0.0, 1.0), norm_slice_idx=0.5):
    density = mrcfile.mmap(os.path.join(project_configuration["root"],
                                        project_configuration["tomogram_dir"],
                                        f"{tomo}.mrc")).data
    score = mrcfile.mmap(os.path.join(project_configuration["root"],
                                      "astm",
                                      f"{job_name}",
                                      f"{tomo}__score.mrc")).data

    density_image = copy(density[int(density.shape[0] * norm_slice_idx), :, :])
    score_image = copy(score[int(score.shape[0] * norm_slice_idx), :, :])

    # density_image -= np.mean(density_image)
    # density_image /= np.std(density_image)


    fig1, ax1 = plt.subplots()
    ax1.imshow(density_image[::-1, :], cmap="gray")
    ax1.axis('off')
    fig2, ax2 = plt.subplots()
    ax2.imshow(score_image[::-1, :], cmap="jet", vmin=score_contrast_lims[0], vmax=score_contrast_lims[1])
    ax2.axis('off')

    # fig3, ax3 = plt.subplots()
    # fig3.set_figheight(0.2)
    # norm = colors.Normalize(vmin=score_contrast_lims[0], vmax=score_contrast_lims[1], clip=True)
    #
    # # Create a horizontal colorbar
    # cb = colorbar.ColorbarBase(ax3, cmap="jet", norm=norm, orientation="horizontal")
    # ax3.axis('off')
    # plt.xlim([0.0, 1.0])

    fig4, ax4 = plt.subplots()
    fig4.set_figheight(2)
    hist_vals = score_image[score_image != 0].flatten()
    ax4.hist(hist_vals, log=True, bins=50)
    ax4.get_yaxis().set_visible(False)
    ax4.set_xticks([0.1, 0.5, 0.9])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    ylim = ax4.get_ylim()
    ax4.plot([score_contrast_lims[0], score_contrast_lims[0]], ylim, c=(1.0, 0.2, 0.2))
    ax4.plot([score_contrast_lims[1], score_contrast_lims[1]], ylim, c=(1.0, 0.2, 0.2))
    ax4.set_ylim(ylim)
    ax4.set_xlim([0.0, 1.0])
    ax4.set_title("Matching score distribution")

    plt.close(fig1)
    plt.close(fig2)
    # plt.close(fig3)
    plt.close(fig4)
    return fig1, fig2, fig4


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
    job_config["job_name"] = c1.text_input("Job name", value="New ASTM job")
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
                available_tomos = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"*.mrc"))]
                if "mask_tomo_preview" not in st.session_state:
                    st.session_state.mask_tomo_preview = available_tomos[0]
                _idx = 0 if st.session_state.mask_tomo_preview not in available_tomos else available_tomos.index(st.session_state.mask_tomo_preview)
                c1, c2 = st.columns([19, 1], vertical_alignment="bottom")
                with c1:
                    st.session_state.mask_tomo_preview = st.selectbox("Test tomogram index", index=_idx, options=available_tomos)
                with c2:
                    if st.button("🎲"):
                        st.session_state.mask_tomo_preview = random.choice(available_tomos)
                preview_slice, preview_mask = generate_mask_preview(job_config, st.session_state.mask_tomo_preview)
                c1, c2 = st.columns([2, 2])
                with c1:
                    st.image(preview_slice, clamp=True, use_container_width=True)
                with c2:
                    st.image(preview_mask, clamp=True, use_container_width=True)

                # select tomogram to perview

    st.divider()

    if st.columns([3, 1, 3])[1].button("Save job", use_container_width=True, type="primary"):
        save_job(job_config)


def view_particles(job_name):


    N_ROWS = 6
    N_COLS = 5
    IMG_SIZE = 128

    # load particle data.
    data_path = os.path.join(project_configuration["root"], "astm", job_name, "all_particles.tsv")
    if not os.path.exists(data_path):
        st.markdown("<p style='text-align: center; font-size: 20px;'>No particles found (yet).</p>",
                        unsafe_allow_html=True)
        return

    with open(os.path.join(project_configuration["root"], "astm", job_name, f"config.json"), 'r') as f:
        job_config = json.load(f)

    unbin = job_config["template_binning"]
    all_particles = pd.read_table(data_path, header=None)

    if "particle_page" not in st.session_state:
        st.session_state.particle_page = 1

    n_pages = int(np.ceil(len(all_particles) / (N_COLS * N_ROWS)))
    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")

    with c1:
        filter_value = st.text_input("Filter tomograms", value="")
        all_particles = all_particles[all_particles.iloc[:, 4].str.contains(filter_value, na=False)]

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
        sort_options = ["Score (high to low)", "Score (low to high)", "Tomo (A to Z)", "Tomo (Z to A)"]
        sort_option = st.selectbox("Sorting options", options=sort_options, label_visibility="collapsed")

        if sort_option == sort_options[0]:
            all_particles = all_particles.sort_values(by=all_particles.columns[3], ascending=False)
        elif sort_option == sort_options[1]:
            all_particles = all_particles.sort_values(by=all_particles.columns[3])
        elif sort_option == sort_options[2]:
            all_particles = all_particles.sort_values(by=all_particles.columns[4])
        elif sort_option == sort_options[3]:
            all_particles = all_particles.sort_values(by=all_particles.columns[4], ascending=False)

    def display_particle(x, y, z, score, tomo, t_idx, uid):
        img = crop_from_tomo(tomo, x, y, z, IMG_SIZE)
        st.pyplot(img, use_container_width=True)

        c1, c2 = st.columns(2, vertical_alignment="center")
        with c1:
            st.markdown(f"<p style='text-align: center; font-size: 14px;'>Particle score: <b>{score:.3f}</b></p>", #+ "\nBest transform: <b>{t_idx}</b></p>",
                        unsafe_allow_html=True)
        if c2.button("open in Ais", use_container_width=True, key=f"{uid}btn"):
            open_in_ais(tomo, slice=z)
        st.markdown(
            f"<p style='text-align: center; font-size: 15px; margin: 0px;'>{tomo}</p>",
            unsafe_allow_html=True
        )

    idx_offset = N_COLS * N_ROWS * (st.session_state.particle_page - 1)
    for j in range(min(N_ROWS, int(np.ceil((len(all_particles) - idx_offset) / N_COLS)))):
        for k, c in enumerate(st.columns(N_COLS, border=True)):
            idx = idx_offset + j * N_COLS + k
            if idx < len(all_particles):
                with c:
                    x = unbin * int(all_particles.iloc[idx, 0])
                    y = unbin * int(all_particles.iloc[idx, 1])
                    z = unbin * int(all_particles.iloc[idx, 2])
                    score = float(all_particles.iloc[idx, 3])
                    tomo = all_particles.iloc[idx, 4]
                    t_idx = int(all_particles.iloc[idx, 5])
                    display_particle(x, y, z, score, tomo, t_idx, uid=idx)

    # c1, c2, c3 = st.columns(3, vertical_alignment="center")
    # with c2:
    #     _c = st.columns([1, 1, 4, 1, 1])
    #     if _c[0].button(":material/First_Page:", key="_c0bn", type="primary"):
    #         st.session_state.particle_page = 1
    #     if _c[1].button(":material/Keyboard_Arrow_Left:", key="_c1bn", type="primary"):
    #         st.session_state.particle_page = max(st.session_state.particle_page - 1, 1)
    #     if _c[3].button(":material/Keyboard_Arrow_Right:", key="_c2bn", type="primary"):
    #         st.session_state.particle_page = min(st.session_state.particle_page + 1, n_pages)
    #     if _c[4].button(":material/Last_Page:", key="_c3bn", type="primary"):
    #         st.session_state.particle_page = n_pages
    #     _c[2].markdown(
    #         f"<p style='text-align: center; font-size: 20px;'>page {st.session_state.particle_page}/{n_pages}</p>",
    #         unsafe_allow_html=True
    #     )


def view_job(job_name):
    with st.expander("**Processing instructions**", expanded=True):
        st.text(f"Start or continue template matching:")
        st.code(f"pom astm run -c {job_name}")

        st.text(f"Find particle coordinates using the template matching scores:")
        c1, c2 = st.columns(2, vertical_alignment="bottom")
        with c2:
            c = st.columns([1, 1, 2, 1])
            threshold = c[0].number_input("Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.5, format="%0.2f")
            spacing = c[1].number_input("Spacing (px)", min_value=1, step=1, value=10)
            n_max = c[2].number_input("Max particles per tomogram", min_value=1, step=1, value=10)
            blur_px = c[3].number_input("Z-Blur scores (px)", min_value=0, step=1, value=0)
        with c1:
                n_proc_use = multiprocessing.cpu_count() // 2
                st.code(f"pom astm pick -c {job_name} -threshold {threshold:.2f} -spacing-px {spacing} -max {n_max} -blur {blur_px} -p {n_proc_use}")

    with st.expander("**Score volumes**", expanded=True):
        completed_tomos = [os.path.basename(os.path.splitext(p)[0]).split("__")[0] for p in list(filter(lambda f: os.path.getsize(f) > 10000, glob.glob(os.path.join(project_configuration["root"], "astm", f"{job_name}", "*__score.mrc"))))]
        n_completed = len(completed_tomos)
        n_total = len(glob.glob(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"*.mrc")))

        if (n_total - n_completed) == 0:
            st.markdown("<p style='text-align: center; font-size: 20px;'>✔️ all volumes processed.</p>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; font-size: 20px;'>🕜 {n_completed} volumes processed, {n_total - n_completed} remaining.</p>",
                        unsafe_allow_html=True)
        if n_completed > 0:
            columns = st.columns([4, 4, 3])
            if "preview_tomo_name" not in st.session_state:
                st.session_state.preview_tomo_name = completed_tomos[0]
            with columns[2]:
                _c = st.columns([5, 0.7], vertical_alignment="bottom")
                with _c[1]:
                    if st.button("🎲"):
                        st.session_state.preview_tomo_name = random.choice(completed_tomos)
                with _c[0]:
                    st.session_state.preview_tomo_name = st.selectbox("Select tomogram", completed_tomos, index=completed_tomos.index(st.session_state.preview_tomo_name))
                slice_idx = st.slider("Slice to view (norm. z index))", min_value=0.0, max_value=1.0, value=0.5, step=(1 / 256.0))
                score_contrast_range = st.slider("Score contrast range", value=(0.0, 1.0), min_value=0.0, max_value=1.0)

            result_preview = generate_result_previews(job_name, st.session_state.preview_tomo_name, score_contrast_range, slice_idx)
            # with columns[2]:
            #     st.pyplot(result_preview[2], use_container_width=True)
            with columns[0]:
                st.pyplot(result_preview[0], use_container_width=True)
            with columns[1]:
                st.pyplot(result_preview[1], use_container_width=True)
            with columns[2]:
                st.pyplot(result_preview[2], use_container_width=True)


    with st.expander("**Detected particles**", expanded=True):
        view_particles(job_name)


available_jobs = [os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(project_configuration["root"], "astm", "*", "config.json"))]
available_jobs = ["Create new job"] + available_jobs


if "selected_job" not in st.session_state:
    st.session_state.selected_job = "Create new job"

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
    st.header("Area-selective template matching")
with c3:
    selected_job = st.selectbox("Select job", options=available_jobs, key="selected_job", on_change=redirect())
st.divider()

if selected_job == "Create new job":
    new_job()
else:
    view_job(selected_job)
