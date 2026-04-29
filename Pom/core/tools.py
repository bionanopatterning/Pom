import os, json, glob
import pandas as pd
import starfile
import mrcfile
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from scipy import ndimage

DEFAULT_COLOURS = {
    "membrane": "#4e4e4e",
    "ribosome": "#ffd700",
    "microtubule": "#00ffff",
    "impdh": "#ff00ff",
    "actin": "#1f77b4",
    "tric": "#ff0000",
    "vault": "#8100bd",
    "cytoplasmic_granule": "#d1d1d1",
    "mitochondrial_granule": "#0F0F0F",
    "vimentin": "#21ac21",
    "prohibitin": "#FFE5C5",
    "apoferritin": "#6fb8a8",
    "npc": "#00ff0d",
    "nucleus": "#fcff51",
    "cytoplasm": "#00ffff",
    "void": "#b0b8c1",
    "nuclear_envelope": "#0000ff",
}

AIS_DEFAULT_COLOURS = [
    "#42d6a4",
    "#fff300",
    "#ff6800",
    "#ff0d00",
    "#ae00ff",
    "#1500ff",
    "#0088ff",
    "#00f7ff",
    "#00ff00",
]

def get_feature_library():
    library_path = os.path.join('pom', 'feature_library.json')
    if not os.path.exists(library_path):
        return {}
    else:
        with open(library_path, 'r') as f:
            return json.load(f)

def save_feature_library(library):
    with open(os.path.join('pom', 'feature_library.json'), 'w') as f:
        json.dump(library, f, indent=4)

def add_feature_to_library(feature):
    feature_library = get_feature_library()
    if feature in DEFAULT_COLOURS:
        color = DEFAULT_COLOURS[feature]
    else:
        color = min(AIS_DEFAULT_COLOURS, key=lambda c: [f.get("color") for f in feature_library.values()].count(c))

    feature_library[feature] = {
        'visible': False if feature == 'void' else True,
        'color': color,
        'sigma': 0.0,
        'dust': 100000.0,
        'threshold': 0.5,
    }

    save_feature_library(feature_library)

def get_compositions(new=False):
    if not new:
        composition_path = os.path.join('pom', 'image_compositions.json')
        if not os.path.exists(composition_path):
            get_compositions(new=True)
        else:
            with open(composition_path, 'r') as f:
                return json.load(f)
    else:
        return {
            'thumbnail': ['rank1', 'rank2', 'rank3']
        }

def save_compositions(compositions):
    with open(os.path.join('pom', 'image_compositions.json'), 'w') as f:
        json.dump(compositions, f, indent=4)

def get_config(new=False):
    if not new:
        config_path = os.path.join('pom', 'config.json')
        if not os.path.isfile(config_path):
            print(f'No Pom project found in the current directory. Please run "pom initialize" to create a new project.')
            exit()
        else:
            with open(config_path, 'r') as f:
                return json.load(f)
    else:
        if os.path.exists(os.path.join('pom', 'config.json')):
            print("Config file already exists: pom/config.json")
        return {
            'tomogram_sources': [],
            'segmentation_sources': []
        }

def save_config(config):
    with open(os.path.join('pom', 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def initialize():
    os.makedirs('pom', exist_ok=True)
    os.makedirs(os.path.join('pom', 'images'), exist_ok=True)
    os.makedirs(os.path.join('pom', 'subsets'), exist_ok=True)

    save_config(get_config(new=True))
    save_compositions(get_compositions(new=True))

def get_tomogram_by_name(tomo):
    config = get_config()
    for src in config['tomogram_sources']:
        path = os.path.join(src, f'{tomo}.mrc')
        if os.path.isfile(path):
            return path
        path = os.path.join(src, f'{tomo}')
        if os.path.isfile(path):
            return path
    return None

def get_segmentation_for_tomogram(tomo, feature):
    config = get_config()
    for src in config['segmentation_sources']:
        path = os.path.join(src, f'{tomo}__{feature}.mrc')
        if os.path.isfile(path):
            return path
        path = os.path.join(src, f'{tomo.replace(".mrc", "")}__{feature}.mrc')
        if os.path.isfile(path):
            return path
    return None

def list_sources():
    config = get_config()

    print("\nTomogram sources:")
    for src in config['tomogram_sources']:
        print(f"\t{src} - {len(glob.glob(os.path.join(src, '*.mrc')))} .mrc files")

    print("\nSegmentation sources:")
    for src in config['segmentation_sources']:
        print(f"\t{src} - {len(glob.glob(os.path.join(src, '*.mrc')))} .mrc files")

def add_source(tomogram_source=None, segmentation_source=None):
    config = get_config()

    if tomogram_source:
        if not os.path.exists(tomogram_source):
            print(f"Tomogram source '{tomogram_source}' does not exist.")
            exit()
        if tomogram_source not in config['tomogram_sources']:
            config['tomogram_sources'].append(tomogram_source)

    if segmentation_source:
        if not os.path.exists(segmentation_source):
            print(f"Segmentation source '{segmentation_source}' does not exist.")
            exit()
        if segmentation_source not in config['segmentation_sources']:
            config['segmentation_sources'].append(segmentation_source)

    save_config(config)

def remove_source(tomogram_source=None, segmentation_source=None):
    config = get_config()

    if tomogram_source in config['tomogram_sources']:
        config['tomogram_sources'].remove(tomogram_source)
    if segmentation_source in config['segmentation_sources']:
        config['segmentation_sources'].remove(segmentation_source)

    save_config(config)

def _is_placeholder(path, threshold_kb=10):
    try:
        return os.path.getsize(path) < threshold_kb * 1024
    except OSError:
        return True


def _process_segmentation(args):
    seg_file, tomo_name, feature_name = args
    try:
        if _is_placeholder(seg_file):
            return None

        volume = mrcfile.mmap(seg_file).data

        if volume[0, 0, 0] == -1:
            return None

        if volume.dtype == np.float32:
            max_val = 1.0
            volume = np.where(volume == 2, 0, volume)  # little hack for easymode 3D training data + pom visualization. to be removed, eventually
        elif volume.dtype == np.int8:
            max_val = 127
        elif volume.dtype == np.uint16:
            max_val = 255
        else:
            max_val = float(np.iinfo(volume.dtype).max)

        val = np.sum(volume) / max_val / np.prod(volume.shape) * 100.0

        return (tomo_name, feature_name, val)
    except:
        return None

def summarize(overwrite=True, target_feature=None):
    workers = os.cpu_count()
    config = get_config()
    summary_path = os.path.join('pom', 'summary.star')

    tomograms = []
    for src in config['tomogram_sources']:
        tomograms.extend(glob.glob(os.path.join(src, '*.mrc')))

    print(f"Found {len(tomograms)} tomograms")

    if len(tomograms) == 0:
        print("No tomograms found")
        return

    if not os.path.exists(summary_path) or overwrite:
        tomo_names = [str(os.path.splitext(os.path.basename(t))[0]) for t in tomograms]
        df = pd.DataFrame(index=tomo_names)
        df.index.name = 'tomogram'
    else:
        df = starfile.read(summary_path, parse_as_string=["tomogram"])
        df = df.set_index('tomogram')
        df.index = df.index.astype(str)
        current_names = {str(os.path.splitext(os.path.basename(t))[0]) for t in tomograms}
        df = df[df.index.isin(current_names)] # remove entries from now-missing sources or tomograms
        for t in tomograms:
            name = str(os.path.splitext(os.path.basename(t))[0])
            if name not in df.index:
                df.loc[name] = np.nan

    tasks = []
    feature_key = "*" if target_feature is None else f'{target_feature}'
    for tomo_path in tomograms:
        tomo_name = str(os.path.splitext(os.path.basename(tomo_path))[0])

        segmentations = {}
        for src in config['segmentation_sources']:
            pattern = os.path.join(src, f'{tomo_name}__{feature_key}.mrc')
            for seg_path in glob.glob(pattern):
                feature_name = os.path.splitext(os.path.basename(seg_path))[0].split('__', 1)[1]
                segmentations[feature_name] = seg_path

        if len(segmentations) == 0:
            continue

        for feature_name, seg_file in segmentations.items():
            if target_feature is not None and feature_name != target_feature:
                continue

            if not overwrite and tomo_name in df.index and feature_name in df.columns and not pd.isna(df.at[tomo_name, feature_name]):
                continue

            tasks.append((seg_file, tomo_name, feature_name))

    print(f"Summarizing {len(tasks)} segmentations with {workers} workers...")

    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(_process_segmentation, tasks), total=len(tasks)))

    for result in results:
        if result is not None:
            tomo_name, feature_name, val = result
            df.at[tomo_name, feature_name] = val

    df.index = df.index.astype(str)
    df.index.name = 'tomogram'

    df_to_write = df.reset_index()
    df_to_write['tomogram'] = df_to_write['tomogram'].astype(str)
    starfile.write(df_to_write, summary_path)
    print(f"\nSummary saved at {summary_path}")

    # update feature library.
    df_columns = set(df.columns)
    feature_library = get_feature_library()
    for f in df_columns:
        if f not in feature_library:
            print(f'Added new feature "{f}" to the feature library.')
            add_feature_to_library(f)

def summarize_star(star_path, tomo_col=None, column_name=None, substitutions=None, overwrite=True):
    config = get_config()
    summary_path = os.path.join('pom', 'summary.star')

    df_star = starfile.read(star_path)
    tomo_col = tomo_col or ("rlnMicrographName" if "rlnMicrographName" in df_star.columns else "wrpSourceName")
    if tomo_col not in df_star.columns:
        print(f'No column "{tomo_col}" found in {star_path}. Aborting.')
        return

    valid_tomos = set()
    for src in config['tomogram_sources']:
        for t in glob.glob(os.path.join(src, '*.mrc')):
            name = os.path.basename(t)
            if name.endswith('.mrc'):
                name = name[:-4]
            valid_tomos.add(str(name))

    counts = {}
    all_star_tomos = set()
    for val in df_star[tomo_col]:
        name = val
        if substitutions:
            for search, replace in [s.split(':', 1) for s in substitutions]:
                name = name.replace(search, replace)
        name = os.path.basename(name)
        if name.endswith('.mrc'):
            name = name[:-4]
        name = str(name)
        all_star_tomos.add(name)
        if name in valid_tomos:
            counts[name] = counts.get(name, 0) + 1

    print(f'Found {len(all_star_tomos)} tomograms in star file, {len(counts)} match a tomogram source.')

    if not counts:
        print('No particles found matching any sourced tomograms. Check substitutions and tomo column.')
        return

    col_name = "particle_" + (column_name or os.path.splitext(os.path.basename(star_path))[0])

    if os.path.exists(summary_path):
        df_summary = starfile.read(summary_path, parse_as_string=["tomogram"])
        df_summary = df_summary.set_index('tomogram')
        df_summary.index = df_summary.index.astype(str)
    else:
        df_summary = pd.DataFrame(index=pd.Index(sorted(valid_tomos), name='tomogram'))

    for name in valid_tomos:
        if name not in df_summary.index:
            df_summary.loc[name] = np.nan

    if not overwrite and col_name in df_summary.columns:
        print(f'Column "{col_name}" already exists. Use overwrite=True to replace.')
        return

    df_summary[col_name] = df_summary.index.map(lambda x: counts.get(x, 0))

    df_to_write = df_summary.reset_index()
    df_to_write['tomogram'] = df_to_write['tomogram'].astype(str)
    starfile.write(df_to_write, summary_path)
    print(f'Added column "{col_name}" to {summary_path} ({sum(counts.values())} particles across {len(counts)} tomograms)')

def _process_projection(args):
    from PIL import Image
    file_path, output_path, is_tomogram = args
    try:
        volume = mrcfile.read(file_path)

        if is_tomogram:
            central_idx = volume.shape[0] // 2
            slice_data = volume[central_idx, :, :]
            p_low, p_high = np.percentile(slice_data, [1, 99])
            slice_data = np.clip(slice_data, p_low, p_high)
            denom = p_high - p_low
            slice_data = np.zeros_like(slice_data, dtype=np.uint8) if denom == 0 else ((slice_data - p_low) / denom * 255).astype(np.uint8)
        else:
            max_value = 127 if volume.dtype == np.int8 else 255 if volume.dtype == np.uint16 else 1.0
            if max_value == 1.0:
                volume[volume == 2] = 0
            volume[volume < max_value / 2] = 0
            slice_data = np.sum(volume, axis=0)
            p_high = np.percentile(slice_data[slice_data > 0], 99) if np.any(slice_data > 0) else 1
            slice_data = (np.clip(slice_data, 0, p_high) / p_high * 255).astype(np.uint8)

        img = Image.fromarray(slice_data, mode='L')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)

        return True
    except:
        return False

def projections(overwrite=False):
    workers = 32
    config = get_config()

    tasks = []

    for src in config['tomogram_sources']:
        for tomo_path in glob.glob(os.path.join(src, '*.mrc')):
            tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
            output_path = os.path.join('pom', 'images', 'density', f'{tomo_name}.png')
            if overwrite or not os.path.exists(output_path):
                tasks.append((tomo_path, output_path, True))

    for src in config['segmentation_sources']:
        for seg_path in glob.glob(os.path.join(src, '*__*.mrc')):
            if _is_placeholder(seg_path):
                continue
            seg_filename = os.path.splitext(os.path.basename(seg_path))[0]
            tomo_name, feature_name = seg_filename.split('__', 1)
            output_path = os.path.join('pom', 'images', f'{feature_name}_projection', f'{tomo_name}.png')
            if overwrite or not os.path.exists(output_path):
                tasks.append((seg_path, output_path, False))

    if not tasks:
        print("All projections already exist. Use --overwrite to regenerate.")
        return

    print(f"Projecting images for {len(tasks)} volumes with {workers} workers...")

    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(_process_projection, tasks), total=len(tasks)))

def _render_worker(tomo_paths, df, config, feature_library, compositions, overwrite, counter, lock):
    """Worker process that creates ONE renderer and processes all assigned tomograms."""
    from Pom.core.render import Renderer, SurfaceModel
    from PIL import Image

    try:
        renderer = Renderer(image_size=1024)

        for tomo_path in tomo_paths:
            tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]

            if tomo_name not in df.index:
                with lock:
                    counter.value += 1
                continue

            sorted_features = df.loc[tomo_name].sort_values(ascending=False).index.tolist()

            for comp_name, comp_def in compositions.items():
                output_path = os.path.join('pom', 'images', comp_name, f'{tomo_name}.png')

                if not overwrite and os.path.exists(output_path):
                    continue

                available_features = [f for f in sorted_features if f in feature_library and feature_library[f].get('visible', True)]

                features_to_render = []

                for item in comp_def:
                    if item.startswith('!'):
                        exclude_feature = item[1:]
                        if exclude_feature in available_features:
                            available_features.remove(exclude_feature)
                    elif item.startswith('rank'):
                        rank = int(item[4:]) - 1
                        if rank < len(available_features):
                            features_to_render.append(available_features[rank])
                    else:
                        features_to_render.append(item)

                renderables = []
                for feature in features_to_render:
                    if feature not in feature_library:
                        continue

                    seg_path = None
                    for src in config['segmentation_sources']:
                        path = os.path.join(src, f'{tomo_name}__{feature}.mrc')
                        if os.path.exists(path):
                            seg_path = path
                            break

                    if not seg_path:
                        continue

                    with mrcfile.open(seg_path, permissive=True) as mrc:
                        volume = np.copy(mrc.data)
                        pixel_size = mrc.voxel_size.x

                    renderable = SurfaceModel(volume, feature_library[feature], pixel_size)
                    renderables.append(renderable)

                if renderables:
                    renderer.new_image()
                    renderer.render(renderables)
                    image = renderer.get_image()
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    Image.fromarray(image).save(output_path)

                    for r in renderables:
                        r.delete()

            # Update progress after completing this tomogram
            with lock:
                counter.value += 1

        renderer.delete()
    except KeyboardInterrupt:
        pass

def render(overwrite=False):
    import multiprocessing
    import itertools
    import time

    config = get_config()
    feature_library = get_feature_library()
    compositions = get_compositions()

    summary_path = os.path.join('pom', 'summary.star')
    if not os.path.exists(summary_path):
        print("No summary found. Run 'pom summarize' first.")
        return

    df = starfile.read(os.path.join('pom', 'summary.star'), parse_as_string=["tomogram"])
    df = df.set_index('tomogram')

    if 'tomogram' in df.columns:
        df = df.set_index('tomogram')

    for comp_name in compositions.keys():
        os.makedirs(os.path.join('pom', 'images', comp_name), exist_ok=True)

    tomograms = []
    for src in config['tomogram_sources']:
        tomograms.extend(glob.glob(os.path.join(src, '*.mrc')))

    parallel_processes = min(os.cpu_count(), 16)
    print(f"Rendering {len(tomograms)} tomograms in {len(compositions)} {'composition' if len(compositions) == 1 else 'compositions'} with {parallel_processes} workers...")

    # Divide tomograms among processes
    process_div = {p: [] for p in range(parallel_processes)}
    for p, tomo_path in zip(itertools.cycle(range(parallel_processes)), tomograms):
        process_div[p].append(tomo_path)

    # Create shared counter and lock for progress tracking
    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    processes = []
    try:
        for p in process_div:
            proc = multiprocessing.Process(
                target=_render_worker,
                args=(process_div[p], df, config, feature_library, compositions, overwrite, counter, lock)
            )
            processes.append(proc)
            proc.start()

        # Monitor progress
        with tqdm(total=len(tomograms)) as pbar:
            last_value = 0
            while any(proc.is_alive() for proc in processes):
                current_value = counter.value
                if current_value > last_value:
                    pbar.update(current_value - last_value)
                    last_value = current_value
                time.sleep(0.1)

            # Final update
            current_value = counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)

        for proc in processes:
            proc.join()

        print("Rendering complete!")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating processes...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join()
        print("Rendering cancelled.")

def _distance_to_surface(segmentation_volume, normalization_value, threshold, dust_angstrom3, apix, coordinates):
    binary = (segmentation_volume.astype(np.float32) / normalization_value) >= threshold

    labeled, n_feat = ndimage.label(binary)
    sizes = ndimage.sum(binary, labeled, range(1, n_feat + 1))
    if dust_angstrom3 < 0:
        keep_n = int(-dust_angstrom3)
        if keep_n < n_feat:
            threshold_size = sorted(sizes, reverse=True)[keep_n - 1]
            for i, sz in enumerate(sizes):
                if sz < threshold_size:
                    binary[labeled == (i + 1)] = False
    else:
        min_voxels = int(np.ceil(dust_angstrom3 / (apix ** 3)))
        for i, sz in enumerate(sizes):
            if sz < min_voxels:
                binary[labeled == (i + 1)] = False

    if not binary.any():
        return np.full(len(coordinates), np.nan, dtype=np.float32)

    edt_out = ndimage.distance_transform_edt(~binary) * apix
    edt_in = ndimage.distance_transform_edt(binary) * apix
    dist_vol = np.where(binary, -edt_in, edt_out)

    distances = np.full(len(coordinates), np.nan, dtype=np.float32)
    for j, (cj, ck, cl) in enumerate(coordinates):
        c = np.round([cj, ck, cl]).astype(int)
        if np.all((c >= 0) & (c < np.array(dist_vol.shape))):
            distances[j] = dist_vol[c[0], c[1], c[2]]
    return distances


def _euler_to_primary_axis(tilt_deg, psi_deg):
    tilt = np.radians(tilt_deg)
    psi = np.radians(psi_deg)
    return np.array([
        np.cos(tilt),
        np.sin(tilt) * np.sin(psi),
        -np.sin(tilt) * np.cos(psi),
    ])


def _contextualize_job(df, samplers_parsed, tomogram_name, coords_angpix):
    tomogram_path = get_tomogram_by_name(tomogram_name)
    if tomogram_path is None:
        print(f'Original tomogram (.mrc) for {tomogram_name} missing.')
        return
    with mrcfile.open(tomogram_path, header_only=True) as mrc:
        apix = mrc.voxel_size.x
    if "rlnCoordinateX" in df.columns:
        if coords_angpix is not None:
            scale = coords_angpix / apix
        else:
            scale = 1.0  # assume same pixel size as tomogram
        x = df["rlnCoordinateX"].values * scale
        y = df["rlnCoordinateY"].values * scale
        z = df["rlnCoordinateZ"].values * scale
    elif "wrpCoordinateX1" in df.columns:
        if coords_angpix is not None:
            scale = coords_angpix / apix
        else:
            scale = 1.0 / apix  # M stores in Angstrom
        x = df["wrpCoordinateX1"].values * scale
        y = df["wrpCoordinateY1"].values * scale
        z = df["wrpCoordinateZ1"].values * scale

    coordinates = np.stack([z, y, x], axis=1)

    has_offset = any(s[0] == 'sphere' and s[3] != 0.0 for s in samplers_parsed)
    axes = None
    if has_offset:
        if 'rlnAngleTilt' in df.columns and 'rlnAnglePsi' in df.columns:
            axes = np.stack([
                _euler_to_primary_axis(t, p)
                for t, p in zip(df['rlnAngleTilt'].values, df['rlnAnglePsi'].values)
            ])
        else:
            print(f'Warning: axis offset requested but rlnAngleTilt/rlnAnglePsi not found. Sampling at particle center.')

    def spherical_mask(radius, apix):
        _r = int(np.ceil(radius / apix))
        grid = np.arange(-_r, _r+1)
        jj, kk, ll = np.meshgrid(grid, grid, grid, indexing='ij')
        mask = (jj**2 + kk**2 + ll**2) < (radius / apix)**2
        indices = np.stack([jj[mask], kk[mask], ll[mask]], axis=1)
        return indices

    for sampler in samplers_parsed:
        if sampler[0] == 'sphere':
            _, feature, radius, offset = sampler
            if offset != 0.0:
                sign = 'p' if offset > 0 else 'm'
                column_name = f"pom{feature.replace('_', ' ').title().replace(' ', '')}{int(radius)}A{sign}{int(round(abs(offset)))}"
            else:
                column_name = f"pom{feature.replace('_', ' ').title().replace(' ', '')}{int(radius)}A"
            segmentation_path = get_segmentation_for_tomogram(tomogram_name, feature)
            if segmentation_path is None:
                df[column_name] = np.nan
                continue

            segmentation_volume = mrcfile.read(segmentation_path)
            normalization_value = 127 if segmentation_volume.dtype == np.int8 else 255 if segmentation_volume.dtype == np.uint16 else 1.0
            indices = spherical_mask(radius, apix)

            context_values = np.full(len(coordinates), np.nan, dtype=np.float32)
            for j, coord in enumerate(coordinates):
                if offset != 0.0 and axes is not None:
                    center = coord + axes[j] * (offset / apix)
                else:
                    center = coord
                c = np.round(center).astype(int)
                sample_indices = indices + c
                within_bounds = np.all((sample_indices >= 0) & (sample_indices < segmentation_volume.shape), axis=1)
                sample_indices = sample_indices[within_bounds]
                context_values[j] = np.mean(segmentation_volume[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]].astype(np.float32)) / normalization_value

            df[column_name] = context_values

        elif sampler[0] == 'distance':
            _, feature, threshold, dust = sampler
            column_name = f"pomDist{feature.replace('_', ' ').title().replace(' ', '')}T{int(threshold*100)}"
            segmentation_path = get_segmentation_for_tomogram(tomogram_name, feature)
            if segmentation_path is None:
                df[column_name] = np.nan
                continue

            segmentation_volume = mrcfile.read(segmentation_path)
            normalization_value = 127 if segmentation_volume.dtype == np.int8 else 255 if segmentation_volume.dtype == np.uint16 else 1.0
            df[column_name] = _distance_to_surface(segmentation_volume, normalization_value, threshold, dust, apix, coordinates)

    return df


def _contextualize_worker(tomo_batch, samplers_parsed, df, tomo_col, coords_angpix, counter, lock):
    results = {}
    for df_tomo_name, mrc_tomo_name in tomo_batch:
        df_subset = df[df[tomo_col] == df_tomo_name].copy()
        results[df_tomo_name] = _contextualize_job(df_subset, samplers_parsed, mrc_tomo_name, coords_angpix)
        with lock:
            counter.value += 1
    return results


def contextualize_starfile(star_path, samplers, tomogram_name=None, substitutions=None, out_star=None, coords_angpix=None):
    import itertools, multiprocessing, time
    df = starfile.read(star_path)

    print()
    samplers_parsed = []
    for s in samplers:
        parts = s.split(':')
        if len(parts) == 2:
            samplers_parsed.append(('sphere', parts[0], float(parts[1]), 0.0))
        elif len(parts) == 3 and float(parts[1]) > 1.0 and (parts[2].startswith('+') or parts[2].startswith('-')):
            samplers_parsed.append(('sphere', parts[0], float(parts[1]), float(parts[2])))
        elif len(parts) == 3:
            samplers_parsed.append(('distance', parts[0], float(parts[1]), float(parts[2])))
        else:
            print(f'Invalid sampler format: {s}')
            return

    print(f'\nContextualizing {len(df)} particles with {len(samplers_parsed)} samplers:')
    for i, s in enumerate(samplers_parsed, 1):
        if s[0] == 'sphere':
            _, feature, radius, offset = s
            col = f"pom{feature.replace('_', ' ').title().replace(' ', '')}{int(radius)}A"
            if offset != 0.0:
                sign = 'p' if offset > 0 else 'm'
                col += f"{sign}{int(round(abs(offset)))}"
            offset_str = f", offset {offset:+.0f} Å along primary axis" if offset != 0.0 else ""
            print(f'\t{i}. {col}: average {feature} value within {radius:.0f} Å radius{offset_str}.')
        elif s[0] == 'distance':
            _, feature, threshold, dust = s
            col = f"pomDist{feature.replace('_', ' ').title().replace(' ', '')}T{int(threshold*100)}"
            if dust < 0:
                dust_desc = f"keeping only {int(-dust)} largest blob(s)"
            else:
                dust_desc = f"ignoring blobs < {dust:.2e} cubic Å"
            print(f'\t{i}. {col}: distance to {feature} (threshold={threshold}, {dust_desc}).')
    print()

    for f in set(s[1] for s in samplers_parsed):
        n_feature_volumes = 0
        for src in get_config()['segmentation_sources']:
            pattern = os.path.join(src, f'*__{f}.mrc')
            n_feature_volumes += len(glob.glob(pattern))
        print(f'found {n_feature_volumes} volumes for feature "{f}".')
    print()

    tomo_col = tomogram_name or ("rlnMicrographName" if "rlnMicrographName" in df.columns else "wrpSourceName")
    if tomo_col not in df.columns:
        print(f'No column "{tomo_col}" found in {star_path}. Aborting.')
        return

    parallel_processes = min([os.cpu_count(), 32])
    tomograms = df[tomo_col].unique()
    process_div = {p: [] for p in range(parallel_processes)}
    for p, tomogram in zip(itertools.cycle(range(parallel_processes)), tomograms):
        mrc_tomo_name = tomogram
        if substitutions:
            for search, replace in [s.split(':', 1) for s in substitutions]:
                mrc_tomo_name = mrc_tomo_name.replace(search, replace)
        process_div[p].append((tomogram, mrc_tomo_name))

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    print(f'sampling context for {len(tomograms)} tomograms with {parallel_processes} workers.')
    with multiprocessing.Pool(processes=parallel_processes) as pool:
        async_results = [pool.apply_async(_contextualize_worker,
                                          (process_div[p], samplers_parsed, df, tomo_col, coords_angpix, counter, lock))
                         for p in range(parallel_processes)]

        with tqdm(total=len(tomograms)) as pbar:
            last_value = 0
            while not all(r.ready() for r in async_results):
                current_value = counter.value
                if current_value > last_value:
                    pbar.update(current_value - last_value)
                    last_value = current_value
                time.sleep(0.1)

            current_value = counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)

        results = [r.get() for r in async_results]

    output_df = pd.concat([df_result for batch_results in results for df_result in batch_results.values()], ignore_index=True)

    if out_star is None:
        starfile.write(output_df, star_path)
    else:
        if not '.star' in out_star:
            out_star += '.star'
        starfile.write(output_df, out_star)