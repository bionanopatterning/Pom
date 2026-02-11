import os, json, glob
import pandas as pd
import starfile
import mrcfile
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

DEFAULT_COLOURS = {
    "membrane": "#ff9999",
    "ribosome": "#ffd700",
    "microtubule": "#00ffff",
    "impdh": "#ff00ff",
    "actin": "#1f77b4",
    "tric": "#ff0000",
    "vault": "#e5ff6c",
    "cytoplasmic_granule": "#c4c3d0",
    "mitochondrial_granule": "#555555",
    "vimentin": "#00ff00",
    "prohibitin": "#ff8c00",
    "apoferritin": "#6fb8a8",
    "npc": "#ffb000",
    "nucleus": "#ff9900",
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

def _process_segmentation(args):
    seg_file, tomo_name, feature_name = args
    try:
        volume = mrcfile.mmap(seg_file).data

        if volume[0, 0, 0] == -1:
            return None

        if volume.dtype == np.float32:
            max_val = 1.0
        elif volume.dtype == np.uint8:
            max_val = 255.0
        elif volume.dtype == np.uint16:
            max_val = 65535.0
        else:
            max_val = float(np.iinfo(volume.dtype).max)

        val = np.sum(volume) / max_val / np.prod(volume.shape) * 100.0

        return (tomo_name, feature_name, val)
    except:
        return None

def summarize(overwrite=False, target_feature=None):
    workers = os.cpu_count()
    config = get_config()
    summary_path = os.path.join('pom', 'summary.star')

    if not os.path.exists(summary_path) or overwrite:
        df = pd.DataFrame()
    else:
        df = starfile.read(os.path.join('pom', 'summary.star'), parse_as_string=["tomogram"])
        df = df.set_index('tomogram')
        df.index = df.index.astype(str)

    tomograms = []
    for src in config['tomogram_sources']:
        tomograms.extend(glob.glob(os.path.join(src, '*.mrc')))

    print(f"Found {len(tomograms)} tomograms")

    if len(tomograms) == 0:
        print("No tomograms found")
        return

    tasks = []
    for tomo_path in tomograms:
        tomo_name = str(os.path.splitext(os.path.basename(tomo_path))[0])

        segmentations = {}
        for src in config['segmentation_sources']:
            pattern = os.path.join(src, f'{tomo_name}__*.mrc')
            for seg_path in glob.glob(pattern):
                feature_name = os.path.splitext(os.path.basename(seg_path))[0].split('__', 1)[1]
                segmentations[feature_name] = seg_path

        if len(segmentations) == 0:
            if tomo_name not in df.index:
                df.loc[tomo_name] = np.nan
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

def _process_projection(args):
    from PIL import Image
    file_path, output_path, is_tomogram = args
    try:
        volume = mrcfile.mmap(file_path).data

        if is_tomogram:
            central_idx = volume.shape[0] // 2
            slice_data = volume[central_idx, :, :]
        else:
            slice_data = np.sum(volume, axis=0)

        p_low, p_high = np.percentile(slice_data, [1, 99])
        slice_data = np.clip(slice_data, p_low, p_high)
        slice_data = ((slice_data - p_low) / (p_high - p_low) * 255).astype(np.uint8)

        img = Image.fromarray(slice_data, mode='L')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)

        return True
    except:
        return False

def projections():
    workers = os.cpu_count()
    config = get_config()

    tasks = []

    for src in config['tomogram_sources']:
        for tomo_path in glob.glob(os.path.join(src, '*.mrc')):
            tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
            output_path = os.path.join('pom', 'images', 'density', f'{tomo_name}.png')
            tasks.append((tomo_path, output_path, True))

    for src in config['segmentation_sources']:
        for seg_path in glob.glob(os.path.join(src, '*.mrc')):
            seg_filename = os.path.splitext(os.path.basename(seg_path))[0]
            tomo_name, feature_name = seg_filename.split('__', 1)
            output_path = os.path.join('pom', 'images', f'{feature_name}_projection', f'{tomo_name}.png')
            tasks.append((seg_path, output_path, False))

    print(f"Projecting {len(tasks)} segmentations with {workers} workers...")

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

def render(overwrite=True):
    import multiprocessing
    import itertools
    import time

    config = get_config()
    feature_library = get_feature_library()
    compositions = get_compositions()

    summary_path = os.path.join('pom', 'summary.star')
    if not os.path.exists(summary_path):
        print("No summary found. Please run 'pom summarize' first.")
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

    parallel_processes = os.cpu_count()
    print(f"Rendering {len(tomograms)} tomograms in {len(compositions)} compositions with {parallel_processes} workers...")

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


