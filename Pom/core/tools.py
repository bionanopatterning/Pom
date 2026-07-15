import os, json, glob, re
import pandas as pd
import starfile
import mrcfile
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from scipy import ndimage

DEFAULT_COLOURS = {
    "membrane": "#777777",
    "ribosome": "#ffd700",
    "microtubule": "#ffae00",
    "fhimpdh": "#ff00ff",
    "actin": "#1f77b4",
    "tric": "#ff0000",
    "vault": "#8100bd",
    "cytoplasmic_granule": "#d1d1d1",
    "mitochondrial_granule": "#0F0F0F",
    "mitochondrion": "#FF0000",
    "intermediate_filament": "#21ac21",
    "prohibitin": "#FDC5FF",
    "apoferritin": "#6fb8a8",
    "npc": "#00ff0d",
    "nucleus": "#fcff51",
    "cytoplasm": "#00ffff",
    "void": "#b0b8c1",
    "nuclear_envelope": "#0000ff",
    "lipid_droplet": "#ffbd67"
}

# Features hidden by default (not rendered at all).
DEFAULT_HIDDEN_FEATURES = {"void"}

# Large context features that look better as volume clouds than as isosurfaces; visible by
# default but rendered with the volume ray-tracer.
DEFAULT_VOLUME_FEATURES = {"nucleus", "cytoplasm", "mitochondrion", "nuclear_envelope"}

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
        'visible': feature not in DEFAULT_HIDDEN_FEATURES,
        'color': color,
        'sigma': 0.0,
        'dust': 100000.0,
        'threshold': 0.5,
        'render_as': 'volume' if feature in DEFAULT_VOLUME_FEATURES else 'isosurface',
    }

    save_feature_library(feature_library)

# Render resolutions (px) for static PNG vs spin GIF. PNGs render full-size for the
# detail page / gallery; spin GIFs render smaller since they're shown smaller and need
# to encode 180 frames cheaply.
RENDER_IMAGE_SIZE = 1024
SPIN_RENDER_SIZE = 512

# Spin movie defaults (used when a composition has spin enabled).
SPIN_FRAMES = 180
SPIN_FPS = 10  # ~100 ms per frame, ~18 s full rotation (3x smoother than 60 frames @ 3.3 fps)
# For spin movies, bin isosurface segmentations so that their largest axis is at most this many
# voxels. Triangle count (and per-frame draw cost) scales with surface area ~ axis^2, so capping
# the largest axis bounds the worst-case mesh size regardless of tomogram aspect ratio.
SPIN_SURFACE_MAX_AXIS = 256


def composition_features(comp):
    """Ordered feature/rank list of a composition, stored as either a plain list (legacy)
    or a dict {'features': [...], 'spin': bool}."""
    if isinstance(comp, dict):
        return comp.get('features', [])
    return comp


def composition_spin(comp):
    """Whether a composition should also be rendered as a 360-degree spin GIF."""
    return bool(comp.get('spin', False)) if isinstance(comp, dict) else False


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
            'thumbnail': {
                'features': ['membrane', 'ribosome', 'microtubule', 'cytoplasmic_granule', 'cytoplasm', 'mitochondrion'],
                'spin': False,
            }
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

    config = get_config(new=True)

    # Auto-detect a tomogram source: prefer 'denoised/', otherwise fall back to common
    # Warp layouts. First hit wins.
    tomo_candidates = [
        'denoised',
        os.path.join('warp_tiltseries', 'reconstruction', 'denoised'),
        os.path.join('warp_tiltseries', 'reconstruction'),
    ]
    for cand in tomo_candidates:
        if os.path.isdir(cand):
            config['tomogram_sources'].append(cand)
            print(f"  Auto-detected tomogram source: {cand}")
            break
    else:
        print("  No tomogram source auto-detected. Add one with 'pom add_source -t <path>'.")

    if os.path.isdir('segmented'):
        config['segmentation_sources'].append('segmented')
        print("  Auto-detected segmentation source: segmented")
    else:
        print("  No segmentation source auto-detected. Add one with 'pom add_source -s <path>'.")

    save_config(config)
    save_compositions(get_compositions(new=True))

    list_sources()

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

def collect_tomogram_names(config):
    """Collapse the tomogram sources into unique tomogram identities. A tomogram's
    identity is its filename, so the same name appearing in several sources (e.g. raw +
    denoised) is one tomogram with multiple flavours. Additional sources are only
    'flavours' insofar as their names actually overlap an existing tomogram; a source
    holding a disjoint set of names simply contributes more tomograms. Returns
    (unique_names, n_volume_files, max_flavours): the de-duplicated names (in first-seen
    order), the total number of .mrc files scanned, and the largest number of sources
    any single tomogram appears in (1 = no flavour overlap at all)."""
    names = []
    counts = {}
    n_files = 0
    for src in config['tomogram_sources']:
        for t in glob.glob(os.path.join(src, '*.mrc')):
            n_files += 1
            name = str(os.path.splitext(os.path.basename(t))[0])
            if name not in counts:
                counts[name] = 0
                names.append(name)
            counts[name] += 1
    max_flavours = max(counts.values()) if counts else 0
    return names, n_files, max_flavours

def describe_tomogram_count(n_unique, n_files, max_flavours):
    """Human-readable count that distinguishes unique tomograms from flavour volumes.
    Only mentions flavours when tomogram names actually overlap across sources
    (n_files > n_unique); otherwise reports the plain tomogram count."""
    if n_files == n_unique:
        return f"{n_unique} tomograms"
    return f"{n_unique} tomograms ({n_files} volume files, up to {max_flavours} flavours each)"

def source_subset_name(src):
    """Derive a subset name for a tomogram source directory. The name starts at the first
    'dataset' folder in the path (digits + underscore, e.g. '001_HELA') and joins every
    component from there to the end with underscores. If no dataset folder is present, the
    last two path components are used instead. Examples:
        a/b/c/d/001_HELA/denoised                     -> 001_HELA_denoised
        e/f/g/h/002_TEMP/warp_tiltseries/reconstruction -> 002_TEMP_warp_tiltseries_reconstruction"""
    parts = [p for p in re.split(r'[\\/]+', os.path.normpath(src)) if p and p != '.']
    if not parts:
        return 'source'
    start = next((i for i, p in enumerate(parts) if re.match(r'^\d+_', p)), None)
    chosen = parts[start:] if start is not None else parts[-2:]
    label = '_'.join(_sanitize_source_label(p) for p in chosen)
    return label or 'source'

def create_source_subsets(config):
    """When multiple tomogram sources are configured, (re)generate one subset per source
    directory, named via `source_subset_name` and listing that directory's tomograms.
    No-op for a single source. Returns a list of (subset_name, n_tomograms) created."""
    sources = config['tomogram_sources']
    if len(sources) < 2:
        return []
    subsets_dir = os.path.join('pom', 'subsets')
    os.makedirs(subsets_dir, exist_ok=True)
    created = []
    for src in sources:
        name = source_subset_name(src)
        paths = sorted(glob.glob(os.path.join(src, '*.mrc')))
        if not paths:
            continue
        with open(os.path.join(subsets_dir, f'{name}.txt'), 'w') as f:
            f.write('\n'.join(paths) + '\n')
        created.append((name, len(paths)))
    return created

def list_sources():
    """Print all configured sources with a unified 1-based index spanning tomograms
    first then segmentations. The index is what `pom remove_source N` consumes."""
    config = get_config()

    n = 1
    print("\nTomogram sources:")
    if not config['tomogram_sources']:
        print("  (none)")
    for src in config['tomogram_sources']:
        n_mrc = len(glob.glob(os.path.join(src, '*.mrc')))
        print(f"  {n}. {src} - {n_mrc} .mrc files")
        n += 1
    if len(config['tomogram_sources']) > 1:
        tomo_names, n_files, max_flavours = collect_tomogram_names(config)
        if n_files == len(tomo_names):
            print(f"  -> {len(tomo_names)} unique tomograms, {n_files} volume files (no flavour overlap)")
        else:
            print(f"  -> {len(tomo_names)} unique tomograms, {n_files} volume files (up to {max_flavours} flavours each)")

    print("\nSegmentation sources:")
    if not config['segmentation_sources']:
        print("  (none)")
    for src in config['segmentation_sources']:
        n_seg = len(glob.glob(os.path.join(src, '*__*.mrc')))
        print(f"  {n}. {src} - {n_seg} segmentation volumes")
        n += 1
    print()


def add_source(tomogram_source=None, segmentation_source=None):
    config = get_config()

    if tomogram_source:
        if not os.path.exists(tomogram_source):
            print(f"Tomogram source '{tomogram_source}' does not exist.")
            exit()
        if tomogram_source not in config['tomogram_sources']:
            config['tomogram_sources'].append(tomogram_source)
            print(f"Added tomogram source: {tomogram_source}")
        else:
            print(f"Tomogram source already configured: {tomogram_source}")

    if segmentation_source:
        if not os.path.exists(segmentation_source):
            print(f"Segmentation source '{segmentation_source}' does not exist.")
            exit()
        if segmentation_source not in config['segmentation_sources']:
            config['segmentation_sources'].append(segmentation_source)
            print(f"Added segmentation source: {segmentation_source}")
        else:
            print(f"Segmentation source already configured: {segmentation_source}")

    save_config(config)
    list_sources()


def remove_source(tomogram_source=None, segmentation_source=None, index=None):
    """Remove a source by path (--tomograms / --segmentations) or by 1-based index
    matching what `list_sources` prints (tomograms numbered first, then segmentations)."""
    config = get_config()

    if index is not None:
        n_tomo = len(config['tomogram_sources'])
        n_seg = len(config['segmentation_sources'])
        total = n_tomo + n_seg
        if total == 0:
            print("No sources configured.")
            return
        if index < 1 or index > total:
            print(f"Index {index} out of range. Valid range is 1..{total}.")
            return
        if index <= n_tomo:
            removed = config['tomogram_sources'].pop(index - 1)
            print(f"Removed tomogram source: {removed}")
        else:
            removed = config['segmentation_sources'].pop(index - n_tomo - 1)
            print(f"Removed segmentation source: {removed}")
    else:
        if tomogram_source and tomogram_source in config['tomogram_sources']:
            config['tomogram_sources'].remove(tomogram_source)
            print(f"Removed tomogram source: {tomogram_source}")
        if segmentation_source and segmentation_source in config['segmentation_sources']:
            config['segmentation_sources'].remove(segmentation_source)
            print(f"Removed segmentation source: {segmentation_source}")

    save_config(config)
    list_sources()

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
    workers = min([os.cpu_count(), 32])
    config = get_config()
    summary_path = os.path.join('pom', 'summary.star')

    tomo_names, n_files, max_flavours = collect_tomogram_names(config)

    if len(tomo_names) == 0:
        print("No tomograms found.")
        return

    # Count total segmentation volumes (linked to tomograms by name, so per unique tomogram).
    feature_key = "*" if target_feature is None else f'{target_feature}'
    total_segmentations = 0
    for tomo_name in tomo_names:
        for src in config['segmentation_sources']:
            pattern = os.path.join(src, f'{tomo_name}__{feature_key}.mrc')
            total_segmentations += len(glob.glob(pattern))

    print(f"Found {describe_tomogram_count(len(tomo_names), n_files, max_flavours)}, {total_segmentations} segmentation volumes.")

    if not os.path.exists(summary_path) or overwrite:
        df = pd.DataFrame(index=tomo_names)
        df.index.name = 'tomogram'
    else:
        df = starfile.read(summary_path, parse_as_string=["tomogram"])
        df = df.set_index('tomogram')
        df.index = df.index.astype(str)
        current_names = set(tomo_names)
        df = df[df.index.isin(current_names)] # remove entries from now-missing sources or tomograms
        for name in tomo_names:
            if name not in df.index:
                df.loc[name] = np.nan

    tasks = []
    feature_key = "*" if target_feature is None else f'{target_feature}'
    for tomo_name in tomo_names:
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

    already = total_segmentations - len(tasks)
    if already > 0:
        print(f"{already} already summarized, {len(tasks)} new to measure.")
    print(f"Summarizing {len(tasks)} segmentations with {workers} workers...")

    with Pool(workers) as pool:
        results = list(tqdm(pool.imap_unordered(_process_segmentation, tasks), total=len(tasks)))

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

    # With multiple tomogram sources, keep a per-source subset in sync with each directory.
    source_subsets = create_source_subsets(config)
    if source_subsets:
        print(f"Updated {len(source_subsets)} per-source subsets: " +
              ", ".join(f"{name} ({n})" for name, n in source_subsets))

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

def _sanitize_source_label(name):
    """Turn an arbitrary directory name into an identifier safe for filesystem paths
    and UI option strings (alphanumerics only, runs of anything else -> '_')."""
    label = re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    return label or 'src'

def density_dirnames(sources):
    """Map each tomogram source to the image subdir holding its central-slice ('density')
    images. The first source is the main one and keeps the bare name 'density'; every
    additional source becomes 'density_<sanitized basename>', de-duplicated with a numeric
    suffix when two sources share a basename. Returns a list parallel to `sources`."""
    names = []
    used = set()
    for i, src in enumerate(sources):
        if i == 0:
            name = 'density'
        else:
            base = _sanitize_source_label(os.path.basename(os.path.normpath(src)))
            name = f'density_{base}'
            if name in used:
                n = 2
                while f'{name}_{n}' in used:
                    n += 1
                name = f'{name}_{n}'
        used.add(name)
        names.append(name)
    return names

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

    density_dirs = density_dirnames(config['tomogram_sources'])
    for src, subdir in zip(config['tomogram_sources'], density_dirs):
        for tomo_path in glob.glob(os.path.join(src, '*.mrc')):
            tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
            output_path = os.path.join('pom', 'images', subdir, f'{tomo_name}.png')
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
        results = list(tqdm(pool.imap_unordered(_process_projection, tasks), total=len(tasks)))

def _resolve_composition_features(comp_def, sorted_features, feature_library):
    """Resolve a composition definition (feature names, 'rankN' placeholders, and '!exclude'
    items) into an ordered list of concrete feature names to render."""
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
    return features_to_render


def _bin_surface_volume(data, bin_factor):
    """Bin a segmentation volume for surface meshing. Normalizes dtype-specific scales to 0..1
    so the binned float32 volume can be marching-cubes'd with the standard 0..1 threshold,
    bypassing SurfaceModel's per-dtype level remapping (which doesn't fire on float32 input)."""
    if data.dtype == np.int8:
        data = data.astype(np.float32) / 127.0
    elif data.dtype == np.uint16:
        data = data.astype(np.float32) / 255.0
    else:
        data = data.astype(np.float32)
    return _bin_volume(data, bin_factor).astype(np.float32)


def _build_renderables(features_to_render, tomo_name, config, feature_library, surface_max_axis=None, skip_volumes=False):
    """Load segmentations and build SurfaceModel/VolumeModel renderables for one tomogram.

    surface_max_axis: if set, surface segmentations whose largest axis exceeds this are
    mean-pooled by bin_factor = ceil(max(shape) / surface_max_axis) before marching cubes.
    Cuts both build time (~bin^3 fewer voxels) and triangle count (~bin^2 fewer triangles),
    scaled per-volume so small tomograms aren't over-binned.
    skip_volumes skips VolumeModel construction (used when reusing already-built volumes).
    """
    from Pom.core.render import SurfaceModel, VolumeModel
    renderables = []
    for feature in features_to_render:
        if feature not in feature_library:
            continue

        is_volume = feature_library[feature].get('render_as', 'isosurface') == 'volume'
        if is_volume and skip_volumes:
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

        if is_volume:
            renderables.append(VolumeModel(volume, feature_library[feature], pixel_size))
        else:
            bin_factor = 1
            if surface_max_axis and max(volume.shape) > surface_max_axis:
                bin_factor = int(np.ceil(max(volume.shape) / surface_max_axis))
            if bin_factor > 1:
                volume = _bin_surface_volume(volume, bin_factor)
                pixel_size = pixel_size * bin_factor  # preserve world-space scale & dust units
            renderables.append(SurfaceModel(volume, feature_library[feature], pixel_size))
    return renderables


def _render_worker(tomo_names, df, config, feature_library, compositions, overwrite, counter, lock):
    """Worker process that creates ONE renderer and processes all assigned tomograms."""
    from Pom.core.render import Renderer, VolumeModel
    from PIL import Image

    try:
        renderer = Renderer(image_size=RENDER_IMAGE_SIZE)

        for tomo_name in tomo_names:

            if tomo_name not in df.index:
                with lock:
                    counter.value += 1
                continue

            sorted_features = df.loc[tomo_name].sort_values(ascending=False).index.tolist()

            for comp_name, comp_def in compositions.items():
                do_spin = composition_spin(comp_def)
                png_path = os.path.join('pom', 'images', comp_name, f'{tomo_name}.png')
                gif_path = os.path.join('pom', 'images', comp_name, f'{tomo_name}.gif')

                required_outputs = [png_path] + ([gif_path] if do_spin else [])
                if not overwrite and all(os.path.exists(p) for p in required_outputs):
                    continue

                features_to_render = _resolve_composition_features(composition_features(comp_def), sorted_features, feature_library)
                renderables = _build_renderables(features_to_render, tomo_name, config, feature_library)

                if renderables:
                    os.makedirs(os.path.dirname(png_path), exist_ok=True)

                    # static image - render at full PNG resolution
                    renderer.set_image_size(RENDER_IMAGE_SIZE)
                    renderer.new_image()
                    renderer.render(renderables)
                    Image.fromarray(renderer.get_image()).save(png_path)

                    # 360-degree spin movie at the smaller spin resolution. Uses coarser
                    # surface meshes (binned segmentations) to cut per-frame draw cost.
                    # Volume models are reused (their cached 3D textures stay valid).
                    if do_spin:
                        renderer.set_image_size(SPIN_RENDER_SIZE)
                        spin_surfaces = _build_renderables(
                            features_to_render, tomo_name, config, feature_library,
                            surface_max_axis=SPIN_SURFACE_MAX_AXIS, skip_volumes=True,
                        )
                        spin_renderables = [r for r in renderables if isinstance(r, VolumeModel)] + spin_surfaces
                        _render_spin_gif(renderer, spin_renderables, gif_path)
                        for r in spin_surfaces:
                            r.delete()

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

    if not config.get('segmentation_sources'):
        print("no segmentation source found - skipping rendering.")
        return

    df = starfile.read(os.path.join('pom', 'summary.star'), parse_as_string=["tomogram"])
    df = df.set_index('tomogram')

    if 'tomogram' in df.columns:
        df = df.set_index('tomogram')

    for comp_name in compositions.keys():
        os.makedirs(os.path.join('pom', 'images', comp_name), exist_ok=True)

    tomograms, n_files, max_flavours = collect_tomogram_names(config)

    parallel_processes = min(os.cpu_count(), 16)
    print(f"Rendering {describe_tomogram_count(len(tomograms), n_files, max_flavours)} in {len(compositions)} {'composition' if len(compositions) == 1 else 'compositions'} with {parallel_processes} workers...")

    spinning = [name for name, comp in compositions.items() if composition_spin(comp)]
    if spinning:
        comp_path = os.path.abspath(os.path.join('pom', 'image_compositions.json'))
        print(f"  Note: {len(spinning)} composition(s) render a spin movie ({', '.join(spinning)}) - this is much slower.")
        print(f"  If it is too slow, uncheck 'Spin (GIF)' in the Visualization settings tab of the Pom app, or edit {comp_path}.")

    # Divide tomograms among processes
    process_div = {p: [] for p in range(parallel_processes)}
    for p, tomo_name in zip(itertools.cycle(range(parallel_processes)), tomograms):
        process_div[p].append(tomo_name)

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

def _render_spin_gif(renderer, renderables, output_path, frames=SPIN_FRAMES, fps=SPIN_FPS):
    """Render a 360-degree spin movie (animated GIF) of already-built renderables.

    The camera yaw sweeps a full turn over `frames` frames; the original yaw is restored
    afterwards so subsequent static renders are unaffected.
    """
    from PIL import Image

    base_yaw = renderer.camera.yaw
    images = []
    for i in range(frames):
        renderer.camera.yaw = base_yaw + 360.0 * i / frames
        renderer.camera.on_update()
        renderer.new_image()
        renderer.render(renderables)
        images.append(Image.fromarray(renderer.get_image()))

    renderer.camera.yaw = base_yaw
    renderer.camera.on_update()

    duration_ms = int(round(1000.0 / fps))
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, disposal=2, optimize=True)


def _bin_volume(vol, b):
    """Bin a 3D volume by factor b using mean pooling. Trims edges to fit."""
    if b <= 1:
        return vol
    d, h, w = vol.shape
    d, h, w = (d // b) * b, (h // b) * b, (w // b) * b
    return vol[:d, :h, :w].reshape(d // b, b, h // b, b, w // b, b).mean(axis=(1, 3, 5))


def _remove_dust(binary, dust_value, apix):
    """Remove connected components from a binary volume.

    dust_value > 0: drop components smaller than dust_value cubic Angstrom.
    dust_value < 0: keep only the -dust_value largest components.
    dust_value == 0: no-op."""
    if dust_value == 0.0 or not binary.any():
        return binary
    labeled, n = ndimage.label(binary)
    if n == 0:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    out = binary.copy()
    if dust_value < 0:
        keep_n = int(-dust_value)
        if keep_n < n:
            threshold_size = sorted(sizes, reverse=True)[keep_n - 1]
            for i, sz in enumerate(sizes):
                if sz < threshold_size:
                    out[labeled == (i + 1)] = False
    else:
        min_voxels = int(np.ceil(dust_value / (apix ** 3)))
        for i, sz in enumerate(sizes):
            if sz < min_voxels:
                out[labeled == (i + 1)] = False
    return out


def _build_sampler_mask(sampler, tomo_name, apix):
    """Build a boolean mask volume for one sampler applied to one tomogram.

    sampler: (feature, sigma_A, threshold, dust, subtract) -- gaussian-smooth the segmentation
    by sigma_A (Angstrom, 0 = no smoothing), threshold at `threshold` (0..1), optionally remove
    per-feature dust (positive A^3 = min component size, negative N = keep N largest).
    Returns None if the segmentation file is missing."""
    feature, sigma_A, threshold, dust, _subtract = sampler
    seg_path = get_segmentation_for_tomogram(tomo_name, feature)
    if seg_path is None:
        return None
    seg = mrcfile.read(seg_path)
    norm = 127 if seg.dtype == np.int8 else 255 if seg.dtype == np.uint16 else 1.0
    seg_f = seg.astype(np.float32) / norm

    if sigma_A > 0:
        sigma_vox = sigma_A / max(apix, 1.0)
        seg_f = ndimage.gaussian_filter(seg_f, sigma=sigma_vox)

    binary = seg_f >= threshold
    if dust != 0.0:
        binary = _remove_dust(binary, dust, apix)
    return binary


def _create_mask_for_tomogram(tomo_name, samplers_parsed, final_dust=0.0):
    """Assemble the per-tomogram mask. Returns (mask_bool_array, apix) or (None, apix)."""
    tomogram_path = get_tomogram_by_name(tomo_name)
    apix = None
    if tomogram_path is not None:
        with mrcfile.open(tomogram_path, header_only=True, permissive=True) as mrc:
            apix = float(mrc.voxel_size.x)

    mask = None
    for sampler in samplers_parsed:
        # Need apix to translate Angstrom to voxels; fall back to segmentation apix if needed.
        if apix is None or apix == 0:
            seg_path = get_segmentation_for_tomogram(tomo_name, sampler[1])
            if seg_path is None:
                continue
            with mrcfile.open(seg_path, header_only=True, permissive=True) as mrc:
                apix = float(mrc.voxel_size.x) or 1.0

        m = _build_sampler_mask(sampler, tomo_name, apix)
        if m is None:
            continue
        if mask is None:
            mask = np.zeros(m.shape, dtype=bool)
        if m.shape != mask.shape:
            print(f"  {tomo_name}: skipping {sampler[1]} - shape {m.shape} != mask shape {mask.shape}")
            continue
        if sampler[4]:  # subtract
            mask &= ~m
        else:
            mask |= m

    if mask is not None and final_dust != 0.0:
        mask = _remove_dust(mask, final_dust, apix or 1.0)

    return mask, apix


def _create_mask_worker(tomo_batch, name, samplers_parsed, output_dir, final_dust, overwrite, counter, lock):
    for tomo_name in tomo_batch:
        try:
            output_path = os.path.join(output_dir, f'{tomo_name}__{name}.mrc')
            if not overwrite and os.path.exists(output_path):
                continue
            mask, apix = _create_mask_for_tomogram(tomo_name, samplers_parsed, final_dust=final_dust)
            if mask is None or not mask.any():
                continue
            # Save as int8 (0 / 127) matching Ais' segmentation convention so the masks
            # directory can be added as a segmentation source and reused.
            data = (mask.astype(np.int8) * 127)
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(data)
                if apix and apix > 0:
                    mrc.voxel_size = apix
        except Exception as e:
            print(f'Error processing {tomo_name}: {e}')
        finally:
            with lock:
                counter.value += 1


def _resolve_subset_or_tomogram(arg):
    """Resolve a --subset value: if pom/subsets/<arg>.txt exists, return its tomogram
    names (parsed from path-or-name entries). Otherwise treat arg as a single tomo name."""
    subset_path = os.path.join('pom', 'subsets', f'{arg}.txt')
    if os.path.isfile(subset_path):
        names = []
        with open(subset_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                base = line.replace('\\', '/').rsplit('/', 1)[-1]
                if base.endswith('.mrc'):
                    base = base[:-4]
                names.append(base)
        return names, True
    return [arg], False


def create_mask(name, samplers, output_dir=None, dust=0.0, subset=None, workers=None, overwrite=False):
    """Save a binary mask per tomogram, derived from segmentation features via samplers."""
    import itertools, multiprocessing, time

    # Parse samplers (with optional ! prefix for subtraction).
    # Form: feature:sigma:threshold or feature:sigma:threshold:dust
    samplers_parsed = []
    for s in samplers:
        subtract = s.startswith('!')
        body = s[1:] if subtract else s
        parts = body.split(':')
        try:
            if len(parts) == 3:
                samplers_parsed.append((parts[0], float(parts[1]), float(parts[2]), 0.0, subtract))
            elif len(parts) == 4:
                samplers_parsed.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3]), subtract))
            else:
                print(f'Invalid sampler "{s}": expected feature:sigma:threshold or feature:sigma:threshold:dust')
                return
        except ValueError as e:
            print(f'Could not parse sampler "{s}": {e}')
            return

    if not samplers_parsed:
        print('No valid samplers.')
        return

    config = get_config()
    if not config.get('segmentation_sources'):
        print('No segmentation source configured. Run "pom add_source -s <path>" first.')
        return

    if output_dir is None:
        output_dir = 'masks'
    os.makedirs(output_dir, exist_ok=True)

    subset_source = None
    if subset:
        tomograms, from_subset_file = _resolve_subset_or_tomogram(subset)
        subset_source = f"subset '{subset}'" if from_subset_file else f"tomogram '{subset}'"
    else:
        tomos = set()
        for src in config.get('tomogram_sources', []):
            for p in glob.glob(os.path.join(src, '*.mrc')):
                tomos.add(os.path.splitext(os.path.basename(p))[0])
        # Also include any tomograms that only show up in segmentation sources.
        for src in config['segmentation_sources']:
            for p in glob.glob(os.path.join(src, '*__*.mrc')):
                tomos.add(os.path.basename(p).split('__', 1)[0])
        tomograms = sorted(tomos)

    if not tomograms:
        print('No tomograms found.')
        return

    scope = f" ({subset_source})" if subset_source else ""
    print(f"\nCreating mask '{name}' for {len(tomograms)} tomogram(s){scope} with {len(samplers_parsed)} sampler(s):")
    for i, s in enumerate(samplers_parsed, 1):
        feature, sigma_A, threshold, dust_f, subtract = s
        op = 'SUBTRACT' if subtract else 'include'
        sigma_str = f", σ={sigma_A:.1f} Å" if sigma_A > 0 else ""
        dust_str = ''
        if dust_f != 0.0:
            dust_str = f", keep {int(-dust_f)} largest" if dust_f < 0 else f", drop blobs < {dust_f:.2e} Å³"
        print(f"  {i}. {op}: {feature} >= {threshold}{sigma_str}{dust_str}")
    if dust != 0.0:
        if dust < 0:
            print(f"  Final: keep {int(-dust)} largest connected component(s)")
        else:
            print(f"  Final: drop components < {dust:.2e} Å³")
    print(f"  Output: {output_dir}/<tomo>__{name}.mrc\n")

    parallel_processes = workers if workers is not None else min(os.cpu_count(), 16)
    process_div = {p: [] for p in range(parallel_processes)}
    for p, tomo in zip(itertools.cycle(range(parallel_processes)), tomograms):
        process_div[p].append(tomo)

    manager = multiprocessing.Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    print(f'Processing with {parallel_processes} workers...')
    with multiprocessing.Pool(processes=parallel_processes) as pool:
        async_results = [
            pool.apply_async(
                _create_mask_worker,
                (process_div[p], name, samplers_parsed, output_dir, dust, overwrite, counter, lock),
            )
            for p in range(parallel_processes)
        ]
        with tqdm(total=len(tomograms)) as pbar:
            last_value = 0
            while any(not r.ready() for r in async_results):
                current_value = counter.value
                if current_value > last_value:
                    pbar.update(current_value - last_value)
                    last_value = current_value
                time.sleep(0.1)
            current_value = counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)
        for r in async_results:
            r.get()

    print('Mask creation complete.')


def _distance_to_surface(segmentation_volume, normalization_value, threshold, dust_angstrom3, apix, coordinates, binning=1):
    if binning > 1:
        vol = _bin_volume(segmentation_volume.astype(np.float32), binning)
        vol = vol / normalization_value
        binned_apix = apix * binning
    else:
        vol = segmentation_volume.astype(np.float32) / normalization_value
        binned_apix = apix

    binary = vol >= threshold

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
        min_voxels = int(np.ceil(dust_angstrom3 / (binned_apix ** 3)))
        for i, sz in enumerate(sizes):
            if sz < min_voxels:
                binary[labeled == (i + 1)] = False

    if not binary.any():
        return np.full(len(coordinates), np.nan, dtype=np.float32)

    edt_out = ndimage.distance_transform_edt(~binary) * binned_apix
    edt_in = ndimage.distance_transform_edt(binary) * binned_apix
    dist_vol = np.where(binary, -edt_in, edt_out)

    distances = np.full(len(coordinates), np.nan, dtype=np.float32)
    for j, (cj, ck, cl) in enumerate(coordinates):
        c = np.round(np.array([cj, ck, cl]) / binning).astype(int) if binning > 1 else np.round([cj, ck, cl]).astype(int)
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


def _contextualize_job(df, samplers_parsed, tomogram_name, coords_angpix, binning=1):
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

    has_offset = any((s[0] == 'sphere' and s[3] != 0.0) or (s[0] == 'distance' and s[4] != 0.0) for s in samplers_parsed)
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
            _, feature, threshold, dust, offset = sampler
            column_name = f"pomDist{feature.replace('_', ' ').title().replace(' ', '')}T{int(threshold*100)}"
            if offset != 0.0:
                sign = 'p' if offset > 0 else 'm'
                column_name += f"{sign}{int(round(abs(offset)))}"
            segmentation_path = get_segmentation_for_tomogram(tomogram_name, feature)
            if segmentation_path is None:
                df[column_name] = np.nan
                continue

            segmentation_volume = mrcfile.read(segmentation_path)
            normalization_value = 127 if segmentation_volume.dtype == np.int8 else 255 if segmentation_volume.dtype == np.uint16 else 1.0
            if offset != 0.0 and axes is not None:
                offset_coordinates = coordinates + axes * (offset / apix)
            else:
                offset_coordinates = coordinates
            df[column_name] = _distance_to_surface(segmentation_volume, normalization_value, threshold, dust, apix, offset_coordinates, binning=binning)

    return df


def _contextualize_worker(tomo_batch, samplers_parsed, df, tomo_col, coords_angpix, counter, lock, binning=1):
    results = {}
    for df_tomo_name, mrc_tomo_name in tomo_batch:
        df_subset = df[df[tomo_col] == df_tomo_name].copy()
        results[df_tomo_name] = _contextualize_job(df_subset, samplers_parsed, mrc_tomo_name, coords_angpix, binning=binning)
        with lock:
            counter.value += 1
    return results


def contextualize_starfile(star_path, samplers, tomogram_name=None, substitutions=None, out_star=None, coords_angpix=None, binning=1, workers=None):
    import itertools, multiprocessing, time
    star_data = starfile.read(star_path)
    if isinstance(star_data, dict):
        df = star_data['particles']
    else:
        df = star_data

    print()
    samplers_parsed = []
    for s in samplers:
        parts = s.split(':')
        if len(parts) == 2:
            # feature:radius -> sphere sampler
            samplers_parsed.append(('sphere', parts[0], float(parts[1]), 0.0))
        elif len(parts) == 3:
            # Disambiguate sphere (feature:radius:offset) vs distance (feature:threshold:dust)
            # Sphere radius is large (>1), distance threshold is 0.0-1.0
            if float(parts[1]) > 1.0:
                samplers_parsed.append(('sphere', parts[0], float(parts[1]), float(parts[2])))
            else:
                samplers_parsed.append(('distance', parts[0], float(parts[1]), float(parts[2]), 0.0))
        elif len(parts) == 4:
            samplers_parsed.append(('distance', parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
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
            _, feature, threshold, dust, offset = s
            col = f"pomDist{feature.replace('_', ' ').title().replace(' ', '')}T{int(threshold*100)}"
            if offset != 0.0:
                sign = 'p' if offset > 0 else 'm'
                col += f"{sign}{int(round(abs(offset)))}"
            if dust < 0:
                dust_desc = f"keeping only {int(-dust)} largest blob(s)"
            else:
                dust_desc = f"ignoring blobs < {dust:.2e} cubic Å"
            offset_str = f", offset {offset:+.0f} Å along primary axis" if offset != 0.0 else ""
            print(f'\t{i}. {col}: distance to {feature} (threshold={threshold}, {dust_desc}{offset_str}).')
    print()

    seg_sources = get_config().get('segmentation_sources', [])
    if not seg_sources:
        print(f'\033[33mNo segmentation sources configured. Run "pom add_source" first.\033[0m')
        return

    skip_features = set()
    for f in set(s[1] for s in samplers_parsed):
        n_feature_volumes = 0
        for src in seg_sources:
            pattern = os.path.join(src, f'*__{f}.mrc')
            n_feature_volumes += len(glob.glob(pattern))
        if n_feature_volumes == 0:
            print(f'\033[33mNo segmentation volumes found for feature "{f}" — skipping this sampler.\033[0m')
            skip_features.add(f)
        else:
            print(f'found {n_feature_volumes} volumes for feature "{f}".')
    samplers_parsed = [s for s in samplers_parsed if s[1] not in skip_features]
    if not samplers_parsed:
        print(f'\033[33mNo valid samplers remaining. Aborting.\033[0m')
        return
    print()

    if out_star is not None:
        out_dir = os.path.dirname(out_star)
        if out_dir and not os.path.isdir(out_dir):
            print(f'\033[33mOutput directory "{out_dir}" does not exist. Aborting.\033[0m')
            return

    tomo_col = tomogram_name or ("rlnMicrographName" if "rlnMicrographName" in df.columns else "wrpSourceName")
    if tomo_col not in df.columns:
        print(f'No column "{tomo_col}" found in {star_path}. Aborting.')
        return

    parallel_processes = workers if workers is not None else min(os.cpu_count(), 32)
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
                                          (process_div[p], samplers_parsed, df, tomo_col, coords_angpix, counter, lock, binning))
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

    if isinstance(star_data, dict):
        star_data['particles'] = output_df
        output_data = star_data
    else:
        output_data = output_df

    if out_star is None:
        starfile.write(output_data, star_path)
    else:
        if not '.star' in out_star:
            out_star += '.star'
        starfile.write(output_data, out_star)