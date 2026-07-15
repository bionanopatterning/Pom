import starfile, os, glob
from PIL import Image, ImageDraw

def load_data():
    df = starfile.read(os.path.join('pom', 'summary.star'), parse_as_string=["tomogram"])
    df = df.set_index('tomogram')
    # Guard against pre-fix summary.star files that duplicated each tomogram once per
    # flavour source: a duplicate index makes df.loc[names] cross-join (e.g. 4x4=16 tiles).
    df = df[~df.index.duplicated(keep='first')]
    return df


def _placeholder_image():
    img = Image.new("RGB", (256, 256), (200, 200, 200))
    draw = ImageDraw.Draw(img)

    for y in range(0, 256, 16):
        for x in range(0, 256, 16):
            if (x // 16 + y // 16) % 2 == 0:
                draw.rectangle([x, y, x + 15, y + 15], fill=(240, 240, 240))

    return img

def density_options():
    """List the available central-slice ('density') flavours, one per tomogram source,
    by scanning the image directories actually produced by `pom projections`. The main
    source's 'density' is always first; additional flavours ('density_<name>') follow,
    sorted. Returns ['density'] as a fallback when nothing has been generated yet."""
    base = os.path.join('pom', 'images')
    opts = []
    if os.path.isdir(os.path.join(base, 'density')):
        opts.append('density')
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            if name.startswith('density_') and os.path.isdir(os.path.join(base, name)):
                opts.append(name)
    return opts or ['density']

# ----------------------------------------------------------------------------------
# Subset helpers
#
# A subset is a plain-text file 'pom/subsets/<name>.txt', one tomogram per line. Lines
# are stored as full paths to the .mrc (so a subset survives across flavour sources),
# but a tomogram's identity is its bare filename, which `subset_tomo_name` extracts.
# ----------------------------------------------------------------------------------

def subsets_dir():
    return os.path.join('pom', 'subsets')

def subset_path(name):
    return os.path.join(subsets_dir(), f'{name}.txt')

def list_subsets():
    return sorted(os.path.splitext(os.path.basename(j))[0]
                  for j in glob.glob(os.path.join(subsets_dir(), '*.txt')))

def subset_tomo_name(entry):
    """Return the bare tomogram name from a subset entry (full path or plain name)."""
    base = entry.replace('\\', '/').rsplit('/', 1)[-1]
    return base[:-4] if base.endswith('.mrc') else base

def _migrate_subset_to_full_paths(path, entries):
    # backwards compatibility (260331): rewrite plain tomo names to full paths on first read
    if not any('/' not in e and '\\' not in e for e in entries):
        return entries
    from Pom.core.tools import get_tomogram_by_name
    migrated = [(get_tomogram_by_name(e) or e) if ('/' not in e and '\\' not in e) else e
                for e in entries]
    with open(path, 'w') as f:
        f.write('\n'.join(migrated) + '\n')
    return migrated

def read_subset(subset):
    """Return the raw entries (full paths) stored in a subset, [] if it does not exist."""
    path = subset_path(subset)
    if not os.path.exists(path):
        return []
    entries = [t for t in open(path).read().splitlines() if t.strip()]
    return _migrate_subset_to_full_paths(path, entries)

def read_subset_names(subset):
    """Return the bare tomogram names contained in a subset."""
    return [subset_tomo_name(e) for e in read_subset(subset)]

def write_subset(subset, entries):
    os.makedirs(subsets_dir(), exist_ok=True)
    with open(subset_path(subset), 'w') as f:
        f.write('\n'.join(entries) + ('\n' if entries else ''))

def add_to_subset(subset, tomo):
    from Pom.core.tools import get_tomogram_by_name
    full_path = get_tomogram_by_name(tomo) or tomo
    entries = read_subset(subset)
    if tomo not in [subset_tomo_name(t) for t in entries]:
        entries.append(full_path)
        write_subset(subset, entries)

def remove_from_subset(subset, tomo):
    if not os.path.exists(subset_path(subset)):
        return
    entries = [t for t in read_subset(subset) if subset_tomo_name(t) != tomo]
    write_subset(subset, entries)

def delete_subset(subset):
    path = subset_path(subset)
    if os.path.exists(path):
        os.remove(path)

def get_image(tomo_name, feature):
    if feature == 'density':
        img_path = os.path.join('pom', 'images', 'density', f'{tomo_name}.png')
    else:
        # A spin composition writes an animated GIF alongside the PNG; prefer it and
        # return the path (st.image animates a file path/bytes, but not a PIL Image).
        gif_path = os.path.join('pom', 'images', feature, f'{tomo_name}.gif')
        if os.path.exists(gif_path):
            return gif_path
        img_path = os.path.join('pom', 'images', feature, f'{tomo_name}.png')
        if not os.path.exists(img_path):
            img_path = os.path.join('pom', 'images', f'{feature}_projection', f'{tomo_name}.png')

    if os.path.exists(img_path):
        try:
            return Image.open(img_path)
        except Exception:
            return _placeholder_image()
    else:
        return _placeholder_image()