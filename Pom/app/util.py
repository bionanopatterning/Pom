import starfile, os, glob
from PIL import Image, ImageDraw

def load_data():
    df = starfile.read(os.path.join('pom', 'summary.star'), parse_as_string=["tomogram"])
    df = df.set_index('tomogram')
    return df


def _placeholder_image():
    img = Image.new("RGB", (256, 256), (200, 200, 200))
    draw = ImageDraw.Draw(img)

    for y in range(0, 256, 16):
        for x in range(0, 256, 16):
            if (x // 16 + y // 16) % 2 == 0:
                draw.rectangle([x, y, x + 15, y + 15], fill=(240, 240, 240))

    return img

def get_image(tomo_name, feature):
    if feature == 'density':
        img_path = os.path.join('pom', 'images', 'density', f'{tomo_name}.png')
    else:
        # Try composition first, then projection
        img_path = os.path.join('pom', 'images', feature, f'{tomo_name}.png')
        if not os.path.exists(img_path):
            img_path = os.path.join('pom', 'images', f'{feature}_projection', f'{tomo_name}.png')

    if os.path.exists(img_path):
        return Image.open(img_path)
    else:
        return _placeholder_image()
