import os
import json
import shutil
from PIL import Image
import numpy as np

root = os.path.dirname(os.path.dirname(__file__))

project_configuration_json_path = os.path.join(os.getcwd(), "project_configuration.json")
if not os.path.exists(project_configuration_json_path):
    shutil.copy(os.path.join(os.path.dirname(__file__), "project_configuration.json"), project_configuration_json_path)
    print(f"created project_configuration.json at {project_configuration_json_path}")
    shutil.copy(os.path.join(os.path.dirname(__file__), "render_configuration.json"), os.path.join(os.getcwd(), "render_configuration.json"))
    print(f"created render_configuration.json at {os.path.join(os.getcwd(), 'render_configuration.json')}")

def get_image(tomo, image, projection=False):
    image_dir = image.split("_")[0]
    if projection:
        img_path = os.path.join(project_configuration["root"], project_configuration["image_dir"], f"{image_dir}_projection", f"{tomo}_{image}.png")
    else:
        img_path = os.path.join(project_configuration["root"], project_configuration["image_dir"], image_dir, f"{tomo}_{image}.png")
    if os.path.exists(img_path):
        return Image.open(img_path)
    else:
        return Image.fromarray(np.zeros((128, 128)), mode='L')


def recolor(color, style=0):
    if style == 0:
        c = (np.array(color) / 2.0 + 0.5)
    elif style == 1:
        c = (np.array(color) / 8 + 0.875)
    else:
        c = color
    return np.clip(c, 0.0, 1.0)


def load_config():
    with open(os.path.join(os.getcwd(), "project_configuration.json"), 'r') as f:
        project_configuration = json.load(f)
    return project_configuration


def parse_feature_library(feature_library_path):
    if not os.path.exists(feature_library_path):
        return dict()
    with open(feature_library_path, 'r') as f:
        flist = json.load(f)
    feature_library = dict()
    for f in flist:
        feature_definition = FeatureLibraryFeature.from_dict(f)
        feature_library[feature_definition.title] = feature_definition
    return feature_library


class FeatureLibraryFeature:
    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 255 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]
    CLR_COUNTER = 0
    SORT_TITLE = "\n"

    def __init__(self):
        self.title = "New feature"
        self.colour = FeatureLibraryFeature.DEFAULT_COLOURS[FeatureLibraryFeature.CLR_COUNTER % len(FeatureLibraryFeature.DEFAULT_COLOURS)]
        self.box_size = 64
        self.brush_size = 10.0  # nm
        self.alpha = 1.0
        self.use = True
        self.dust = 1.0
        self.level = 128
        self.render_alpha = 1.0
        self.hide = False
        FeatureLibraryFeature.CLR_COUNTER += 1

    def to_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(data):
        ret = FeatureLibraryFeature()
        ret.title = data['title']
        ret.colour = data['colour']
        ret.box_size = data['box_size']
        ret.brush_size = data['brush_size']
        ret.alpha = data['alpha']
        ret.use = data['use']
        ret.dust = data['dust']
        ret.level = data['level']
        ret.render_alpha = data['render_alpha']
        ret.hide = data['hide']
        return ret


project_configuration = load_config()
feature_library = parse_feature_library(os.path.join(os.path.expanduser("~"), ".Ais", "feature_library.txt"))
for f in project_configuration["ontologies"] + project_configuration["macromolecules"] + ["Unknown"]:
    if f not in feature_library:
        feature_library[f] = FeatureLibraryFeature()
        feature_library[f].title = f
