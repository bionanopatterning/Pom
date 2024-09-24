import os
import glob
import pickle
import mrcfile
import tifffile
import time
import json
from Pom.core.util import *
import copy

project_configuration = dict()

def phase_1_initialize():
    from Ais.core.se_frame import SEFrame
    from Ais.main import windowless

    os.makedirs(os.path.join(project_configuration['root'], "training_datasets"), exist_ok=True)


    windowless()

    data = dict()
    for o in project_configuration['ontologies']:
        data[o] = dict()
        data[o]['y'] = list()
        for m in project_configuration['macromolecules']:
            data[o][m] = list()

    annotated_datasets = glob.glob(os.path.join(project_configuration['root'], project_configuration['tomogram_dir'], "*.scns"))
    for j, scns in enumerate(annotated_datasets):
        print(f"Annotated dataset {j}/{len(annotated_datasets)}: {os.path.basename(scns)}")
        with open(scns, 'rb') as pf:
            se_frame = pickle.load(pf)

        tomo_name = os.path.splitext(os.path.basename(scns))[0]
        macromolecules = dict()
        for m in macromolecules:
            if m == "Density":
                path = os.path.join(project_configuration['root'], project_configuration['tomogram_dir'], tomo_name + f".mrc")
            else:
                path = os.path.join(project_configuration['root'], project_configuration["macromolecule_dir"], tomo_name + f"__{m}.mrc")
            macromolecules[m] = mrcfile.mmap(path)

        for feature in se_frame.features:
            o = feature.title
            if o not in project_configuration['ontologies']:
                continue
            print(f"\t{o}")
            for z in feature.boxes.keys():
                for (j, k) in feature.boxes[z]:
                    j_min = (j - project_configuration['ontology_annotation_box_size'] // 2)
                    j_max = (j + project_configuration['ontology_annotation_box_size'] // 2)
                    k_min = (k - project_configuration['ontology_annotation_box_size'] // 2)
                    k_max = (k + project_configuration['ontology_annotation_box_size'] // 2)
                    if z in feature.slices and feature.slices[z] is not None and (z > project_configuration['z_sum']) and (z < se_frame.n_slices - project_configuration['z_sum']):
                        annotation = feature.slices[z][k_min:k_max, j_min:j_max]
                        if annotation.shape == (project_configuration['ontology_annotation_box_size'], project_configuration['ontology_annotation_box_size']):
                            data[o]['y'].append(bin_img(annotation, 2))
                            for m in macromolecules:
                                m_vol = macromolecules[m].data[z - project_configuration['z_sum']:z + project_configuration['z_sum'] + 1, k_min:k_max, j_min:j_max]
                                m_img = bin_img(m_vol.mean(0), 2)
                                if m == "Density":
                                    m_img -= np.mean(m_img)
                                    _std = np.std(m_img)
                                    if _std != 0.0:
                                        m_img /= np.std(m_img)
                                else:
                                    m_img = m_img / 255.0 * 2.0 - 1.0
                                data[o][m].append(m_img)

    # Save the data.
    for o in data:
        for m in data[o]:
            dataset = np.array(data[o][m])
            tifffile.imwrite(os.path.join(project_configuration['root'], "training_datasets", f"{o}_{m}.tif"), dataset)

    print(f"Training datasets generated and saved to:\n\t{os.path.join(project_configuration['root'], 'training_datasets')}")


def phase_1_train(gpus, ontology):
    import tensorflow as tf
    import keras.callbacks
    from Pom.models.vggnet import create_model

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    os.makedirs(os.path.join(project_configuration["root"], "models", "phase1"), exist_ok=True)
    os.makedirs(os.path.join(project_configuration["root"], "training_datasets", "phase1"), exist_ok=True)

    def add_redundancy(data):
        data_out = list()
        for img in data:
            for k in [0, 1, 2, 3]:
                _img = np.rot90(img, k, axes=(0, 1))
                data_out.append(_img)
                data_out.append(np.flip(_img, axis=0))
        return np.array(data_out)

    data_y = add_redundancy(tifffile.imread(os.path.join(project_configuration['root'], "training_datasets", "phase1", f"{ontology}_y.tif")))
    data_x = np.zeros((*data_y.shape, len(project_configuration['macromolecules'])))

    for j, m in enumerate(project_configuration['macromolecules']):
        data_x[:, :, :, j] = add_redundancy(tifffile.imread(os.path.join(project_configuration['root'], "training_datasets", "phase1", f"{ontology}_{m}.tif")))

    def tf_data_generator():
        n_samples = data_x.shape[0]
        indices = np.arange(n_samples)

        while True:
            np.random.shuffle(indices)
            for i in range(0, n_samples, project_configuration['single_model_batch_size']):
                batch_indices = indices[i:i + project_configuration['single_model_batch_size']]
                batch_x = data_x[batch_indices]
                batch_y = data_y[batch_indices]
                yield batch_x, batch_y

    training_data = tf.data.Dataset.from_generator(tf_data_generator, output_signature=(tf.TensorSpec(shape=(None, data_x.shape[1], data_x.shape[2], data_x.shape[3]), dtype=tf.float32), tf.TensorSpec(shape=(None, data_y.shape[1], data_y.shape[2]), dtype=tf.float32)))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    training_data = training_data.with_options(options)

    strategy = tf.distribute.get_strategy() if not project_configuration['tf_distribute_mirrored'] else tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(data_x.shape[1:])

    print(f"Training a model for {ontology} with inputs:")
    [print(f"\t{m}") for m in project_configuration['macromolecules']]

    checkpoint_callback = keras.callbacks.ModelCheckpoint( filepath=os.path.join(project_configuration['root'], "models", "phase1", f"{ontology}_checkpoint.h5"), monitor='loss', mode='min', save_best_only=True)
    model.fit(training_data, epochs=project_configuration['single_model_epochs'], steps_per_epoch=data_x.shape[0] // project_configuration['single_model_batch_size'], shuffle=True, callbacks=[checkpoint_callback])
    model.save(os.path.join(project_configuration["root"], "models", "phase1", f"{ontology}.h5"))


def phase_1_test(gpus, ontology):
    import tensorflow as tf
    import multiprocessing
    import itertools
    from scipy.ndimage import convolve1d

    def preprocess_volume(vol):
        return bin_vol(convolve1d(vol, np.ones(project_configuration["z_sum"] + 1) / (project_configuration["z_sum"] + 1), axis=0, mode='nearest'), 2)

    def segment_tomo(tomo, model):
        components = dict()

        tomo_name = os.path.splitext(os.path.basename(tomo))[0]
        for m in project_configuration["macromolecules"]:
            if m == 'Density':
                components[m] = preprocess_volume(mrcfile.read(os.path.join(project_configuration["root"], "full_dataset", f"{tomo_name}.mrc")))
            else:
                components[m] = preprocess_volume(mrcfile.read(os.path.join(project_configuration["root"], "macromolecules", f"{tomo_name}__{m}.mrc"))) / 255.0 * 2.0 - 1.0

        data_x = np.zeros((*components[project_configuration["macromolecules"][0]].shape, len(project_configuration["macromolecules"])), dtype=np.float32)
        for j, m in enumerate(components):
            data_x[:, :, :, j] = components[m]

        data_y = np.zeros(data_x.shape[0:3], dtype=np.float32)

        normalize = [m == 'Density' for m in project_configuration["macromolecules"]]
        for j in range(data_y.shape[0]):
            boxes, imgsize, padding, stride = image_to_boxes(data_x[j, :, :, :], boxsize=64, overlap=project_configuration["single_model_overlap"], normalize=normalize)
            boxes = model.predict(boxes)
            data_y[j, :, :] = np.squeeze(boxes_to_image(boxes, imgsize, padding, stride))
        return data_y

    def _thread(model_path, tomogram_paths, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        model = tf.keras.models.load_model(model_path, compile=False)

        for tomo in tomogram_paths:
            t_start = time.time()
            try:
                out_name = os.path.join(project_configuration["root"], project_configuration["test_dir"], os.path.basename(os.path.splitext(tomo)[0])+f"__{ontology}.mrc")
                vol_out = segment_tomo(tomo, model)
                with mrcfile.new(out_name, overwrite=True) as f:
                    f.set_data(vol_out.astype(np.float32))
                print(f"{tomogram_paths.index(tomo)+1}/{len(tomogram_paths)}: {ontology} cost: {time.time() - t_start:.1f} seconds.")
            except Exception as e:
                print(e)

    all_tomos = [p for p in glob.glob(os.path.join(project_configuration["root"], project_configuration["test_dir"], "*.mrc")) if not '__' in os.path.basename(p)]

    _gpus = [int(j) for j in gpus.split(",")]
    data_div = {gpu: list() for gpu in _gpus}
    for gpu, tomo_path in zip(itertools.cycle(_gpus), all_tomos):
        data_div[gpu].append(tomo_path)

    processes = []
    for gpu_id in data_div:
        p = multiprocessing.Process(target=_thread, args=(os.path.join(project_configuration["root"], "models", "phase1", f"{ontology}.h5"), data_div[gpu_id], gpu_id))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def phase_2_initialize():
    import tensorflow as tf

    os.makedirs(os.path.join(project_configuration['root'], "training_datasets", "phase2"), exist_ok=True)

    def model_predict_with_redundancy(model, data_x):
        prediction = np.zeros((data_x.shape[0], data_x.shape[1], data_x.shape[2], 1))

        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        for k in [0, 1, 2, 3]:
            d = np.rot90(data_x, k=k, axes=(1, 2))
            _p = model.predict(d)
            p = np.rot90(_p, k=-k, axes=(1, 2))
            prediction += p

            d = np.flip(d, axis=2)
            _p = model.predict(d)
            p = np.rot90(np.flip(_p, axis=2), k=-k, axes=(1,2))
            prediction += p
        prediction /= 8
        prediction = np.squeeze(prediction)
        return prediction

    macromolecule_inputs = project_configuration["macromolecules"]
    ontologies = project_configuration["ontologies"]

    data_x = list()
    data_y = list()
    data_m = list()
    for j, o in enumerate(ontologies):
        o_data_y = tifffile.imread(os.path.join(project_configuration["root"], "training_datasets", "phase1", f"{o}_y.tif"))
        o_data_x = np.zeros((*o_data_y.shape, len(macromolecule_inputs)))
        o_data_m = np.zeros((o_data_y.shape[0])) + j
        for k, m in enumerate(macromolecule_inputs):
            o_data_x[:, :, :, k] = tifffile.imread(os.path.join(project_configuration["root"], "training_datasets", "phase1", f"{o}_{m}.tif"))

        data_x.append(o_data_x)
        data_y.append(o_data_y)
        data_m.append(o_data_m)

    data_x = np.concatenate(data_x, axis=0)
    data_y = np.concatenate(data_y, axis=0)
    data_m = np.concatenate(data_m, axis=0)

    model_outputs = dict()
    for o in ontologies:
        print(f"Applying model {o} to training data.")
        model = tf.keras.models.load_model(os.path.join(project_configuration["root"], "models", "phase1", f"{o}.h5"), compile=False)
        model_outputs[o] = model_predict_with_redundancy(model, data_x)

    # Now parse into a joint dataset.
    for j, m in enumerate(macromolecule_inputs):
        out_tif = np.squeeze(data_x[:, :, :, j]).astype(np.float32)
        tifffile.imwrite(os.path.join(project_configuration["root"], "training_datasets", "phase2", f"in_{m}.tif"), out_tif)

    new_data_y = np.zeros((*data_y.shape, len(ontologies)+1))
    for j, o in enumerate(ontologies):
        print(f"Parsing new output for {o}")
        model_y = model_outputs[o]
        for k in range(model_y.shape[0]):
            output = model_y[k, :, :]
            if data_m[k] == j:
                output = data_y[k, :, :]
            else:
                output[data_y[k, :, :] == 1] = 0
            model_y[k, :, :] = output

        new_data_y[:, :, :, j] = model_y
    new_data_y[:, :, :, -1] = project_configuration['shared_model_unknown_class_threshold']

    max_indices = np.argmax(new_data_y, axis=-1)
    one_hot_y = np.zeros_like(new_data_y)
    J, K, L = np.indices(max_indices.shape)
    one_hot_y[J, K, L, max_indices] = 1

    ontologies.append("Unknown")
    for j, o in enumerate(ontologies):
        print(j, o)
        out_tif = np.squeeze(one_hot_y[:, :, :, j]).astype(np.float32)
        tifffile.imwrite(os.path.join(project_configuration["root"], "training_datasets", "phase2", f"out_{o}.tif"), out_tif)
    ontologies.remove("Unknown")

    print(f"Saved phase2 training data. Image sizes:\n\tin:  {data_x.shape}\n\tout: {one_hot_y.shape}")
    print(f"Features:\n\tin:  {project_configuration['macromolecules']}\n\tout:  {project_configuration['ontologies']}")


def phase_2_train(gpus):
    import tensorflow as tf
    import keras.callbacks
    from Pom.models.unet import create_model

    os.makedirs(os.path.join(project_configuration['root'], "models", "phase2"), exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    def add_redundancy(data):
        data_out = list()
        for img in data:
            for k in [0, 1, 2, 3]:
                _img = np.rot90(img, k, axes=(0, 1))
                data_out.append(_img)
                data_out.append(np.flip(_img, axis=0))
        return np.array(data_out)

    macromolecule_inputs = project_configuration["macromolecules"]
    ontologies = project_configuration["ontologies"]
    ontologies.append("Unknown")

    data_y = list()
    data_x = list()

    for o in ontologies:
        data_y.append(add_redundancy(tifffile.imread(os.path.join(project_configuration["root"], "training_datasets", "phase2", f"out_{o}.tif"))))
    for m in macromolecule_inputs:
        data_x.append(add_redundancy(tifffile.imread(os.path.join(project_configuration["root"], "training_datasets", "phase2", f"in_{m}.tif"))))

    data_y = np.stack(data_y, axis=-1)
    data_x = np.stack(data_x, axis=-1)

    def tf_data_generator():
        n_samples = data_x.shape[0]
        indices = np.arange(n_samples)

        while True:
            np.random.shuffle(indices)
            for i in range(0, n_samples, project_configuration["shared_model_batch_size"]):
                batch_indices = indices[i:i + project_configuration["shared_model_batch_size"]]
                batch_x = data_x[batch_indices]
                batch_y = data_y[batch_indices]
                yield batch_x, batch_y

    training_data = tf.data.Dataset.from_generator(tf_data_generator, output_signature=(tf.TensorSpec(shape=(None, data_x.shape[1], data_x.shape[2], data_x.shape[3]), dtype=tf.float32), tf.TensorSpec(shape=(None, data_y.shape[1], data_y.shape[2], data_y.shape[3]), dtype=tf.float32)))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    training_data = training_data.with_options(options)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(data_x.shape[1:], output_dimensionality=data_y.shape[-1])

    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(project_configuration["root"], "models", "phase2", f"CombinedOntologies_checkpoint.h5"), monitor='loss', mode='min', save_best_only=True)
    model.fit(training_data, epochs=project_configuration["shared_model_epochs"], steps_per_epoch=data_x.shape[0] // project_configuration["shared_model_batch_size"], shuffle=True, callbacks=[checkpoint_callback])
    model.save(os.path.join(project_configuration["root"], "models", "phase2", f"CombinedOntologies.h5"))


def phase_2_process(gpus="0"):
    import tensorflow as tf
    from scipy.ndimage.filters import convolve1d
    from numpy.random import shuffle
    import itertools
    import multiprocessing

    os.makedirs(os.path.join(project_configuration["root"], project_configuration['output_dir']), exist_ok=True)

    def preprocess_volume(vol):
        return bin_vol(convolve1d(vol, np.ones(project_configuration["z_sum"] + 1) / (project_configuration["z_sum"] + 1), axis=0, mode='nearest'), 2)

    def segment_tomo(tomo, model, n_features):
        components = dict()

        tomo_name = os.path.splitext(os.path.basename(tomo))[0]
        for m in project_configuration["macromolecules"]:
            if m == 'Density':
                components[m] = preprocess_volume(mrcfile.read(os.path.join(project_configuration["root"], "full_dataset", f"{tomo_name}.mrc")))
            else:
                components[m] = preprocess_volume(mrcfile.read(os.path.join(project_configuration["root"], "macromolecules", f"{tomo_name}__{m}.mrc"))) / 255.0 * 2.0 - 1.0

        data_x = np.zeros((*components[project_configuration["macromolecules"][0]].shape, len(project_configuration["macromolecules"])), dtype=np.float32)
        for j, m in enumerate(components):
            data_x[:, :, :, j] = components[m]

        data_y = np.zeros((*data_x.shape[0:3], n_features), dtype=np.float32)

        normalize = [m == 'Density' for m in project_configuration["macromolecules"]]
        n_runs = max(min(4, project_configuration['shared_model_runs_per_volume']), 1)
        for j in range(data_y.shape[0]):
            slice_j = data_x[j, :, :, :]
            for k in range(n_runs):
                rotated_slice = np.rot90(slice_j, k=k, axes=(0, 1))
                boxes, imgsize, padding, stride = image_to_boxes(rotated_slice, boxsize=64, overlap=project_configuration["shared_model_overlap"], normalize=normalize)
                boxes = model.predict(boxes)
                segmented_slice = np.rot90(np.squeeze(boxes_to_image(boxes, imgsize, padding, stride)), k=-k, axes=(0, 1))
                data_y[j, :, :, :] += segmented_slice
        data_y /= n_runs
        return data_y

    def _thread(model_path, tomogram_paths, gpu_id, ontology_names):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        model = tf.keras.models.load_model(model_path, compile=False)


        for tomo in tomogram_paths:
            t_start = time.time()
            try:
                out_name = os.path.join(project_configuration["root"], project_configuration['output_dir'], os.path.basename(os.path.splitext(tomo)[0])+f"__{ontology_names[0]}.mrc")
                if not os.path.exists(out_name):
                    with mrcfile.new(out_name, overwrite=True) as f:
                        f.set_data(-np.ones((10, 10, 10), dtype=np.float32))
                else:
                    continue
                vol_out = segment_tomo(tomo, model, len(ontology_names))
                for k, o in enumerate(ontology_names):
                    with mrcfile.new(os.path.join(project_configuration["root"], project_configuration['output_dir'], os.path.basename(os.path.splitext(tomo)[0])+f"__{o}.mrc"), overwrite=True) as f:
                        f.set_data(vol_out[:, :, :, k].astype(np.float32))
                print(f"CombinedOntologies cost: {time.time() - t_start:.1f} seconds {(time.time() - t_start) / len(ontology_names):.1f} per ontology.")
            except Exception as e:
                print(e)

    ontologies = project_configuration["ontologies"]
    all_tomos = glob.glob(os.path.join(project_configuration["root"], project_configuration["test_dir"], "*.mrc"))

    shuffle(all_tomos)
    _gpus = [int(j) for j in gpus.split(",")]
    data_div = {gpu: list() for gpu in _gpus}
    for gpu, tomo_path in zip(itertools.cycle(_gpus), all_tomos):
        data_div[gpu].append(tomo_path)

    ontologies.append("Unknown")
    processes = []
    for gpu_id in data_div:
        p = multiprocessing.Process(target=_thread, args=(os.path.join(project_configuration["root"], "models", "phase2", f"CombinedOntologies.h5"), data_div[gpu_id], gpu_id, ontologies))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def phase_3_summarize(overwrite=True):
    import pandas as pd

    data_directories = [project_configuration["output_dir"], project_configuration["macromolecule_dir"]]
    summary_path = os.path.join(project_configuration["root"], 'summary.xlsx')
    df = None if not os.path.exists(summary_path) else pd.read_excel(summary_path, index_col=0)

    files = list()
    for d in data_directories:
        files += glob.glob(os.path.join(project_configuration["root"], d, "*__*.mrc"))

    data = dict()
    i = 0
    for f in files:
        i += 1

        tag = os.path.basename(f).split("__")[0]
        feature = os.path.splitext(os.path.basename(f))[0].split("__")[-1]

        if tag not in data:
            data[tag] = dict()

        if not overwrite and df is not None:
            if tag in df.index and feature in df.columns:
                v = df.loc[tag, feature]
                if not pd.isna(v):
                    data[tag][feature] = v
                    continue

        print(f"{i}/{len(files)}\t{feature}{' ' * (20 - len(feature))}{os.path.basename(f)}")
        volume = mrcfile.read(f)
        n_slices_margin = int(project_configuration["z_margin_summary"] * volume.data.shape[0])
        volume = volume[n_slices_margin:-n_slices_margin, :, :]
        if volume[0, 0, 0] == -1:  # then it's a placeholder volume.
            continue
        if volume.dtype == np.float32:
            data[tag][feature] = np.mean((volume > 0.5)) * 100.0
        else:
            data[tag][feature] = np.mean((volume > 128)) * 100.0

    keys = sorted(list(data.keys()))
    sorted_data = dict()
    for k in keys:
        sorted_data[k] = data[k]
    df = pd.DataFrame.from_dict(sorted_data, orient='index')
    df.to_excel(summary_path, index=True, index_label="tomogram")

    print(f"Dataset summary saved at {summary_path}")


def render_volumes(renderer, tomo_path, requested_compositions, feature_library, overwrite=False, save=True, df_summary=None):
    from Pom.core.render import FeatureLibraryFeature, SurfaceModel
    from PIL import Image

    def get_volume(tomo_path, feature_name):
        tomo_tag = os.path.splitext(os.path.basename(tomo_path))[0]

        # Look for the feature in the macromolecule and output directories
        m_path = glob.glob(os.path.join(project_configuration["root"], project_configuration["macromolecule_dir"], f"{tomo_tag}*{feature_name}.mrc"))
        o_path = glob.glob(os.path.join(project_configuration["root"], project_configuration["output_dir"], f"{tomo_tag}*{feature_name}.mrc"))

        # Try to load the MRC file
        mrc_file = None
        if len(m_path) > 0 and os.path.exists(m_path[0]):
            mrc_file = m_path[0]
        elif len(o_path) > 0 and os.path.exists(o_path[0]):
            mrc_file = o_path[0]

        # If the MRC file is found, read it and get the pixel size
        if mrc_file:
            with mrcfile.open(mrc_file, permissive=True) as mrc:
                volume = np.copy(mrc.data)
                pixel_size = mrc.voxel_size.x  # Get the pixel size (in nanometers by default)

            # Return the volume and pixel size
            return volume, pixel_size

        # If no MRC file is found, raise an exception
        raise Exception(f"Could not find feature {feature_name} for tomogram at {tomo_path}")

    def render_compositions(tomo_path, requested_compositions, feature_library, overwrite=False):
        skip_composition = list()
        if not overwrite:
            image_base_name = os.path.basename(os.path.splitext(tomo_path)[0])
            for composition_name in requested_compositions:
                out_filename = os.path.join(project_configuration["root"], project_configuration["image_dir"],
                                            f"{image_base_name}_{composition_name}.png")
                os.makedirs(os.path.join(project_configuration["root"], project_configuration["image_dir"], composition_name), exist_ok=True)
                if os.path.exists(out_filename):
                    skip_composition.append(True)
                else:
                    skip_composition.append(False)
            # check if any image already there.

        surface_models = dict()
        for j, c in enumerate(requested_compositions.values()):
            if not overwrite and skip_composition[j]:
                continue
            for feature in c:
                if feature not in surface_models:
                    t_start = time.time()
                    feature_volume, pixel_size = get_volume(tomo_path, feature)
                    surface_models[feature] = SurfaceModel(feature_volume, feature_library[feature], pixel_size)
                    print(f"Parsed {feature} ({time.time() - t_start:.1f} seconds)")

        image_base_name = os.path.basename(os.path.splitext(tomo_path)[0])
        out_images = dict()
        for composition_name in requested_compositions:
            renderer.new_image()
            renderer.render_surface_models([surface_models[f] for f in requested_compositions[composition_name] if f in surface_models])
            image = renderer.get_image()
            if save:
                Image.fromarray(image).save(os.path.join(project_configuration["root"], project_configuration["image_dir"], composition_name, f"{image_base_name}_{composition_name}.png"))
            else:
                out_images[composition_name] = image

        for s in surface_models.values():
            s.delete()

        return out_images

    # parse compositions
    tomo_name = os.path.splitext(os.path.basename(tomo_path))[0].split("_bin2")[0] # TODO: remove this
    sorted_ontologies = df_summary.loc[tomo_name].sort_values(ascending=False).index.tolist()
    for f in project_configuration["soft_ignore_in_summary"] + project_configuration["macromolecules"]:
        if f in sorted_ontologies:
            sorted_ontologies.remove(f)
    tomo_req_compositions = dict()
    for name, composition in zip(requested_compositions.keys(), requested_compositions.values()):
        composition_features = list()
        available_ontologies = copy.deepcopy(sorted_ontologies)
        for feature in composition:
            if feature[0] == "!":
                if feature[1:] in available_ontologies:
                    available_ontologies.remove(feature[1:])
        for feature in composition:
            if "rank" in feature:
                j = int(feature.split("rank")[-1]) - 1
                if j < len(available_ontologies):
                    feature = available_ontologies[j]
            composition_features.append(feature)
            if feature not in feature_library:
                feature_library[feature] = FeatureLibraryFeature()
                feature_library[feature].title = feature
        tomo_req_compositions[name] = composition_features

    out_images = render_compositions(tomo_path, tomo_req_compositions, feature_library, overwrite=overwrite)
    return out_images


def phase_3_render(composition_path="", style=0, n=-1, feature_library_path="", tomo_name='', overwrite=False):
    from Pom.core.render import Renderer, parse_feature_library
    from numpy.random import shuffle
    import pandas as pd

    os.makedirs(os.path.join(project_configuration["root"], project_configuration["image_dir"]), exist_ok=True)

    if not os.path.isabs(composition_path):
        composition_path = os.path.join(os.getcwd(), composition_path)
    if not os.path.isabs(feature_library_path):
        feature_library_path = os.path.join(os.getcwd(), feature_library_path)

    all_tomograms = glob.glob(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], "*.mrc"))
    shuffle(all_tomograms)
    if n > -1:
        all_tomograms = all_tomograms[:min(n, len(all_tomograms))]
    if tomo_name != '':
        all_tomograms = glob.glob(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"*{tomo_name}*"))

    with open(composition_path, 'r') as f:
        requested_compositions = json.load(f)

    feature_library = parse_feature_library(feature_library_path)
    renderer = Renderer(style, image_size=project_configuration["image_size"])

    df_summary = pd.read_excel(os.path.join(project_configuration["root"], "summary.xlsx"), index_col=0)
    for t in all_tomograms:
        try:
            render_volumes(renderer, t, requested_compositions, feature_library, overwrite, df_summary=df_summary)
        except Exception as e:
            print(e)

    renderer.delete()


def phase_3_projections(overwrite=False):
    from PIL import Image
    tomograms = list()
    for tomo in glob.glob(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], "*.mrc")):
        tomo_name = os.path.splitext(os.path.basename(tomo))[0]
        density_out = os.path.join(project_configuration["root"], project_configuration["image_dir"], "density", f"{tomo_name}_density.png")
        if not os.path.exists(density_out):
            tomograms.append(tomo_name)

    ontologies = project_configuration["ontologies"]
    if not "Unknown" in ontologies:
        ontologies.append("Unknown")

    os.makedirs(os.path.join(project_configuration["root"], project_configuration["image_dir"], "density"), exist_ok=True)
    for o in ontologies:
        os.makedirs(os.path.join(project_configuration["root"], project_configuration["image_dir"], o), exist_ok=True)

    for tomo in tomograms:
        out_base = os.path.join(project_configuration["root"], project_configuration["image_dir"])

        images_xy = dict()
        images_xz = dict()
        xy_max = -1e9
        xz_max = -1e9
        for o in ontologies:
            path = os.path.join(project_configuration["root"], project_configuration["output_dir"], f"{tomo}__{o}.mrc")
            if not os.path.exists(path):
                continue
            with mrcfile.open(path) as mrc:
                n_slices_margin = int(project_configuration["z_margin_summary"] * mrc.data.shape[0])
                data = copy.copy(mrc.data[n_slices_margin:-n_slices_margin, :, :])

            images_xy[o] = np.sum(data, axis=0)
            images_xz[o] = np.sum(data, axis=1)
            _max = np.amax(images_xy[o])
            if _max > xy_max:
                xy_max = _max
            _max = np.amax(images_xz[o])
            if _max > xz_max:
                xz_max = _max
        for o in ontologies:
            if not o in images_xy:
                continue
            img = images_xy[o]
            img = img / xy_max * 255
            img = img.astype(np.uint8)
            Image.fromarray(img, mode='L').save(os.path.join(out_base, o, f"{tomo}_{o}.png"))

            img = images_xz[o]
            img = img / xz_max * 255
            img = img.astype(np.uint8)
            Image.fromarray(img, mode='L').save(os.path.join(out_base, o, f"{tomo}_{o}_side.png"))

        with mrcfile.open(os.path.join(project_configuration["root"], project_configuration["tomogram_dir"], f"{tomo}.mrc")) as mrc:
            n_slices = mrc.data.shape[0]
            density = mrc.data[n_slices//2, :, :]
            Image.fromarray(density, mode='L').save(os.path.join(out_base, "density", f"{tomo}_density.png"))



def phase_3_browse():

    os.system(f"streamlit run {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'Introduction.py')}")