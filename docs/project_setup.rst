Project setup
__________

Before doing any segmentation, we set up a new Pom project by running the following command in the base folder of your project:

::

   pom

If no previous project is found in the directory, the output will be:

::

   created project_configuration.json at /lmb/home/mlast/PycharmProjects/Pom/project_configuration.json
   created render_configuration.json at /lmb/home/mlast/PycharmProjects/Pom/render_configuration.json

These two newly created files can be used to configure the project and rendering settings. In the project configuration, you'll find the following entries:

.. code-block:: JSON

    {
      "root": "",
      "tomogram_dir": "full_dataset",
      "macromolecule_dir": "macromolecules",
      "output_dir": "output",
      "test_dir": "test",
      "image_dir": "images",
      "ontology_annotation_box_size": 128,
      "apix": 15.68,
      "macromolecules": [
        "Density",
        "Membrane",
        "Ribosome"
      ],
      "ontologies": [
        "Cytoplasm",
        "Mitochondrion",
        "Nuclear envelope",
        "Nucleoplasm",
        "Pyrenoid tube",
        "Thylakoid",
        "Vesicle",
        "Void",
        "Golgi",
        "Pyrenoid",
        "Stroma"
      ],
      "z_sum": 4,
      "single_model_epochs": 100,
      "single_model_batch_size": 32,
      "single_model_overlap": 0.0,
      "single_model_box_size": 256,
      "shared_model_epochs": 200,
      "shared_model_batch_size": 64,
      "shared_model_overlap": 0.0,
      "shared_model_box_size": 256,
      "shared_model_unknown_class_threshold": 0.33,
      "shared_model_unknown_output_thresold": 0.5,
      "tf_distribute_mirrored": true,
      "shared_model_runs_per_volume": 4,
      "GPUS": "0,1,2,3,4,5,6,7",
      "image_size": 1024,
      "soft_ignore_in_summary": ["Void"],
      "gallery_categories": ["Macromolecules", "Top3", "density"],
      "raytraced_ontologies": false,
      "camera_pitch": -30.0,
      "camera_yaw": 180.0,
      "z_margin_summary": 0.3
    }

Most of these can be ignored for now, except for the first few. Pom expects the directory of a project to be structures as follows:

.. code-block:: text

   root/
   ├── tomogram_dir/                  (directory where input tomograms and annotated tomograms are stored)
   │   ├── TS_001.mrc
   │   ├── TS_002.mrc
   │   ├── TS_002.scns
   │   └── etc.
   ├── macromolecule_dir/             (directory where macromolecule segmentations are saved)
   │   ├── TS_001__Membrane.mrc
   │   ├── TS_001__Ribosome.mrc
   │   └── etc.
   ├── output_dir/                    (directory where organelle segmentations are saved)
   │   ├── TS_001__Cytoplasm.mrc
   │   ├── TS_001__Mitochondrion.mrc
   │   └── etc.
   ├── test_dir/                      (directory where a subset of tomograms can be placed, which can then be used for testing)
   ├── image_dir/                     (directory where 3D and 2D projection images are saved)
   │   ├── Density/
   │   ├── Cytoplasm/
   │   ├── Mitochondrion/
   │   └── etc.

Here, 'tomogram_dir', 'macromolecule_dir', etc. should have the names as specified in the project configuration.

Next, define the input and ouput channels to be used in the segmentation. In the example below one would use density volumes plus membrane and ribosome segmentations as the input, and output the listed organelles.

.. code-block:: JSON

    {
      "macromolecules": [
        "Density",
        "Membrane",
        "Ribosome"
      ],
      "ontologies": [
        "Cytoplasm",
        "Mitochondrion",
        "Nuclear envelope",
        "Nucleoplasm",
        "Pyrenoid tube",
        "Thylakoid",
        "Vesicle",
        "Void",
        "Golgi",
        "Pyrenoid",
    }
