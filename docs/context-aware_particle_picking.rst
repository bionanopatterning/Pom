Context aware particle picking
__________

Below we demonstrate how to perform context-aware particle picking (CAPP) with Pom. We first we use Ais for picking and then apply CAPP to contextualize the resulting coordinates. You can use any other software for picking, but Pom expects the data directory to be structured in a specific way, so you may have to manually address that. By default Pom expects .tsv coordinate files; if you want to use something else (e.g. .star), let us know and we'll add it (mlast@mrc-lmb.cam.ac.uk).

CAPP with Ais
~~~~

After segmenting macromolecules (for picking) and organelles (for contextualization), the project directory should have the following structure:

::

   root/
   ├── tomograms/
   │   ├── TS_001.mrc
   │   ├── TS_002.mrc
   │   ├── TS_002.scns
   │   └── ...
   ├── macromolecules/
   │   ├── TS_001__Membrane.mrc
   │   ├── TS_001__Ribosome.mrc
   │   └── ...
   ├── organelles/
   │   ├── TS_001__Cytoplasm.mrc
   │   ├── TS_001__Mitochondrion.mrc
   │   └── ...

In this example we'll pick Ribosomes with Ais (documentation `here <https://ais-cryoet.readthedocs.io/en/latest/>`_)

::

   ais pick -d macromolecules -t Ribosome -ou capp/Ribosome -threshold 128 -spacing 250 -size 1000000 -p 64

Which adds the following directory and coordinate data files.

::

   ├── capp/
   │   ├── Ribosome/
   │   │   ├── TS_001__Ribosome_coords.tsv
   │   │   ├── TS_002__Ribosome_coords.tsv
   │   │   └── ...


To contextualize these coordinates, use the following command:

::

   pom capp -t <target_particle> -w <context_window_size> -b <bin_factor> -p <parallel_jobs>

Arguments:
    ``-t``:
    Target particle type. In the example above, the value would be 'Ribosome'

    ``-w``:
    Size of the context window. Pom measures the 'context vector' by sampling average segmentation output values for all features within a box of this size.

    ``-b``:
    The bin factor between i) volumes from which coordinates were derived and ii) organelle-segmentation volume size. In the Pom workflow, this would be 2 - which is the default value for -b. When using coordinates obtained with other software, ensure that any possible difference in volume sizes is taken in to account.

    ``-p``:
    Number of parallel processes to run; e.g. 16. One or two threads per available CPU is best.

In this example we'll run:

::

   pom capp -t Ribosome -w 32 -p 16


After this is completed the contextualized coordinates will be saved to root/capp/Ribosome/result/. For example, in .../result/TS_001__Ribosome_coords.tsv:

::

    Z      Y      X      Cytoplasm      Mitochondrion      ...       Stroma
    47     14     179    0.82102        0.00320            ...       0.00000
    71     172    477    1.00000        0.00000            ...       0.00000
    etc.




