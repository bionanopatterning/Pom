Context aware particle picking
__________

Below we demonstrate how to perform context-aware particle picking (CAPP) with Pom. We first we use Ais for picking and then apply CAPP to contextualize the resulting coordinates. You can use any other software for picking, but Pom expects the data directory to be structures in a specific way, so you may have to manually address that. By default Pom expects .tsv coordinate files; if you want to use something else (e.g. .star), let us know and we'll add it (mlast@mrc-lmb.cam.ac.uk).

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


# TODO: add argument to also write the number of particles to the summary!
