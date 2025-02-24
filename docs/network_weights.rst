Network weights
__________

Below we'll provide links to download the network weights, and show example code to use these to process volumes yourself. Please note that this particular instance of the network was _not_ trained to be a general network for organelle segmentation! I.e., if your data has a significantly different pixel size, if you used a different tomogram reconstruction algorithm, or if your sample looks very different, the network will not perform well. In fact it will probably segment everything as 'void' and 'unknown' (>99% of the output corresponded to these classes when we tested the same networks on very different data).

Macromolecule segmentation
~~~

Membrane:       https://aiscryoet.org/models/31
Ribosome:       https://aiscryoet.org/models/32
ATP synthase:   https://aiscryoet.org/models/33
RuBisCo:        https://aiscryoet.org/models/29

These can be downloaded and used in Ais directly. E.g.:

::

   ais segment -d tomograms -ou macromolecules -m downloads/15.68_64_Membrane.scnm -gpu 0,1,2,3

Organelle segmentation
~~~

The weights for the shared network can be downloaded here: https://drive.google.com/file/d/1811hx4Y8sXd8LgauCZjcuQZ8XzHRmDp9/view?usp=sharing

Processing the Chlamy dataset
~~~

To generate the same segmented volumes as we did in the article, download the below network weights and run the commands below. Note that the network we're sharing is one that uses density as the input only! If you want the density to macromolecules to organelles network, let us know and we'll re-train an instance of that to share.

First, set the following parameters in the project_configuration.json file:

::

    "macromolecules": [
    "Density"
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
    "Stroma",
    "Lipid droplet",
    "Dense layer",
    "Endoplasmic reticulum",
    "NPC",
    "Starch granule",
    "Cilium",
    "IFRA",
    "Cell wall",
    "Chloroplast membrane"
    ],
    "z_sum": 2,

Then copy the weights file (.h5) to root/models/phase2/ and run 'pom shared process' as below. If you do not want to save the output volumes for some of the output features, replace their name by a single underscore ('_'). Do not change the number of values in the list or their order though - that would lead to output volumes being given the wrong name.

::

   pom shared process -gpu 0,1,2,3,4,5,6,7,8

