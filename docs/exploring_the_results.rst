Exploring the results
__________

Once all the segmentations are complete, a number of commands can be used to summarize, render, and explore the data:

::

   pom summarize

This will generate an excel file that summarizes the composition of all tomograms.

::

   pom projections -p 64

This will generate 2D projection images for all organelle segmentations, as well as preview images of each tomogram. The -p argument specified how many parallel jobs to run; we used a value of 64 on a PC with 96 CPUs.

::

   pom render -p 64

This will generate 3D images of all tomograms. Multiple images will be generated per tomogram, with compositions as defined in render_configuration.json. See pom render -h for additional options.

::

   pom browse


Finally, pom browse will start the data browsing application. The IP address of the host is printed in the terminal; navigate here in a browser to explore the segmentation results.
