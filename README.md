![Pom](docs/res/pom_banner.png)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/bionanopatterning/Pom/blob/main/Pom/license.txt)
[![Downloads](https://img.shields.io/pypi/dm/Pom-cryoET)](https://pypi.org/project/Pom-cryoET/)
[![Documentation Status](https://readthedocs.org/projects/pom-cryoet/badge/?version=latest)](https://pom-cryoet.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/bionanopatterning/Pom)

# Pom

**Curation, visualization, & contextualization for large scale cryoET.** Pom is a simple CLI tool plus a browser-based app for exploring and organising large cryoET datasets. Initialize a database with `pom initialize`, add data sources with `pom add_source --tomograms warp_tiltseries/reconstruction/` and `pom add_source --segmentations segmented/`, then fill the database with `pom summarize`, generate projection images with `pom projections`, 3D visualizations with `pom render`, and launch the app with `pom browse`. Or run `pom auto` after setting up the sources to do it all at once. For a dataset of 100 tomograms and 3 segmented features per tomo, this should take around 2 - 5 minutes.

Once inside the browser app, you can compile data subsets (which can directly be used in Ais, easymode, and WarpTools), change the visualisation settings, and explore the dataset. If you've registered both a tomogram source and a segmentation source, you can run `pom contextualize --starfile particles.star --samplers <see docs>` to take measurements of particles' context. Examples are distances to membranes, whether particles are inside some compartment, or the distance to the lamella surface. See the tutorials at https://mgflast.github.io/easymode/user_guide more details.


## Gallery ##
### Pom database browser ###

https://github.com/user-attachments/assets/39aa4768-e06b-4083-ba45-288d582c6a7d

https://github.com/user-attachments/assets/17d42bc4-c638-4d00-b1f4-40cc887bf5f2

### Links

**Documentation**: https://mgflast.github.io/easymode/user_guide/pom/installation/

## See our other tools ##

<p align="center">
  <a href="https://github.com/bionanopatterning/Ais"><img src="https://github.com/bionanopatterning/Ais/raw/master/docs/res/ais_banner.png" width="49%"></a>
  <a href="https://github.com/mgflast/easymode"><img src="https://github.com/mgflast/easymode/raw/master/assets/easymode_banner.png" width="49%"></a>
</p>

Mart So-Last, 2026 | mgflast@gmail.com 
