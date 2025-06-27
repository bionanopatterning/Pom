[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/bionanopatterning/Pom/blob/main/Pom/license.txt)
[![Downloads](https://img.shields.io/pypi/dm/Pom-cryoET)](https://pypi.org/project/Pom-cryoET/)
[![Documentation Status](https://readthedocs.org/projects/pom-cryoet/badge/?version=latest)](https://pom-cryoet.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/bionanopatterning/Pom)

# Pom

**Scaling data analyses in cellular cryoET using comprehensive segmentation.**  

## Abstract

Automation and improved hardware have greatly accelerated the rate of data generation in cryoET. As the field moves towards quantitative cryoET, the scale of the resulting datasets presents a significant challenge for analysis and interpretation. To explore ways of handling datasets comprising thousands of tomograms, we investigated a comprehensive segmentation strategy – assigning an ontology-based identity to every voxel in a dataset – that is based on the sequential application of multiple convolutional neural networks. Using an openly available [dataset](https://cryoetdataportal.czscience.com/datasets/10302) of over 1800 Chlamydomonas reinhardtii tomograms as a test case, we demonstrate the segmentation of 25 different subcellular features across the full dataset, while requiring only a few seconds of processing time per tomogram. We show how the approach enables the representation of large datasets as searchable databases and propose the usage of ontology-based segmentations for improving two common processing tasks in cryoET. First, we explore context-aware particle picking as a method to retain biological context when selecting particles for subtomogram averaging and other downstream analyses. Secondly, we demonstrate area-selective template matching, where we use segmentation-based masks to avoid redundant computations in template matching and enable >500-fold faster processing in specific cases. To illustrate the utility of the approach, all segmentation results have also been made available online via cryopom.streamlit.app.

## Links

**Documentation**: https://pom-cryoet.readthedocs.io/en/latest/

**Data browser**: https://cryopom.streamlit.app/

**Ais**: https://www.github.com/bionanopatterning/Ais

**Ais model repository**: https://www.aiscryoet.org/

**Area-selective template matching**: https://www.github.com/bionanopatterning/Pommie

