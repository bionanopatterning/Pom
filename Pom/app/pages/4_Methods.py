import streamlit as st
import json

st.set_page_config(
    page_title="Methods",
    layout='wide'
)


with open("C:/Users/mgflast/PycharmProjects/ontoseg/project_configuration.json", 'r') as f:
    project_configuration = json.load(f)


st.title("Methods")
st.markdown("The idea of this project was that _if we can automatically, comprehensively annotate many tomograms_, we could summarize large datasets in both a machine and human interpretable way. To do so, we tested a staged approach using multiple convolutional neural networks.")
macromolecules = project_configuration["macromolecules"]
if "Density" in macromolecules:
    macromolecules.remove("Density")
st.markdown(f"In the **first stage** we use **[Ais](https://aiscryoet.org/)** to convert the **input density** volumes into multiple separate **macromolecule segmentations**. Here, the chosen macromolecular structures were _{', '.join(m for m in macromolecules[:-1])}, and {macromolecules[-1]}_.")
with st.expander("Stage 1 details"):
    st.markdown("We trained a separate network for each of the selected macromolecules, and used a sparsely, manually annotated set of images, sampled from across 20 tomograms (~1% of the full dataset) as the training data. For each  network, we limited the preparation time to around 30 minutes annotation. The resulting macromolecule-segmenting models are _far from perfect_, but are nonetheless useful in the next processing stage.")
    st.markdown("Conceptually, the rationale behind this strategy was as follows: if a human is presented with an image that depicts just membranes, ribosomes, and ATP synthase, they are already able to conclude a number of things about the underlying structure. A small circular area enclosed by a membrane and surrounded by ribosomes, for instance, is likely a vesicle or endosome surrounded by cytoplasm. A tubular membrane decorated with ATP synthase is likely to be part of a mitochondrion. Two somewhat parallel membranes with ribosomes on one side and only few distinct macromolecules on the other, probably represents a section of the nuclear envelope. By providing a second layer of neural networks with not just density, but also a number of macromolecule segmentations as the input, we hypothesized that a neural network might more accurately learn to segment and distinguish between more abstract features, or _ontologies_, such as the cytoplasm, mitochondria, nuclear envelope, the Golgi apparatus, vesicles, or chloroplasts.")
    st.markdown("The **network architecture** can be found [here](https://github.com/bionanopatterning/Ais/blob/master/Ais/models/vggnet_l.py). It is a VGGNet with 12 convolutional layers trained using a binary crossentropy loss function.")
    st.markdown("After preparing models on a desktop PC, the segmentations were computed using four Linux systems with 8 CUDA-enabled GPUs each (RTX 3080 and RTX 2080 Ti), requiring on average ~2.2 seconds per macromolecule per tomogram (note: that is by virtue of running 32 parallel processes; on a single GPU, a volume processed in ~70 seconds). The tomograms binned a factor 2, to a size of 512 x 512 x 256 voxels.")
    st.code("ais segment -m /models/15.68_64_0.0261_Membrane.scnm -d /full_dataset -ou /macromolecules -gpu 0,1,2,3,4,5,6,7")

ontologies = project_configuration["ontologies"]
st.markdown(f"The **second stage** comprises the preparation of multiple separate neural networks that each segment one particular [**ontology**](https://geneontology.org/docs/ontology-documentation/), using **macromolecule segmentations and density as the input.** Here, the chosen ontologies were _{', '.join(o for o in ontologies[:-1])}, and {ontologies[-1]}_. In order to prepare separate training datasets for each ontology, we used **[Ais](https://aiscryoet.org/)** to sparsely annotate the chosen ontologies across the full dataset. For each ontology, between 10 and 30 tomograms were sampled for the training data. Note that we train these models as an intermediate step; they will not be applied to segment the full dataset.")
with st.expander("Stage 2 details"):
    st.markdown("We use the term ontology to indicate _some distinct compartment, structure, or location; some abstract area of interest_ that often makes up a core concept in a cryoET project. Whichever structure is studied, certain specific ontologies represent that study's areas of interest, while others ontologies represent regions of the sample that may be irrelevant. For example, when studying chromatin, we typically need not search for it the cytoplasm. If it is feasible to automatically annotate many different ontologies in a large dataset, it should thus be possible to filter and select parts of the dataset that are the most relevant for a particular study, while reducing the time spent studying volumes or subtomograms that are less likely to contain useful information. Concretely, we envision two main use cases. First, to automatically select subsets of large datasets, by filtering based on the ontology content and quality metrics. Second, to use ontological segmentations to generate masks for template matching or particle picking. A possible third use might be to enrich the metadata of a particle set, by measuring the local environment within which a particle was found (e.g. nuclear vs. chloroplast vs. cytosolic ribosomes)")
    st.markdown("**Void**: this is the name that we use to refer to ill-defined densities in tomograms. The most prevalent representation of void is found in the top and bottom of reconstructed volumes, when the reconstruction depth is larger than the actual thickness of the imaged area. Besides annotating these regions as void, we also considered reconstruction artefacts, blurry image regions, and large empty areas of the sample void. As a result, the void segmentations provide an indication not only of the boundaries of the lamella within the reconstructed volume, but also of the quality of a tomogram.")
    st.markdown("The **network architecture** was similar to that used in step 1, but with a number of dilating kernels used in the last two encoding layers.")

    st.code("pom single initialize   # parse training data for all ontologies")
    st.code("pom single train -ontology Cytoplasm -gpus 0,1,2,3,4,5,6,7   # train one of the models")
    st.code("pom single test -ontology Cytoplasm -gpus 0,1,2,3,4,5,6,7   # optionally, test models on a selected set of tomograms")

st.markdown(f"The **third and last stage** is to train a single, combined model, which outputs segmentations for all selected ontologies. By applying all previously prepared single-ontology models to all the single-ontology training datasets, and combining the results, we prepared a joint training dataset that consisted of close to 4000 multi-channel images, each channel representing density, one macromolecule, or one ontology. We then trained a final model to output all ontologies, using the selected macromolecules density images as the input. This model was then deployed on four Linux PCs with 8 CUDA-enabled GPUs each. Processing the full dataset took **1 hour and 40 minutes**, at an average segmentation time of around 300 milliseconds per ontology per tomogram (256 x 256 x 128 voxels).")

with st.expander("Stage 3 details"):
    st.markdown("Any combination of macromolecule and density inputs and ontology outputs can be selected to train a network for. Although we used all macromolecules _and_ density here, it might be interesting to test how accurately a network can get if it is trained on macromolecules but _not on density_. If this would work, the resulting networks might be density-agnostic, and thus perhaps more suited for reuse with different datasets. If so, it might be possible to automate a part of stage 1 by using specialized systems such as [MemBrain](https://github.com/teamtomo/membrain-seg) to prepare the macromolecule segmentations. Or perhaps going from density straight to ontologies may be more practica:; if a combination of different convolutional neural networks works, a single network should also be able to do the same thing. The challenge, however, is always to prepare suitable training data.")
    st.markdown("We used a UNet **network architecture** with 18 convolutional layers, three skip connections, and a softmax output layer, trained using a combined categorical crossentropy and dice loss, using a one-hot encoded training output.")

    st.code("pom shared initialize   # parse & segment all training data for all ontologies")
    st.code("pom shared train -gpus 0,1,2,3,4,5,6,7  # train the joint model")
    st.code("pom shared process -gpus 0,1,2,3,4,5,6,7  # process data")

st.markdown(f"Once all segmentations are complete, a summary was generated and thumbnail images automatically generated using Pom and the Ais renderer. Once this summary is complete, a streamlit app (like this one) can be launched to browse the data.")

with st.expander("Summary details"):
    st.code("pom summarize   # measure ontology volume fractions for all ontologies and macromolecules and write to an .xlsx file")
    st.code("pom projections   # output xy and xz projection images of all ontology segmentations")

    st.code("pom render -s render_configuration.json   # batch-output 3D rendered images showing ontologies and/or macromolecules as defined in render_configuration.json")
    st.text("render_configuration.json:")
    st.code('"Top3": ["rank1", "rank2", "rank3", "!Unknown"],\n"Macromolecules": ["Membrane", "Ribosome", "ATP synthase", "RuBisCo"]')

    st.text("after the above, launch a streamlit app using:")
    st.code("pom browse")

st.markdown("Contact: mgflast@gmail.com")
