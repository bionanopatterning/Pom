import Pom.core.config as cfg
import Pom.core.cli_fn as cli_fn
import argparse
import json
import os
import shutil

root = os.path.dirname(os.path.dirname(__file__))

# TODO: measure thickness
# TODO: find top and bottom of lamella, measure particle distance to.
# TODO: add Warp metrics for CTF, movement, etc.
# TODO: add global context values to the particle subset thing.
# TODO: add lamella images to the browse tomograms thing.

def main():
    parser = argparse.ArgumentParser(description=f"Ontoseg cli tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    p1p = subparsers.add_parser('single', help='Initialize, train, or test phase1 single-ontology output models.')
    p1sp = p1p.add_subparsers(dest='phase1_command', help='Single-model commands')
    p1sp.add_parser('initialize', help='Initialize the training data for selected annotations.')

    p1sp_train = p1sp.add_parser('train', help='Train a single-feature output model for a selected feature.')
    p1sp_train.add_argument('-o', '--ontology', required=False, default="", help='The feature for which to train a network.')
    p1sp_train.add_argument('-all', '--all-features', required=False, default=0, type=int, help="Use '-all 1' to train for all features (overrides -o argument)")
    p1sp_train.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')
    p1sp_train.add_argument('-c', '--counterexamples', required=False, default=1, help='(1 or 0 (default)). When 1, all annotations for all organelles are used for training the network. The annotations for the chosen -o are used in full, whereas loss masking is used to ensure only the annotated pixels of all other training datasets are used. ')

    p1sp_test = p1sp.add_parser('test', help='Test a single-feature output model for a selected feature.')
    p1sp_test.add_argument('-o', '--ontology', required=True, help='The feature for which to test the trained network.')
    p1sp_test.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')

    p1sp_process = p1sp.add_parser('process', help='Process all volumes using a single-featyre output model for a selected feature.')
    p1sp_process.add_argument('-o', '--ontology', required=True, help='Which feature to segment.')
    p1sp_process.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')

    # Shared model commands
    p2p = subparsers.add_parser('shared', help='Initialize, train, or launch phase2 combined models.')
    p2sp = p2p.add_subparsers(dest='phase2_command', help='Shared-model commands')
    p2sp_init = p2sp.add_parser('initialize', help='Compile the training data for the shared model.')
    p2sp_init.add_argument('-selective', required=False, default=1, help='Whether to use all original training data or only those images where there is at least 1 pixel annotated positively. Default is 1.')

    p2sp_train = p2sp.add_parser('train', help='Train a single model to output all configured ontologies.')
    p2sp_train.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')
    p2sp_train.add_argument('-checkpoint', required=False, default='', help="If used, continue training the model at <path> argument of -checkpoint.")
    #p2sp_train.add_argument('-split', required=False, default=0.0, help='Validation split size (default is no split applied; 0.1 = 10%, 0.2 = 20%, etc.).')

    p2sp_process = p2sp.add_parser('process', help='Process all tomograms with the shared model.')
    p2sp_process.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')

    # Analysis commands
    p3p = subparsers.add_parser('summarize', help='Summarize the dataset (or the fraction of the dataset processed so-far) in an Excel file.')
    p3p.add_argument('-overwrite', required=False, default=0, help='Specify whether to re-analyze volumes for which values are already found in the previous summary. Default is 0 (do not overwrite).')
    p3p.add_argument('-skip', '--skip-macromolecules', required=False, default=0, help='Specify whether to re-analyze volumes for which values are already found in the previous summary. Default is 0 (do not overwrite).')
    p3p.add_argument('-feature', '--target-feature', required=False, default=None, help='When used, measure values for only this feature.')

    p3p3_warp = subparsers.add_parser('parse_warp', help='Parse Warp .xml files and add the values to the summary.')
    p3p3_warp.add_argument('-d', required=False, default='warp_tiltseries', help='Name of the warp tiltseries directory (default is "warp_tiltseries").')

    p4p = subparsers.add_parser('render', help='Render segmentations and output .png files.')
    p4p.add_argument('-c', '--configuration', required=False, default="", help='Path to a .json configuration file that specifies named compositions to render for each tomogram. If not supplied, default compositions are the top 3 ontologies and all macromolecules. ')
    p4p.add_argument('-n', '--max_number', required=False, default=-1, help='Specify a maximum number of tomograms to render images for (e.g. when testing settings)')
    p4p.add_argument('-f', '--feature-library-path', required=False, default=None, help='Path to an Ais feature library to define rendering parameters. If none supplied, it is taken from the Ais installation directory, if possible')
    p4p.add_argument('-t', '--tomogram', required=False, default='', help='Optional: path to a specific tomogram filename to render segmentations for. Overrides -n argument.')
    p4p.add_argument('-o', '--overwrite', required=False, default=0, help='Set to 1 to overwrite previously rendered images. Default is 0.')
    p4p.add_argument('-p', '--processes', required=False, default=1, help='Number of parallel processing Renderer instances.')

    subparsers.add_parser('browse', help='Launch a local streamlit app to browse the summarized dataset.')

    p5p = subparsers.add_parser('projections', help='Launch a local streamlit app to browse the summarized dataset.')
    p5p.add_argument('-o', '--overwrite', required=False, default=0, help='Set to 1 to overwrite previously rendered images with the same render configuration. Default is 0.')
    p5p.add_argument('-p', '--processes', required=False, default=1, help='Number of parallel processing jobs. Default is 1, a higher value is likely faster.')

    capp = subparsers.add_parser('capp', help='Context-aware particle picking.')
    capp.add_argument('-c', '--config', required=True, help='Job name, or path to a config.json job definition file.')
    capp.add_argument('-w', '--window-size', required=False, default=16, type=int, help='Size (in pixels, at the same scale as the coordinates) of the context window.')
    capp.add_argument('-b', '--bin-factor', required=False, default=2, type=int, help='Difference in the sizes of i) particle picking volumes, and ii) organelle segmentation volumes. If (i) is a Pom macromolecule segmentation, the value for -b should be 2 (default).')
    capp.add_argument('-p', '--parallel', required=False, default=1, type=int, help='Number of parallel processes to run.')



    p1sp.add_parser('initialize', help='Initialize the training data for selected annotations.')
    astm = subparsers.add_parser('astm', help="Area-selective template matching.")
    astm_parser = astm.add_subparsers(dest='astm_command', help="ASTM commands")
    astm_run = astm_parser.add_parser('run', help="Run ASTM jobs.")
    astm_run.add_argument('-c', '--config', required=True, help="Job name, or path to a config.json job definition file.")
    astm_run.add_argument('-o', '--overwrite', required=False, default=0, help="Overwrite (1) or skip (0, default) tomos for which previous output exists.")
    astm_run.add_argument('-indices', '--save-indices', required=False, default=1, help="Whether to save the matching template indices to a separate output .mrc.")
    astm_run.add_argument('-masks', '--save-masks', required=False, default=1, help="Whether to save the volume mask to a separate output .mrc.")
    astm_run.add_argument('-t', '--tomo-name', required=False, default=None, help="Select a specific tomogram to process")

    astm_run = astm_parser.add_parser('pick', help="Pick particles from ASTM score volumes.")
    astm_run.add_argument('-c', '--config', required=True, help="Job name, or path to a config.json job definition file.")
    astm_run.add_argument('-threshold', required=True, type=float, help="Minimum matching score to be considered an instance of the tempalte particle.")
    astm_run.add_argument('-max', required=False, type=int, default=1e9, help="Maximum number of particles to pick per tomogram (default 1e9)")
    astm_run.add_argument('-spacing', '--minimum-spacing', required=False, type=float, default=1, help="Minimum inter-particle spacing (in Angstrom)")
    astm_run.add_argument('-spacing-px', '--minimum-spacing-px', required=False, type=int, default=1, help="Minimum inter-particle spacing (in pixels - overrides '-spacing')")
    astm_run.add_argument('-blur', '--blur-px', required=False, default=0, type=int, help="Apply a Gaussian blur along the Z direction with -blur <sigma_px>. Can be useful to reduce false positives (though possibly at the cost of introducing false negatives).")
    astm_run.add_argument('-p', '--parallel', required=False, default=1, type=int, help="Number of parallel processes to run (one or two per CPU is good; e.g. '-p 16')")

    args = parser.parse_args()
    if args.command == 'single':
        if args.phase1_command == "initialize":
            cli_fn.phase_1_initialize()
        elif args.phase1_command == "train":
            gpus = cfg.project_configuration["GPUS"] if not args.gpus else args.gpus
            cli_fn.phase_1_train(gpus, args.ontology, use_counterexamples=int(args.counterexamples), all_features=args.all_features)
        elif args.phase1_command == "test":
            gpus = cfg.project_configuration["GPUS"] if not args.gpus else args.gpus
            cli_fn.phase_1_test(gpus, args.ontology, process=False)
        elif args.phase1_command == "process":
            gpus = cfg.project_configuration["GPUS"] if not args.gpus else args.gpus
            cli_fn.phase_1_test(gpus, args.ontology, process=True)
    elif args.command == 'shared':
        if args.phase2_command == "initialize":
            cli_fn.phase_2_initialize(selective=args.selective)
        elif args.phase2_command == "train":
            gpus = cfg.project_configuration["GPUS"] if not args.gpus else args.gpus
            cli_fn.phase_2_train(gpus, checkpoint=args.checkpoint)
        elif args.phase2_command == "process":
            gpus = cfg.project_configuration["GPUS"] if not args.gpus else args.gpus
            cli_fn.phase_2_process(gpus)
    elif args.command == 'summarize':
        cli_fn.phase_3_summarize(overwrite=args.overwrite, skip_macromolecules=args.skip_macromolecules, target_feature=args.target_feature)
    elif args.command == 'parse_warp':
        cli_fn.phase_3_parse_warp()
    elif args.command == 'render':
        cli_fn.phase_3_render(args.configuration, args.max_number, args.tomogram, args.overwrite, args.processes, args.feature_library_path)
    elif args.command == 'projections':
        cli_fn.phase_3_projections(args.overwrite, args.processes)
    elif args.command == 'browse':
        cli_fn.phase_3_browse()
    elif args.command == 'capp':
        cli_fn.phase_3_capp(args.config, context_window_size=args.window_size, bin_factor=args.bin_factor, parallel=args.parallel)
    elif args.command == 'astm':
        if args.astm_command == 'run':
            cli_fn.phase_3_astm_run(args.config, True if args.overwrite == 1 else False, args.save_indices, args.save_masks, args.tomo_name)
        if args.astm_command == 'pick':
            cli_fn.phase_3_astm_pick(args.config, args.threshold, args.minimum_spacing, args.minimum_spacing_px, args.parallel, args.max, args.blur_px)


if __name__ == "__main__":
    main()
