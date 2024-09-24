import Pom.core.cli_fn as cli_fn
import Pom.core.config as cfg
import argparse
import json
import os
import shutil

root = os.path.dirname(os.path.dirname(__file__))

def main():
    project_config_json_path = os.path.join(os.getcwd(), "project_configuration.json")
    if not os.path.exists(project_config_json_path):
        shutil.copy(os.path.join(os.path.dirname(__file__), "core", "project_configuration.json"), project_config_json_path)
    with open(project_config_json_path, 'r') as f:
        config = json.load(f)
        cfg.project_configuration = config
    print(cfg.project_configuration)
    render_config_json_path = os.path.join(os.getcwd(), "render_configuration.json")
    if not os.path.exists(render_config_json_path):
        shutil.copy(os.path.join(os.path.dirname(__file__), "core", "render_configuration.json"), render_config_json_path)

    parser = argparse.ArgumentParser(description=f"Ontoseg cli tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Single model commands
    p1p = subparsers.add_parser('single', help='Initialize, train, or test phase1 single-ontology output models.')
    p1sp = p1p.add_subparsers(dest='phase1_command', help='Single-model commands')
    p1sp.add_parser('initialize', help='Initialize the training data for selected annotations.')

    p1sp_train = p1sp.add_parser('train', help='Train a single-ontology output model for a selected ontology.')
    p1sp_train.add_argument('-ontology', required=True, help='The ontology for which to train a network.')
    p1sp_train.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')

    p1sp_test = p1sp.add_parser('test', help='Test a single-ontology output model for a selected ontology.')
    p1sp_test.add_argument('-ontology', required=True, help='The ontology for which to test the trained network.')
    p1sp_test.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')

    # Shared model commands
    p2p = subparsers.add_parser('shared', help='Initialize, train, or launch phase2 combined models.')
    p2sp = p2p.add_subparsers(dest='phase2_command', help='Shared-model commands')
    p2sp.add_parser('initialize', help='Train a single model to output all configured ontologies.')

    p2sp_train = p2sp.add_parser('train', help='Train a single model to output all configured ontologies.')
    p2sp_train.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')

    p2sp_process = p2sp.add_parser('process', help='Process all tomograms with the shared model.')
    p2sp_process.add_argument('-gpus', required=False, help='Which GPUs to use, e.g. "0,1,2,3" for GPU 0-3. If used, overrides the GPU usage set in the project configuration.')

    # Analysis ocmmands
    p3p = subparsers.add_parser('summarize', help='Summarize the dataset (or the fraction of the dataset processed so-far) in an Excel file.')
    p3p.add_argument('-overwrite', required=False, default=0, help='Specify whether to re-analyze volumes for which values are already found in the previous summary. Default is 0 (do not overwrite).')

    p4p = subparsers.add_parser('render', help='Render segmentations and output .png files.')
    p4p.add_argument('-s', '--scene-compositions', required=True, help='Path to a .json configuration file that specifies named compositions to render for each tomogram.')
    p4p.add_argument('-n', required=False, default=-1, help='Specify a maximum number of tomograms to render images for (useful for testing). Default is -1, for all tomomgrams.')
    p4p.add_argument('-style', required=False, default=0, help='Specify the rendering style.')
    p4p.add_argument('-f', '--feature-library-path', required=False, help='Path to an Ais feature library to define rendering parameters. If none supplied, it is taken from the Ais installation directory, if possible')
    p4p.add_argument('-t', '--tomogram', required=False, default='', help='Optional: path to a specific tomogram filename to render segmentations for. Overrides -n argument.')
    p4p.add_argument('-o', '--overwrite', required=False, default=0, help='Set to 1 to overwrite previously rendered images with the same render configuration. Default is 0.')

    p5p = subparsers.add_parser('browse', help='Launch a local streamlit app to browse the summarized dataset.')

    args = parser.parse_args()
    if args.command == 'single':
        if args.phase1_command == "initialize":
            cli_fn.phase_1_initialize()
        elif args.phase1_command == "train":
            gpus = config["GPUS"] if not args.gpus else args.gpus
            if args.ontology not in config['ontologies']:
                print(f"The selected ontology {args.ontology} is not one of the configured ontologies:\n{config['ontologies']}")
            else:
                cli_fn.phase_1_train(gpus, args.ontology)
        elif args.phase1_command == "test":
            gpus = config["GPUS"] if not args.gpus else args.gpus
            if args.ontology not in config['ontologies']:
                print(f"The selected ontology {args.ontology} is not one of the configured ontologies:\n{config['ontologies']}")
            else:
                cli_fn.phase_1_test(gpus, args.ontology)
    elif args.command == 'shared':
        if args.phase2_command == "initialize":
            cli_fn.phase_2_initialize()
        elif args.phase2_command == "train":
            gpus = config["GPUS"] if not args.gpus else args.gpus
            cli_fn.phase_2_train(gpus)
        elif args.phase2_command == "process":
            gpus = config["GPUS"] if not args.gpus else args.gpus
            cli_fn.phase_2_process(gpus)
    elif args.command == 'summarize':
        cli_fn.phase_3_summarize(overwrite=args.overwrite == 1)
    elif args.command == 'render':
        feature_library_path = os.path.join(os.path.expanduser("~"), ".Ais", "feature_library.txt")
        if args.feature_library_path:
            feature_library_path = args.feature_library_path
        cli_fn.phase_3_render(args.scene_compositions, args.style, args.n, feature_library_path, args.tomogram, args.overwrite)
    elif args.command == 'browse':
        cli_fn.phase_3_browse()

if __name__ == "__main__":
    main()
    # with open("C:/Users/mgflast/PycharmProjects/ontoseg/project_configuration.json", 'r') as f:
    #     config = json.load(f)
    #     cfg.project_configuration = config
    #     cli_fn.project_configuration = config
    # feature_library_path = os.path.join(os.path.expanduser("~"), ".Ais", "feature_library.txt")
    #
    # #cli_fn.phase_3_render("C:/Users/mgflast/PycharmProjects/ontoseg/render_configuration.json", style=0, n=-1, feature_library_path=feature_library_path, overwrite=False)
    #cli_fn.phase_3_browse()