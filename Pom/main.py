import Pom.core.tools as tools
import argparse
import os

# TODO: measure thickness
# TODO: find top and bottom of lamella, measure particle distance to.
# TODO: add Warp metrics for CTF, movement, etc.
# TODO: add global context values to the particle subset thing.
# TODO: add lamella images to the browse tomograms thing.

def main():
    parser = argparse.ArgumentParser(description="Pom-cryoET")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    commands = dict()

    commands["initialize"] = subparsers.add_parser("initialize", help="Initialize a new Pom project in the current directory.")

    commands["add_source"] = subparsers.add_parser("add_source", help="Add tomogram and/or segmentation source directories.")
    commands["add_source"].add_argument('--tomograms', required=False, help='Path to tomogram directory')
    commands["add_source"].add_argument('--segmentations', required=False, help='Path to segmentation directory')

    commands["list_sources"] = subparsers.add_parser('list_sources', help='List all configured source directories.')

    commands["remove_source"] = subparsers.add_parser("remove_source", help="Remove tomogram and/or segmentation source directories.")
    commands["remove_source"].add_argument('--tomograms', required=False, help='Path to tomogram directory')
    commands["remove_source"].add_argument('--segmentations', required=False, help='Path to segmentation directory')

    commands["summarize"] = subparsers.add_parser("summarize", help="Generate summary of all tomograms and segmentations.")
    commands["summarize"].add_argument('--overwrite', action='store_true', help='Overwrite existing entries')
    commands["summarize"].add_argument('--feature', required=False, help='Ignore all but this feature')

    commands["projections"] = subparsers.add_parser("projections", help="Generate projection images for all tomograms and segmentations.")

    commands["render"] = subparsers.add_parser("render", help="Render isosurface images for tomogram compositions.")

    commands["browse"] = subparsers.add_parser("browse", help="Launch Streamlit app to browse tomograms and segmentations.")

    commands["build"] = subparsers.add_parser("build", help='Build dataset summary and render all images.')

    commands["contextualize"] = subparsers.add_parser('contextualize', help='Sample contextual information for particles in a star file and add to the star file as new columns.')
    commands["contextualize"].add_argument('--starfile', type=str, required=True, help='Path to the star file.')
    commands["contextualize"].add_argument('--samplers', type=str, nargs='+', required=True, help='Feature:radius (in Angstrom) pairs to sample contexts. For example: mitochondrion:500 membrane:500.')
    commands["contextualize"].add_argument('--tomo-column', type=str, default=None, help='Column name for tomogram identifier. Defaults to rlnMicrographName or wrpSourceName.')
    commands["contextualize"].add_argument('--substitutions', type=str, nargs='*', default=None, help='search:replace pairs for mapping star file tomogram names to .mrc filenames. For example, for an M star file, use .tomostar:_10.00Apx or something like that.')
    commands["contextualize"].add_argument('--out_star', type=str, default=None, help='Path to output star file. If not provided, will overwrite input star file.')

    args = parser.parse_args()

    if args.command == 'initialize':
        tools.initialize()
    elif args.command == 'add_source':
        tools.add_source(args.tomograms, args.segmentations)
    elif args.command == 'remove_source':
        tools.remove_source(args.tomograms, args.segmentations)
    elif args.command == 'list_sources':
        tools.list_sources()
    elif args.command == 'summarize':
        tools.summarize(args.overwrite, args.feature)
    elif args.command == 'projections':
        tools.projections()
    elif args.command == 'render':
        tools.render()
    elif args.command == 'contextualize':
        if not os.path.exists(args.starfile):
            print(f'Star file {args.starfile} not found.')
            exit()
        tools.contextualize_starfile(args.starfile, args.samplers, tomogram_name=args.tomo_column, substitutions=args.substitutions, out_star=args.out_star)
    elif args.command == 'browse':
        import subprocess
        app_path = os.path.join(os.path.dirname(__file__), 'app', 'Introduction.py')
        print(f'streamlit run {app_path} --server.headless=true')
        subprocess.run(['streamlit', 'run', app_path, '--server.headless=true'])
    elif args.command == 'build':
        tools.summarize(overwrite=True)
        tools.projections()
        tools.render(overwrite=True)
        import subprocess
        app_path = os.path.join(os.path.dirname(__file__), 'app', 'Introduction.py')
        print(f'streamlit run {app_path} --server.headless=true')
        subprocess.run(['streamlit', 'run', app_path, '--server.headless=true'])




if __name__ == "__main__":
    main()
