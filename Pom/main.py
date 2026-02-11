import Pom.core.tools as tools
import argparse

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
    commands["render"].add_argument('--overwrite', action='store_true', help='Overwrite existing rendered images')

    commands["browse"] = subparsers.add_parser("browse", help="Launch Streamlit app to browse tomograms and segmentations.")

    commands["build"] = subparsers.add_parser("build", help='Build dataset summary and render all images.')

    
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
        tools.render(args.overwrite)
    elif args.command == 'browse':
        import subprocess, os
        app_path = os.path.join(os.path.dirname(__file__), 'app', 'Introduction.py')
        print(f'streamlit run {app_path} --server.headless=true')
        subprocess.run(['streamlit', 'run', app_path, '--server.headless=true'])
    elif args.command == 'build':
        tools.summarize(overwrite=True)
        tools.projections()
        tools.render(overwrite=True)
        import subprocess, os
        app_path = os.path.join(os.path.dirname(__file__), 'app', 'Introduction.py')
        print(f'streamlit run {app_path} --server.headless=true')
        subprocess.run(['streamlit', 'run', app_path, '--server.headless=true'])




if __name__ == "__main__":
    main()
