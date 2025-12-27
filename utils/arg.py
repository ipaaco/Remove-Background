import os
import sys
import argparse
import glob
from logging import DEBUG

from params import MODALITIES, EXTENSIONS  # Custom parameters for file types and extensions
import log_init  # Custom logging initialization

# Logger initialization
logger = log_init.logger
logger.info('Start!')

# Placeholder for YAML config loader
# TODO: Implement YAML config file and YAML loader

# Attempt to import the ailia library, if available
try:
    import onnxruntime as ort  # Replacing `ailia` with `onnxruntime`
    AILIA_EXIST = True
except ImportError:
    logger.warning('onnxruntime package cannot be found under `sys.path`')
    logger.warning('default env_id is set to 0, you can change the id by [--env_id N]')
    AILIA_EXIST = False

# Function to check if a file exists; exits if not found
def check_file_existance(filename):
    if os.path.isfile(filename):
        return True
    else:
        logger.error(f'{filename} not found')
        sys.exit()

# Creates a command-line argument parser with default arguments for input/output paths, etc.
def get_base_parser(
        description, default_input, default_save, input_ftype='image',
):
    """
    Initialize argument parser with default parameters.
    
    Parameters
    ----------
    description : str : Description for the parser.
    default_input : str : Default input file path.
    default_save : str : Default output file save path.
    input_ftype : str : Type of input file (image, video, etc.)
    
    Returns
    -------
    parser : ArgumentParser : Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description,
        conflict_handler='resolve'
    )
    parser.add_argument(
        '-i', '--input', nargs='*', metavar='IMAGE/VIDEO', default=default_input,
        help='Default input data path (image/video). If a directory is specified, model processes all files within.'
    )
    parser.add_argument(
        '-v', '--video', metavar='VIDEO', default=None,
        help='Specify video for input, or integer for webcam input.'
    )
    parser.add_argument(
        '-s', '--savepath', metavar='SAVE_PATH', default=default_save,
        help='Save path for the output file.'
    )
    parser.add_argument(
        '-b', '--benchmark', action='store_true',
        help='Run inference multiple times to measure performance. Not usable in video mode.'
    )
    parser.add_argument(
        '-e', '--env_id', type=int,
        default=0,  # Placeholder for environment ID selection
        help='Specific environment ID for processing.'
    )
    parser.add_argument(
        '--env_list', action='store_true',
        help='Display available environments list.'
    )
    parser.add_argument(
        '--ftype', metavar='FILE_TYPE', default=input_ftype,
        choices=MODALITIES,
        help='File type options: ' + ' | '.join(MODALITIES)
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logs.'
    )
    parser.add_argument(
        '--profile', action='store_true',
        help='Enable profiling mode for performance analysis.'
    )
    parser.add_argument(
        '-bc', '--benchmark_count', metavar='BENCHMARK_COUNT', default=5, type=int,
        help='Set number of benchmark iterations.'
    )
    return parser

# Updates and validates parser arguments
def update_parser(parser, check_input_type=True, large_model=False):
    """
    Updates parser arguments based on conditions like environment ID and input type.
    
    Parameters
    ----------
    parser : ArgumentParser : The argument parser instance.
    
    Returns
    -------
    args : ArgumentParser : Parsed arguments.
    """
    args = parser.parse_args()

    # Set logger level to DEBUG if `--debug` flag is set
    if args.debug:
        logger.setLevel(DEBUG)

    # Validate and set environment ID, handle specific requirements for large models
    if AILIA_EXIST:
        env = ort.get_available_providers()  # Example environment check with onnxruntime
        if args.env_list:
            logger.info(f'Available providers: {env}')

    # Update input mode if video mode is specified
    if args.video is not None:
        args.ftype = 'video'
        args.input = None

    # Handle input path as file, directory, or list of files
    if isinstance(args.input, list) and len(args.input) == 1:
        args.input = args.input[0]  # Simplify single-item lists to string

    elif os.path.isdir(args.input):
        # Collect files of specified types from directory
        files_grapped = []
        for extension in EXTENSIONS[args.ftype]:
            files_grapped.extend(glob.glob(os.path.join(args.input, extension)))
        logger.info(f'{len(files_grapped)} {args.ftype} files found!')

        args.input = sorted(files_grapped)

        # Create output directory if needed
        if args.savepath is not None:
            if '.' in args.savepath:
                logger.warning('Specify save directory if input is a directory')
                args.savepath = args.input + '_results'
            os.makedirs(args.savepath, exist_ok=True)
            logger.info(f'Output directory: {args.savepath}')

    elif os.path.isfile(args.input):
        args.input = [args.input]

    else:
        if check_input_type:
            logger.error('Input is neither file nor directory')
            sys.exit(0)

    return args

# Generates a save path for an output file
def get_savepath(arg_path, src_path, prefix='', post_fix='_res', ext=None):
    """
    Create or retrieve a save path for output file, with optional prefix, postfix, and extension.
    
    Parameters
    ----------
    arg_path : str : Desired save path or directory.
    src_path : str : Source file path.
    prefix : str : Optional prefix for filename.
    post_fix : str : Optional postfix for filename.
    ext : str : Optional file extension.
    
    Returns
    -------
    new_path : str : Final save path for the output file.
    """
    if '.' in arg_path:
        # arg_path is treated as file path with extension
        arg_base, arg_ext = os.path.splitext(arg_path)
        new_ext = arg_ext if ext is None else ext
        new_path = arg_base + new_ext
    else:
        # arg_path is treated as directory
        src_base, src_ext = os.path.splitext(os.path.basename(src_path))
        new_ext = src_ext if ext is None else ext
        new_path = os.path.join(arg_path, prefix + src_base + post_fix + new_ext)

    # Ensure output directory exists
    dirname = os.path.dirname(new_path)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    return new_path
