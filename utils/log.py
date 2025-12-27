import sys
import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter
from logging import INFO, DEBUG

# ===== User Configuration ====================================================

# Define the log file name based on the current date, formatted as YYYYMMDD.log
now = datetime.datetime.now()
save_filename = now.strftime('%Y%m%d') + '.log'

# Set the logging level (CRITICAL > ERROR > WARNING > INFO > DEBUG)
log_level = INFO

# Configuration parameters
disable_stream_handler = False       # Set True to disable console logging
disable_file_handler = True          # Set False to enable logging to a file
display_date = False                 # Set True to include the date in log messages

# =============================================================================

# Define the default logging format
if display_date:
    # Date format with time in 'YYYY/MM/DD HH:MM:SS' format
    datefmt = '%Y/%m/%d %H:%M:%S'
    # Formatter with date and time for each log entry
    default_fmt = Formatter(
        '[%(asctime)s.%(msecs)03d] %(levelname)5s '
        '(%(process)d) %(filename)s: %(message)s',
        datefmt=datefmt
    )
else:
    # Formatter without date, showing level, filename, line, and message
    default_fmt = Formatter(
        '%(levelname)5s %(filename)s (%(lineno)d) : %(message)s'
    )

# Create the logger
logger = getLogger()

# Remove any duplicate handlers if they already exist on the logger
if logger.hasHandlers():
    logger.handlers.clear()

# Set the logger's level according to the configuration
logger.setLevel(log_level)

# Set up a stream handler for console output, if enabled
if not disable_stream_handler:
    try:
        # Attempt to use RainbowLoggingHandler for colored output
        from rainbow_logging_handler import RainbowLoggingHandler
        color_msecs = ('green', None, True)  # Green color for milliseconds
        stream_handler = RainbowLoggingHandler(
            sys.stdout, color_msecs=color_msecs, datefmt=datefmt
        )
        # Customize color for milliseconds in the output
        stream_handler._column_color['.'] = color_msecs
        stream_handler._column_color['%(asctime)s'] = color_msecs
        stream_handler._column_color['%(msecs)03d'] = color_msecs
    except Exception:
        # Use default StreamHandler if RainbowLoggingHandler is unavailable
        stream_handler = StreamHandler()

    # Set the stream handler to debug level for more detailed output
    stream_handler.setLevel(DEBUG)
    # Apply the default formatter to the stream handler
    stream_handler.setFormatter(default_fmt)
    # Add the stream handler to the logger
    logger.addHandler(stream_handler)

# Set up a file handler for saving logs to a file, if enabled
if not disable_file_handler:
    file_handler = FileHandler(filename=save_filename)

    # Set file handler to debug level for detailed log output
    file_handler.setLevel(DEBUG)
    # Apply the default formatter to the file handler
    file_handler.setFormatter(default_fmt)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
