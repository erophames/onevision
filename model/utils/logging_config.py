import logging

def setup_logging():
    """Sets up logging configuration for the application.

        Configures the logging to display messages of level INFO and higher.
        """
    logging.basicConfig(level=logging.INFO)
