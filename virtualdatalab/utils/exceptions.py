class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class NotFittedError(Error):
    """Exception raised if generator is not fitted

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message