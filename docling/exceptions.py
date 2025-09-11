class BaseError(RuntimeError):
    """A base class for all custom exceptions in the Docling library.

    This class serves as a common ancestor for all other custom exceptions,
    allowing them to be caught together.
    """

    pass


class ConversionError(BaseError):
    """An exception raised when an error occurs during document conversion.

    This exception is used to indicate that a failure occurred while processing
    or converting a document from one format to another.
    """

    pass


class OperationNotAllowed(BaseError):
    """An exception raised when a requested operation is not allowed.

    This exception is used to indicate that an attempted action is forbidden
    or not supported in the current context.
    """

    pass
