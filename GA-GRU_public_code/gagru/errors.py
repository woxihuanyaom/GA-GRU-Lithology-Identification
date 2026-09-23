class ProtocolError(RuntimeError):
    """The frozen protocol is missing, malformed, or internally inconsistent."""


class DataValidationError(RuntimeError):
    """A well file does not satisfy the frozen schema or data invariants."""


class ExternalTestLockedError(PermissionError):
    """Code attempted to read a locked external-test well without explicit access."""


class SearchValidationError(RuntimeError):
    """A search run conflicts with its frozen budget, manifest, or resume log."""
