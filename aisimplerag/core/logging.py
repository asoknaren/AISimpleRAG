import logging

_LOGGING_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _LOGGING_CONFIGURED = True
