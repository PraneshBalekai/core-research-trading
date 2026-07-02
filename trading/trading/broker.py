import threading
import time

from core.ibkr import IBKRSession


def get_broker(broker_name: str, is_docker_run: bool = False):
    """Create and return a broker session based on the broker name.

    Args:
        broker_name: The broker name (e.g., "IBKR")
        is_docker_run: Whether running in docker (for IBKR host resolution)

    Returns:
        The broker session object
    """
    if broker_name == "IBKR":
        ibkr_app = IBKRSession()
        host = "host.docker.internal" if is_docker_run else "127.0.0.1"
        ibkr_app.connect(host, 4002, 0)
        threading.Thread(target=ibkr_app.run).start()
        time.sleep(1)
        return ibkr_app

    raise ValueError(f"Unsupported broker: {broker_name}")


def get_positions(broker_app):
    """Query and return positions from a broker session.

    Args:
        broker_app: The broker session object

    Returns:
        The positions dict from the broker session
    """
    broker_app.reqPositions()
    while not broker_app.position_query_end:
        time.sleep(1)
    broker_app.position_query_end = False
    return broker_app.positions
