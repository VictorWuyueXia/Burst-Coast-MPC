import logging

from wsmpc.utils.config_schema import RuntimeConfig
from wsmpc.utils.resources import configure_runtime_resources


def test_runtime_resource_configuration_does_not_crash() -> None:
    report = configure_runtime_resources(
        RuntimeConfig(
            **{
                "node-id": "Runtime",
                "max-worker-threads": 1,
                "blas-threads": 1,
                "cpu-affinity": [],
                "set-env": True,
            }
        ),
        logger=logging.getLogger("test"),
    )

    assert report.env_threads["OMP_NUM_THREADS"] == "1"
