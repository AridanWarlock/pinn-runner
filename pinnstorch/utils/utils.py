import warnings
from typing import Any, Dict

from omegaconf import DictConfig

from pinnstorch.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras", False):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def get_metric_value(metric_dict: Dict[str, Any], metric_names: list) -> float:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: The name of the metric to retrieve.
    :return: The value of the metric.
    """
    for type_metric, list_metrics in metric_names.items():
        if type_metric == "extra_variables":
            prefix = ""
        elif type_metric == "error":
            prefix = "val/error_"

        for metric_name in list_metrics:
            metric_name = f"{prefix}{metric_name}"

            if not metric_name:
                log.info("Metric name is None! Skipping metric value retrieval...")
                continue

            if metric_name not in metric_dict:
                log.info(
                    f"Metric value not found! <metric_name={metric_name}>\n"
                    "Make sure metric name logged in LightningModule is correct!\n"
                    "Make sure `optimized_metric` name in `hparams_search` config is correct!"
                )
            else:
                metric_value = metric_dict[metric_name].item()
                log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
