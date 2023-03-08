import os
import wandb
import logging
import helpers

logger = logging.getLogger(__name__)


def get_experiment_id():
    try:
        eid = os.environ["EXPERIMENT_ID"]
    except KeyError:
        eid = wandb.util.generate_id()
        os.environ["EXPERIMENT_ID"] = eid
        logger.warning(f"Could not find EXPERIMENT_ID in environment variables. Using generated id '{eid}'.")
    return eid


class _WandBLogger:
    def __init__(self):
        # Below values will be set when `init` is called.
        self.name = None
        self.args = None
        self.run = None

        # Constants from environment variables
        self.eid = None
        self.entity = os.environ["WANDB_ENTITY"]
        self.project = os.environ["WANDB_PROJECT"]

        self._accumulated_logs = {}

    def init(self, args, job_type="evaluate", tags=None):
        self.eid = get_experiment_id()
        self.name = f"{args.dataset}-{args.arch}-{args.embedding}-{args.classifier}-{self.eid}"

        if args.wandb_tags is not None:
            tags = (tags or []) + self._parse_tags(args.wandb_tags)

        cfg = args.to_dict()
        cfg.update(**helpers.versions())
        del cfg["wandb_tags"]

        init_kwargs = dict(
            name=self.name,
            job_type=job_type,
            config=cfg,
            entity=self.entity,
            project=self.project,
            tags=tags,
            reinit=True,
        )

        try:
            self.run = wandb.init(**init_kwargs)
        except wandb.errors.UsageError as err:
            logger.warning(f"Got error: '{str(err)}' when calling wandb.init. Attempting to init with "
                           f"'settings=wandb.Settings(start_method=''fork'')'")
            self.run = wandb.init(settings=wandb.Settings(start_method="fork"), **init_kwargs)

        return self.run

    @staticmethod
    def _parse_tags(tag_str):
        # Assumes comma-delimited tags
        tags = [tag.strip() for tag in tag_str.split(",")]
        return tags

    def accumulate(self, dct, global_step, local_step, max_local_steps):
        total_step = (global_step * max_local_steps) + local_step
        if total_step in self._accumulated_logs:
            self._accumulated_logs[total_step].update(dct)
        else:
            self._accumulated_logs[total_step] = dct

    def log_accumulated(self):
        for step, logs in sorted(self._accumulated_logs.items(), key=lambda item: item[0]):
            self.run.log(logs, step=step)


wandb_logger = _WandBLogger()
