from pathlib import Path
import pandas as pd


def get_artifact(run):
    """
    Download and cache the artifacts (plots) for
    a given run.
    Make sure to display the plot on the Weights and
    Biases website before running this code, otherwise
    the dataset will not exist.

    run: the wandb run to download the artifacts for.
    """
    path = Path("./artifacts/mapping/") / run.name

    if path.exists():
        mapping = pd.read_csv(path)
        data = mapping[run.name].item()
    else:
        artifact = run.logged_artifacts()
        if len(artifact) == 0:
            return pd.DataFrame()
        data = artifact[0].download()
        pd.DataFrame([data], columns=[run.name]).to_csv(path)
    return pd.read_parquet(data)


def get_filtered_runs(filter, api, entity, project):
    """
    Download the artifacts (plots on Weights and Biases)
    from the Weights and Biases api that satisfy a filter.

    filter: the filter that the run needs to satisfy
    api: a wandb.Api() instance
    entity: a wandb username
    project: the name of the project the runs belong to
    """
    runs = api.runs(entity + "/" + project)
    runs = [run for run in runs if filter(run)]

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    arts = {}
    for run in runs:
        if run.name not in arts:
            arts[run.name] = get_artifact(run)

    return arts
