import wandb
from pathlib import Path


class WandbHelper:
    """Helper class for common W&B operations."""

    def __init__(self, entity: str, project: str):
        self.api = wandb.Api()
        self.entity = entity
        self.project = project

    def __init__(self, project: str):
            self.api = wandb.Api()
            self.entity = self.api.default_entity
            self.project = project

    def get_run_id_by_name(self, run_name: str):
        """Retrieve the run ID for a given run name."""
        runs = self.get_runs(filters={"display_name": run_name})
        if not runs:
            raise ValueError(f"No run found with name '{run_name}' in project '{self.project}'.")
        if len(runs) > 1:
            raise ValueError(f"Multiple runs found with name '{run_name}' in project '{self.project}'.")
        return runs[0].id

    def _get_run(self, run_id: str):
        """Retrieve a run object by its ID."""
        return self.api.run(f"{self.entity}/{self.project}/{run_id}")

    def get_runs(self, filters: dict = None):
        """Return all runs in the project, optionally filtered.
        
        Example filters: {"state": "finished", "config.batch_size": 32}
        """
        path = f"{self.entity}/{self.project}"
        return self.api.runs(path=path, filters=filters or {})

    def get_run_summary(self, run_id: str) -> dict:
        """Return the summary metrics of a run."""
        run = self._get_run(run_id)
        return dict(run.summary)

    def get_run_config(self, run_id: str) -> dict:
        """Return the config (hyperparameters) of a run."""
        run = self._get_run(run_id)
        return {k: v for k, v in run.config.items() if not k.startswith("_")}

    def get_run_history(self, run_id: str, keys: list = None):
        """Return the metric history of a run as a DataFrame."""
        run = self._get_run(run_id)
        return run.history(keys=keys, pandas=True)

    def list_artifacts(self, run_id: str):
        """List all artifacts logged by a run."""
        run = self._get_run(run_id)
        for artifact in run.logged_artifacts():
            print(f"Artifact: {artifact.name}, Type: {artifact.type}")
        return list(run.logged_artifacts())

    def download_artifact(self, artifact_path: str, download_dir: str = None) -> Path:
        """Download an artifact by its full path (entity/project/name:version)."""
        artifact = self.api.artifact(artifact_path)
        local_path = artifact.download(root=download_dir)
        return Path(local_path)

    def download_model_from_run(self, run_id: str, download_dir: str = None) -> Path:
        """Download the first 'model' type artifact from a given run."""
        run = self._get_run(run_id)
        for artifact in run.logged_artifacts():
            if "model" in artifact.type.lower():
                local_path = artifact.download(root=download_dir)
                print(f"Downloaded model artifact '{artifact.name}' to {local_path}")
                return Path(local_path)
        raise ValueError(f"No model artifact found for run '{run_id}'.")

    def download_best_model_from_sweep(
        self, sweep_id: str, metric: str = "val_acc", download_dir: str = None
    ) -> Path:
        """Download model artifacts from the best run in a sweep.

        Args:
            sweep_id: The sweep ID.
            metric: The summary metric to rank runs by (higher is better).
            download_dir: Optional local directory to download into.

        Returns:
            Path to the downloaded model artifact directory.
        """
        sweep = self.api.sweep(f"{self.entity}/{self.project}/{sweep_id}")
        runs = sorted(
            sweep.runs,
            key=lambda run: run.summary.get(metric, 0),
            reverse=True,
        )
        best_run = runs[0]
        print(f"Best run: {best_run.name} with {metric}={best_run.summary.get(metric, 0)}")

        for artifact in best_run.logged_artifacts():
            if "model" in artifact.type.lower():
                local_path = artifact.download(root=download_dir)
                print(f"Downloaded model artifact '{artifact.name}' to {local_path}")
                return Path(local_path)

        raise ValueError(f"No model artifact found for best run '{best_run.name}'.")

    def get_ultralytics_best_model_path(
        self, run_id: str
    ) -> str:
        """Retrieve Ultralytics model weights from the best run in a sweep.

        Ultralytics checkpoints are saved as artifacts named 'run_<run_id>_model'
        with aliases like 'epoch_N'. This method downloads the artifact from
        the best-performing run.

        Args:
            run_id: The run ID containing Ultralytics training weights.
            download_dir: Optional local directory to download into.

        Returns:
            Path to the downloaded checkpoint directory.
        """
        artifact_list = self.list_artifacts(run_id)
        # Get the artifact with alias 'best' or 'latest' that is of type 'model'
        for artifact in artifact_list:
            if artifact.type == "model" and ("best" in artifact.aliases or "latest" in artifact.aliases):
                print(f"Found Ultralytics model artifact '{artifact.name}' with aliases {artifact.aliases}")
                return artifact.name  # Return the artifact reference string    
        return None  # No suitable artifact found

    def download_model_from_registry(
        self,
        registry_name: str,
        collection_name: str,
        version: int = 0,
        team_entity: str = None,
        project: str = None,
    ) -> Path:
        """Download a model artifact from the W&B Model Registry.

        Args:
            registry_name: Name of the registry (e.g. "Model").
            collection_name: Name of the collection in the registry.
            version: Version number of the artifact.
            team_entity: Team entity (defaults to self.entity).
            project: Project name (defaults to self.project).

        Returns:
            Path to the downloaded model files.
        """
        artifact_name = f"wandb-registry-{registry_name}/{collection_name}:v{version}"
        entity = team_entity or self.entity
        proj = project or self.project

        with wandb.init(entity=entity, project=proj) as run:
            artifact = run.use_artifact(artifact_or_name=artifact_name)
            local_path = artifact.download()

        return Path(local_path)

    def download_file_from_run(self, run_id: str, filename: str, replace: bool = True):
        """Download a specific file logged to a run.

        Args:
            run_id: The run ID.
            filename: The filename as stored in W&B (e.g. 'model-best.h5').
            replace: Whether to overwrite an existing local file.
        """
        run = self._get_run(run_id)
        run.file(filename).download(replace=replace)
        print(f"Downloaded '{filename}' from run '{run_id}'.")