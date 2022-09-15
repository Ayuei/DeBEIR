import joblib
import optuna
from functools import partial

from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.trial import TrialState
from torch.utils.data import DataLoader

from nir.training.hparm_tuning.trainer import Trainer


def objective(trainer: Trainer, trial: optuna.Trial):
    dataset = trainer.dataset_loading_fn()

    train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    return trainer.fit(trial, train_dataset, val_dataset)


def run_optuna_with_wandb(trainer, n_trials=100, n_jobs=1, maximize_objective=True, save_study_path=".",
                          wandb_kwargs=None):
    """
    Partially initialize the objective function with a trainer and hparams to optimize.

    Optimize using the optuna library.

    :param trainer:
    :param n_trials:
    :param maximize_objective:
    :param wandb_kwargs:
    :return:
    """
    assert hasattr(trainer, "fit")

    if wandb_kwargs is None:
        wandb_kwargs = {"project": "temp"}

    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    study = optuna.create_study(direction="maximize" if maximize_objective else "minimize")
    obj = partial(objective, trainer)

    try:
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, callbacks=[wandbc])
    except:
        pass
    finally:
        joblib.dump(study, save_study_path+".pkl")

    return study


def print_optuna_stats(study: optuna.Study):
    pruned_trials = study.get_trials(deepcopy=False,
                                     states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False,
                                       states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
