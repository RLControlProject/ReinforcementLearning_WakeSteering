from pathlib import Path

import matplotlib.pyplot as plt

from multiprocessing import Pool
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({'font.size': 22}) # Sets default font size to 14
plt.rc('font', size=22)

THISDIR = Path(__file__).parent.resolve()

namedict = {"SmallFarm": "", "MidFarm": "_medium"}


def generatecsv():
    allinputs = []
    for case in allcases:
        for farm in allfarms:
            for seedset in allseedsets:
                for trial in alltrials:
                    for var in allvars:
                        logpath = Path(
                            THISDIR,
                            "..",
                            f"RunFiles_{case}",
                            farm,
                            "Runs",
                            "Group1",
                            f"Seed{seedset}",
                            "yawseeds1",
                            "sac_output",
                            "trials",
                            f"trial_{trial}_10env{namedict[farm]}_1_1/",
                        )
                        csvpath = Path(
                            THISDIR,
                            "..",
                            f"RunFiles_{case}",
                            farm,
                            "Runs",
                            "Group1",
                            f"Seed{seedset}",
                            "yawseeds1",
                            "sac_output",
                            "trials",
                            f"trial_{trial}_10env{namedict[farm]}_1_1/",
                            f"{var}.csv",
                        )
                        currentinput = [str(logpath), f"train/{var}", str(csvpath)]
                        allinputs.append(currentinput)
                        # export_tensorboard_to_csv(logpath, f"train/{var}", csvpath)
                        # pass

    with Pool() as pool:
        pool.starmap(export_tensorboard_to_csv, allinputs)


def export_tensorboard_to_csv(log_dir, tag, output_csv_path):
    """
    Exports scalar data for a specific tag from TensorBoard logs to a CSV file.

    Args:
        log_dir (str): Path to the TensorBoard log directory.
        tag (str): The tag of the scalar data to export (e.g., 'accuracy/train').
        output_csv_path (str): Path to save the output CSV file.
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get all scalar events for the specified tag
    scalar_events = event_acc.Scalars(tag)

    # Create a list of dictionaries for DataFrame creation
    data = [{"step": s.step, "value": s.value} for s in scalar_events]

    # Create a Pandas DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data for tag '{tag}' exported to {output_csv_path}")


def generateplots():
    # colors = [(0.25,0.25,0.25),(0.75,0.75,0.75),(167/256, 199/256, 231/256),(171/256,55/256,46/256),(208/256,136/256,33/256),(0.2, 0.5, 0.8)]
    colordict = {
        3: (171 / 256, 55 / 256, 46 / 256),
        4: (208 / 256, 136 / 256, 33 / 256),
        5: (0.2, 0.5, 0.8),
    }
    dashdict = {4: "solid", 5: "dashed", 6: "dashdot"}
    xlimdict = {"SmallFarm": [0, 1000], "MidFarm": [0, 1500]}
    ntrials = len(alltrials)
    for case in allcases:
        for farm in allfarms:
            for var in allvars:
                f, axs = plt.subplots(1, 1)
                plt.gcf().set_size_inches(13, 6)
                # axs = [axs] if ntrials == 1 else axs
                axs = [axs]
                for trial in alltrials:
                    for i, ax in enumerate(axs):
                        for seedset in allseedsets:
                            df = pd.read_csv(
                                Path(
                                    THISDIR,
                                    "..",
                                    f"RunFiles_{case}",
                                    farm,
                                    "Runs",
                                    "Group1",
                                    f"Seed{seedset}",
                                    "yawseeds1",
                                    "sac_output",
                                    "trials",
                                    f"trial_{trial}_10env{namedict[farm]}_1_1/",
                                    f"{var}.csv",
                                )
                            )
                            ax.plot(
                                df["step"],
                                df["value"],
                                color=colordict[trial],
                                linestyle=dashdict[seedset],
                                label=f"{allagentsdict[trial]}, Seedset {seedset-3}",
                            )
                        # ax.grid("True")
                        ax.set_ylabel(f"{allvars[var]}")
                        ax.legend(fontsize=18,bbox_to_anchor=(1.05, 1.03), loc="upper left")
                        ax.set_xlim(xlimdict[farm])
                    axs[-1].set_xlabel("Episode number")
                    plt.tight_layout()
                    f.savefig(f"{case}-{farm}-{var}.svg")
        pass


if __name__ == "__main__":
    allcases = ["Oneseed", "Standard"]
    allfarms = ["SmallFarm", "MidFarm"]
    allseedsets = [4, 5, 6]
    alltrials = [3, 4, 5]
    allvars = {
        "actor_loss" : "Actor loss [-]",
        "critic_loss": "Critic loss [-]",
        "ent_coef": "Entropy coeff. [-]",
        "ent_coef_loss": "Entropy coeff. loss [-]",
        "learning_rate": "Learning rate [-]",
    }
    allagentsdict = {3: "ST", 4: "PIF", 5: "PIF+WS"}

    # generatecsv()
    generateplots()
