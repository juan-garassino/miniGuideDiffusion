import errno
from colorama import Fore, Style
import os


class Manager:  # make manager work with and with out epochs
    def __init__():
        pass

    @staticmethod
    def output_directories(colab, drive):

        output_directories = []

        if int(colab) == 1:

            out_dir = os.path.join(
                os.environ.get("HOME"),
                "..",
                "content",
                "miniGuideDiffusion",
                #"checkpoints",
            )

            output_directories.append(out_dir)

        if int(colab) == 1 and int(drive) == 1:

            out_dir = os.path.join(
                os.environ.get("HOME"),
                "..",
                "content",
                "drive",
                "MyDrive",
                "repositories",
                "miniGuideDiffusion",
                #"checkpoints",
            )

            output_directories.append(out_dir)

        if int(colab) == 0:

            out_dir = os.path.join(
                os.environ.get("HOME"),
                "Code",
                "juan-garassino",
                "miniGuideDiffusion",
                #"checkpoints",
            )

            output_directories.append(out_dir)

        return output_directories

    @staticmethod
    def make_directory(directory):
        try:
            os.makedirs(directory)

            print(
                "\n⏹ "
                + Fore.GREEN
                + f"This directory has been created {directory}"
                + Style.RESET_ALL
            )

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def animate_diff(i, n_sample, n_classes, axs, x_gen_store): # remove the function from trainer.py and put it here

        print(
            "⏹ "
            + Fore.MAGENTA
            + f"gif animating frame {i} of {x_gen_store.shape[0]}"
            + Style.RESET_ALL,
            end="\r",
        )

        plots = []

        for row in range(
            int(n_sample / int(n_classes))
        ):
            for col in range(int(n_classes)):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                plots.append(
                    axs[row, col].imshow(
                        -x_gen_store[
                            i,
                            (row * int(n_classes))
                            + col,
                            0,
                        ],
                        cmap="gray",
                        vmin=(-x_gen_store[i]).min(),
                        vmax=(-x_gen_store[i]).max(),
                    )
                )
        return plots
