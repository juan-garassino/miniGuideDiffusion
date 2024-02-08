import errno
from colorama import Fore, Style
import os


class Manager:  # make manager work with and with out epochs
    def __init__():
        pass

    @staticmethod
    def output_directories(colab):

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

        if int(colab) == 1 and int(
                os.environ.get('DRIVE')) == 1:

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
