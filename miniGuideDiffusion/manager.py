import errno
from colorama import Fore, Style
import os

class Manager:  # make manager work with and with out epochs
    def __init__():
        pass

    @staticmethod
    def make_directory(directory):
        try:
            os.makedirs(directory)

            print(
                "\n\t⏹\t"
                + Fore.GREEN
                + f"This directory has been created {directory}"
                + Style.RESET_ALL
            )

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
