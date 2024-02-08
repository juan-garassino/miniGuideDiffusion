from miniGuideDiffusion.model.context import ContextUnet
from miniGuideDiffusion.model.diffusor import DDPM
from miniGuideDiffusion.manager.manager import Manager
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# from typing import Dict, Tuple
from tqdm import tqdm
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import save_image, make_grid
import torch
import torch.utils.data as data_utils

# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
from colorama import Fore, Style
from datetime import datetime

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script for MNIST')

    # Add arguments to the parser
    parser.add_argument('--COLAB', type=int, default=0, help='Flag indicating if training is done in a Colab environment')
    parser.add_argument('--DRIVE', type=int, default=0, help='Flag indicating whether to save the trained model in Google Drive or not')
    parser.add_argument('--DATASET', type=str, default='digits', help='Name of the dataset being used')
    parser.add_argument('--DATASET_SIZE', type=int, default=12, help='Size of the dataset (number of samples)')
    parser.add_argument('--N_EPOCHS', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--ANIMATION_STEP', type=int, default=1, help='Step size for generating animations')
    parser.add_argument('--BATCH_SIZE', type=int, default=2, help='Batch size used during training')
    parser.add_argument('--N_DIFFUSION_STEPS', type=int, default=10, help='Number of diffusion steps (iterations) in the training process')
    parser.add_argument('--DEVICE', type=str, default='cpu', help='Device used for computation (\'cpu\' or \'cuda\')')
    parser.add_argument('--N_CLASSES', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--N_FEATURES', type=int, default=128, help='Number of features; dimensionality of the feature space (128 recommended, 256 better but slower)')
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-4, help='Learning rate used in optimization')
    parser.add_argument('--SAVE_MODEL', type=int, default=1, help='Flag indicating whether to save the trained model')
    parser.add_argument('--LOAD_MODEL', type=int, default=0, help='Flag indicating whether to load a pre-trained model')
    parser.add_argument('--SAVE_DIR', type=str, default='./data/diffusion_outputs10/', help='Directory to save the model outputs')
    parser.add_argument('--PLOT_SIZE', type=str, default='8,16', help='Size of plots (e.g., [height, width])')
    parser.add_argument('--WS_TEST', type=str, default='0.0,0.5,2.0', help='Strength of generative guidance for testing')
    parser.add_argument('--N_SAMPLES', type=int, default=2, help='Number of samples used in the application')

    # Parse the arguments
    return parser.parse_args()


def train_mnist(
    colab=0,  # Flag indicating if training is done in a Colab environment
    drive=0,  # Flag indicating if training is done in a Colab environment
    dataset='digits',  # Name of the dataset being used
    dataset_size=12,  # Size of the dataset (number of samples)
    n_epochs=1,  # Number of epochs for training
    animation_step=1,  # Step size for generating animations
    batch_size=2,  # Batch size used during training
    n_diffusion_steps=10,  # Number of diffusion steps (iterations) in the training process
    device='cpu',  # Device used for computation ('cpu' or 'cuda')
    n_classes=10,  # Number of classes in the dataset
    n_features=128,  # Number of features; dimensionality of the feature space (128 recommended, 256 better but slower)
    learning_rate=1e-4,  # Learning rate used in optimization
    save_model=1,  # Flag indicating whether to save the trained model
    load_model=0,  # Flag indicating whether to load a pre-trained model
    save_dir='./data/diffusion_outputs10/',  # Directory to save the model outputs
    plot_size='8,16',  # Size of plots (e.g., [height, width])
    ws_test='0.0,0.5,2.0',  # Strength of generative guidance for testing
    n_samples=2  # Number of samples used in the application
):
    """
    Train a model on the MNIST dataset with specified configurations.

    Parameters:
    - colab (int): Flag indicating if training is done in a Colab environment.
    - dataset (str): Name of the dataset being used.
    - dataset_size (int): Size of the dataset (number of samples).
    - n_epochs (int): Number of epochs for training.
    - animation_step (int): Step size for generating animations.
    - batch_size (int): Batch size used during training.
    - diffusion_steps (int): Number of diffusion steps (iterations) in the training process.
    - device (str): Device used for computation ('cpu' or 'cuda').
    - n_classes (int): Number of classes in the dataset.
    - n_features (int): Number of features; dimensionality of the feature space.
    - learning_rate (float): Learning rate used in optimization.
    - save_model (int): Flag indicating whether to save the trained model.
    - load_model (int): Flag indicating whether to load a pre-trained model.
    - save_dir (str): Directory to save the model outputs.
    - plot_size (list): Size of plots [height, width].
    - ws_test (list): Strength of generative guidance for testing.
    - n_samples (int): Number of samples used in the application.

    Returns:
    None
    """

    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=1,
            n_feat=int(n_features),
            n_classes=int(n_classes),
        ),
        betas=(1e-4, 0.02),
        n_diffusion_steps=int(n_diffusion_steps),
        device=device,
        drop_prob=0.1,
    )

    ddpm.to(device)

    plot_size = plot_size.split(',')

    # optionally load a model

    if load_model == 1:
        ddpm.load_state_dict(
            torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth")
        )

    tf = transforms.Compose(
        [transforms.ToTensor()]
    )  # mnist is already normalised 0 to 1

    output_directories = Manager.output_directories(colab, drive)

    for out_dir in output_directories:

        out_dir = out_dir + '/data'

        Manager.make_directory(out_dir)

        if dataset == "digits":
            dataset = MNIST(out_dir, train=True, download=True,
                            transform=tf)

        if dataset == "fashion":
            dataset = FashionMNIST(
                out_dir, train=True, download=True,
                transform=tf)

        else:
            print("No data has been loaded")

    indices = torch.arange(int(dataset_size))

    dataset = data_utils.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=1#5,
    )

    optim = torch.optim.Adam(
        ddpm.parameters(), lr=float(learning_rate)
    )

    for ep in range(int(n_epochs)):

        print("\n⏹ " + Fore.MAGENTA + f"epoch {ep}" + Style.RESET_ALL)

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = float(learning_rate) * (
            1 - ep / int(n_epochs)
        )

        pbar = tqdm(dataloader)
        loss_ema = None

        print("\n")

        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(
                "⏹ " + Fore.CYAN + f"loss: {loss_ema:.4f}" + Style.RESET_ALL
            )
            optim.step()

        print('\n')

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()

        with torch.no_grad():
            n_sample = int(n_samples) * int(
                n_classes
            )
            for w_i, w in enumerate(ws_test.split(",")):
                x_gen, x_gen_store = ddpm.sample(
                    n_sample, (1, 28, 28), device, guide_weight=float(w)
                )

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(int(n_classes)):
                    for j in range(int(n_sample / int(n_classes))):
                        try:
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * int(n_classes))] = x[idx]

                x_all = torch.cat([x_gen, x_real])

                grid = make_grid(x_all * -1 + 1, nrow=10)

                output_directories = Manager.output_directories(colab, drive)

                for out_dir in output_directories:

                    out_dir = out_dir + '/results/snapshots'

                    Manager.make_directory(out_dir)  # os.environ.get("SAVE_DIR"))

                    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

                    save_image(
                        grid,
                        out_dir + f"/image_ep{ep}_w{w}[{now}].png",  # os.environ.get("SAVE_DIR") +
                    )

                    print(
                        "\n⏹ " + Fore.BLUE + "saved image @ " + out_dir +
                        f"/image_ep{ep}_w{w}[{now}].png"  # os.environ.get("SAVE_DIR") +
                        + Style.RESET_ALL)

                if ep % int(animation_step) == 0 or ep == int(
                    int(n_epochs) - 1
                ):

                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(
                        nrows=int(n_sample / int(n_classes)),
                        ncols=int(n_classes),
                        sharex=True,
                        sharey=True,
                        figsize=(int(plot_size[0]), int(plot_size[1])),
                    )

                    def animate_diff(i, x_gen_store):

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

                    ani = FuncAnimation(
                        fig,
                        animate_diff,
                        fargs=[x_gen_store],
                        interval=200,
                        blit=False,
                        repeat=True,
                        frames=x_gen_store.shape[0],
                    )

                    for out_dir in output_directories:

                        out_dir = out_dir + '/results/animations'

                        Manager.make_directory(out_dir)

                        now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

                        ani.save(
                            out_dir
                            + f"/gif_ep{ep}_w{w}[{now}].gif",  # os.environ.get("SAVE_DIR") +
                            dpi=100,
                            writer=PillowWriter(fps=5),
                        )

                        print("\n⏹ " + Fore.RED + "saved gif @ " + out_dir +
                              f"/gif_ep{ep}_w{w}[{now}].gif" + Style.RESET_ALL)

        # optionally save model
        if (
            int(save_model) == 1
        ):  # and ep == int(n_epochs - 1):

            output_directories = Manager.output_directories(colab, drive)

            for out_dir in output_directories:

                out_dir = out_dir + '/results/checkpoints'

                Manager.make_directory(out_dir)  # os.environ.get("SAVE_DIR"))

                now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

                torch.save(
                    ddpm.state_dict(),
                    out_dir + f"/model_{ep}[{now}].pth",  # os.environ.get("SAVE_DIR")
                )

                print("\n⏹ " + Fore.YELLOW + "saved model @ " + out_dir +
                      f"/model_{ep}[{now}].pth"  # os.environ.get("SAVE_DIR") +
                      + Style.RESET_ALL)

if __name__ == "__main__":
    try:

        args = parse_arguments()

        print(args)

        train_mnist(
            colab=args.COLAB,
            drive=args.DRIVE,
            dataset=args.DATASET,
            dataset_size=args.DATASET_SIZE,
            n_epochs=args.N_EPOCHS,
            animation_step=args.ANIMATION_STEP,
            batch_size=args.BATCH_SIZE,
            n_diffusion_steps=args.N_DIFFUSION_STEPS,
            device=args.DEVICE,
            n_classes=args.N_CLASSES,
            n_features=args.N_FEATURES,
            learning_rate=args.LEARNING_RATE,
            save_model=args.SAVE_MODEL,
            load_model=args.LOAD_MODEL,
            save_dir=args.SAVE_DIR,
            plot_size=args.PLOT_SIZE,
            ws_test=args.WS_TEST,
            n_samples=args.N_SAMPLES
        )

    except:
        import ipdb, traceback, sys

        extype, value, tb = sys.exc_info()

        traceback.print_exc()

        ipdb.post_mortem(tb)
