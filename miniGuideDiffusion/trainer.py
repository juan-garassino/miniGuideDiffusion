"""
This script does conditional image generation on MNIST, using a diffusion model
This code is modified from,
https://github.com/cloneofsimo/minDiffusion
Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239
The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598
This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487
"""
from miniGuideDiffusion.context import ContextUnet
from miniGuideDiffusion.diffusor import DDPM
from miniGuideDiffusion.manager import Manager
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# from typing import Dict, Tuple
from tqdm import tqdm
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import save_image, make_grid
import torch

# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
from colorama import Fore, Style


def train_mnist():

    # hardcoding these here
    """n_epoch = 20
    batch_size = 256
    n_T = 400  # 500
    device = "cpu" #"cuda:0" torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = 10
    n_feat = 128  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './data/diffusion_outputs10/'
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    n_samples = 10"""

    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=1,
            n_feat=int(os.environ.get("N_FEATURES")),
            n_classes=int(os.environ.get("N_CLASSES")),
        ),
        betas=(1e-4, 0.02),
        n_T=int(os.environ.get("N_T")),
        device=os.environ.get("DEVICE"),
        drop_prob=0.1,
    )
    ddpm.to(os.environ.get("DEVICE"))

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose(
        [transforms.ToTensor()]
    )  # mnist is already normalised 0 to 1

    if os.environ.get("DATASET") == "digits":
        dataset = MNIST("./data", train=True, download=True, transform=tf)

    if os.environ.get("DATASET") == "fashion":
        dataset = FashionMNIST("./data", train=True, download=True, transform=tf)

    else:
        print("No data has been loaded")

    dataloader = DataLoader(
        dataset,
        batch_size=int(os.environ.get("BATCH_SIZE")),
        shuffle=True,
        num_workers=5,
    )

    optim = torch.optim.Adam(
        ddpm.parameters(), lr=float(os.environ.get("LEARNING_RATE"))
    )

    for ep in range(int(os.environ.get("N_EPOCHS"))):

        print("\n⏹ " + Fore.BLUE + f"epoch {ep}" + Style.RESET_ALL)

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = float(os.environ.get("LEARNING_RATE")) * (
            1 - ep / int(os.environ.get("N_EPOCHS"))
        )

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(os.environ.get("DEVICE"))
            c = c.to(os.environ.get("DEVICE"))
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(
                "\n⏹ " + Fore.BLUE + f"loss: {loss_ema:.4f}" + Style.RESET_ALL
            )
            optim.step()

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = int(os.environ.get("N_SAMPLES")) * int(
                os.environ.get("N_CLASSES")
            )
            for w_i, w in enumerate(os.environ.get("WS_TEST").split(",")):
                x_gen, x_gen_store = ddpm.sample(
                    n_sample, (1, 28, 28), os.environ.get("DEVICE"), guide_w=float(w)
                )

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(os.environ.get("DEVICE"))
                for k in range(int(os.environ.get("N_CLASSES"))):
                    for j in range(int(n_sample / int(os.environ.get("N_CLASSES")))):
                        try:
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * int(os.environ.get("N_CLASSES")))] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all * -1 + 1, nrow=10)
                Manager.make_directory(os.environ.get("SAVE_DIR"))
                save_image(grid, os.environ.get("SAVE_DIR") + f"image_ep{ep}_w{w}.png")

                print(
                    "\n⏹ "
                    + Fore.BLUE
                    + "saved image at "
                    + os.environ.get("SAVE_DIR")
                    + f"image_ep{ep}_w{w}.png"
                    + Style.RESET_ALL
                )

                if ep % 5 == 0 or ep == int(os.environ.get("N_EPOCHS") - 1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(
                        nrows=int(n_sample / int(os.environ.get("N_CLASSES"))),
                        ncols=int(os.environ.get("N_CLASSES")),
                        sharex=True,
                        sharey=True,
                        figsize=(8, 3),
                    )

                    def animate_diff(i, x_gen_store):

                        print(
                            "\n⏹ "
                            + Fore.PURPLE
                            + f"gif animating frame {i} of {x_gen_store.shape[0]}"
                            + Style.RESET_ALL
                        )

                        plots = []
                        for row in range(
                            int(n_sample / int(os.environ.get("N_CLASSES")))
                        ):
                            for col in range(int(os.environ.get("N_CLASSES"))):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(
                                    axs[row, col].imshow(
                                        -x_gen_store[
                                            i,
                                            (row * int(os.environ.get("N_CLASSES")))
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

                    ani.save(
                        os.environ.get("SAVE_DIR") + f"gif_ep{ep}_w{w}.gif",
                        dpi=100,
                        writer=PillowWriter(fps=5),
                    )

                    print(
                        "\n⏹ "
                        + Fore.RED
                        + "saved image at "
                        + os.environ.get("SAVE_DIR")
                        + f"gif_ep{ep}_w{w}.gif"
                        + Style.RESET_ALL
                    )

        # optionally save model
        if os.environ.get("SAVE_MODEL") and ep == int(os.environ.get("N_EPOCHS") - 1):
            torch.save(
                ddpm.state_dict(), os.environ.get("SAVE_DIR") + f"model_{ep}.pth"
            )

            print(
                "\n⏹ "
                + Fore.YELLOW
                + "saved model at "
                + os.environ.get("SAVE_DIR")
                + f"model_{ep}.pth"
                + Style.RESET_ALL
            )


if __name__ == "__main__":
    train_mnist()
