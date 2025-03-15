import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import wandb
from diffusion import DiffusionModel
from training import generate_samples, train_epoch
from unet import UnetModel
import tqdm
from tqdm import trange
from hparams import config


def main(device: str):
    wandb.init(config=config, project="pipeline_diffusion", name="baseline")

    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=config["hidden_size"]),
        betas=(config["beta_lower"], config["beta_upper"]),
        num_timesteps=config["num_timesteps"],
    )
    ddpm.to(device)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    
    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    
    

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=config["learning_rate"])
    

    
    for i in trange(config["num_epochs"]):
        loss = train_epoch(ddpm, dataloader, optim, device)
        metrics_and_params = {}
        metrics_and_params["loss"] = loss
        outputs = generate_samples(ddpm, device, f"samples/{i:02d}.png")
        metrics_and_params["outputs"] = wandb.Image(outputs)
        wandb.log(metrics_and_params, step=i)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
