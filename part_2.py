import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb
import json
from torchvision.models import resnet34, ResNet34_Weights

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = resnet34(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)
    
    def forward(self, x):
        return self.model(x)

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):
        
        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here

        optimizer.zero_grad()
        lam = np.random.beta(1.0, 1.0)
        index = torch.randperm(inputs.size(0)).to(device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        targets_a, targets_b = labels, labels[index]

        outputs = model(mixed_inputs)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients

        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):

            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) ### TODO -- inference
            loss = criterion(outputs, labels) ### TODO -- loss calculation

            running_loss += loss.item() ### SOLUTION -- add loss from this sample
            _, predicted = torch.max(outputs.data, 1) ### SOLUTION -- predict the class
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def main():
    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": "resnet34_pretrained", 
        "pretrained": False,
        "batch_size": 128,
        "learning_rate": 0.05,
        "epochs": 100,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "-sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])

    ###############
    # TODO Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])

    ############################################################################
    #       Data Loading
    ############################################################################


    trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True,
                                            download=True, transform=transform_train)


    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset)) ### TODO -- Calculate training set size
    val_size = len(trainset) - train_size ### TODO -- Calculate validation set size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size]) ### TODO -- split into training and validation sets  
    valset.dataset.transform = transform_test

    ### TODO -- define loaders and test set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], 
                                            shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], 
                                          shuffle=False, num_workers=CONFIG["num_workers"])

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                          download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"],
                                           shuffle=False, num_workers=CONFIG["num_workers"])

    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = ResNet34() # instantiate your model ### TODO
    model = model.to(CONFIG["device"]) # move it to target device

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  ### TODO -- define loss criterion
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9, weight_decay=5e-4)  ### TODO -- define optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"]) # Add a scheduler   ### TODO -- you can optionally add a LR scheduler

    # Initialize wandb
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # Evaluation on Clean CIFAR-100 Test Set
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # Evaluation on OOD
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # Create Submission File (OOD)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()