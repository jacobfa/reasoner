import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import NCN
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torchvision.models import ViT_B_16_Weights

def main():
    # Number of GPUs to use
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} GPUs')
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

def train(rank, world_size):
    # Setup for DDP
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Use localhost
    os.environ['MASTER_PORT'] = '29500'      # Arbitrary port
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    cudnn.benchmark = True

    # Paths
    data_dir = '/data/jacob/ImageNet'  # Updated path for standard ImageNet

    # Hyperparameters
    num_classes = 1000  # ImageNet has 1000 classes
    concept_dim = 256
    num_concepts = 500  # Adjusted for computational feasibility
    batch_size = 128    # Larger batch size for better GPU utilization
    num_epochs = 90     # Standard number of epochs for ImageNet training
    backbone_lr = 1e-4  # Slightly higher LR for backbone due to larger dataset
    other_lr = 1e-3     # Higher LR for other components
    weight_decay = 1e-4

    # Data Transforms
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    # Data Loading
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=preprocess)
    val_dataset = datasets.ImageFolder(val_dir, transform=preprocess)

    # Distributed Samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # Model
    model = NCN(num_classes=num_classes, concept_dim=concept_dim, num_concepts=num_concepts, pretrained=True)
    model = model.to(device)

    # Optional: Freeze backbone for initial epochs
    model.freeze_backbone()

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.AdamW([
        {'params': model.module.transformer_backbone.parameters(), 'lr': backbone_lr},
        {'params': model.module.concept_bank.parameters(), 'lr': other_lr},
        {'params': model.module.attention.parameters(), 'lr': other_lr},
        {'params': model.module.gnn.parameters(), 'lr': other_lr},
        {'params': model.module.gnn_norm.parameters(), 'lr': other_lr},
        {'params': model.module.aggregation_attention.parameters(), 'lr': other_lr},
        {'params': model.module.classifier.parameters(), 'lr': other_lr}
    ], weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Logging
    if rank == 0:
        log_file = open("log.txt", "w")
        log_file.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")
    
    for epoch in range(num_epochs):
        # Unfreeze backbone after certain epochs
        if epoch == 10:
            model.module.unfreeze_backbone()
        
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)

        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch')
        else:
            pbar = None

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*correct/total:.2f}%"})

        if rank == 0:
            pbar.close()

        # Aggregate statistics from all processes
        total_loss = torch.tensor(running_loss, device=device)
        total_correct = torch.tensor(correct, device=device)
        total_samples = torch.tensor(total, device=device)

        dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_samples, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            epoch_loss = total_loss.item() / len(train_dataset)
            epoch_acc = 100. * total_correct.item() / total_samples.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        if rank == 0:
            pbar = tqdm(total=len(val_loader), desc=f"Validation", unit='batch')
        else:
            pbar = None

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*correct/total:.2f}%"})

        if rank == 0:
            pbar.close()

        # Aggregate validation stats
        total_val_loss = torch.tensor(val_loss, device=device)
        total_val_correct = torch.tensor(correct, device=device)
        total_val_samples = torch.tensor(total, device=device)

        dist.reduce(total_val_loss, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_val_correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_val_samples, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_epoch_loss = total_val_loss.item() / len(val_dataset)
            val_epoch_acc = 100. * total_val_correct.item() / total_val_samples.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
            # Logging
            log_file.write(f"{epoch+1},{epoch_loss:.4f},{epoch_acc:.2f},{val_epoch_loss:.4f},{val_epoch_acc:.2f}\n")
            log_file.flush()

        # Step the scheduler
        scheduler.step()

    if rank == 0:
        log_file.close()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
