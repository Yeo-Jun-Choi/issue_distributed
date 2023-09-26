from functools import partial
from tqdm import tqdm
import os
import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import CLIPImageProcessor
from torchvision import transforms
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from model import UNet, DiffusionModel
from utils import make_grid

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument("--local-rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=401)
    parser.add_argument('--root', type=str, default='data')
    return parser

def init_for_distributed(opts):

    opts.global_rank = int(os.environ['RANK'])
    opts.local_rank = int(os.environ['LOCAL_RANK'])
    opts.world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(opts.local_rank)
    if opts.global_rank is not None and opts.local_rank is not None:
        print("Use GPU: [{}/{}] for training".format(opts.global_rank, opts.local_rank))

    dist.init_process_group(backend="nccl")
    torch.distributed.barrier()
    print('opts :',opts)
    return

class VideoDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
        self.jitter = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
        ])
        # ToTensor 변환을 위한 추가
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.data)
    
    def transform(self, tensor_image):
        # PIL Image로 변환
        pil_image = transforms.ToPILImage()(tensor_image)
        # PIL Image 전처리 후 텐서로 변환
        return self.to_tensor(self.jitter(pil_image))
    
    def __getitem__(self, idx):
        video = self.data[idx]
        # 각 프레임에 대해 변환을 적용
        transformed_video = torch.stack([self.transform(frame) for frame in video])
        return {'img_input': transformed_video}

def load_state(model, optim, sche, path = 'checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    sche.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model. optim, sche, epoch, loss

def main(opts):
    init_for_distributed(opts)
    #torch.distributed.barrier()

    # load model
    model = UNet(in_dim=64,
                dim_mults = (1, 2, 4, 8, 16),
                is_attn = (False, False, False, True, True)
                )
    diffusion = DiffusionModel(model = model,
                            num_timesteps=1_000)
    # load data
    all_frames = torch.load('/data/csc0411/vidio_8frame.pt')
    data = all_frames.view(-1, 8, 3, 240, 320)
    train_data = data[:-16]
    
    test_data = data[9984:]
    test_data = test_data[:, 0, :, :, :].squeeze()
    test_data = test_data.float() / 255.0
    
    dataset = VideoDataset(train_data)
    train_sampler = DistributedSampler(dataset=dataset, shuffle=True)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, 16, drop_last=True)
    train_dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000)
    diffusion.cuda(opts.local_rank)
    diffusion = DistributedDataParallel(diffusion, device_ids=[opts.local_rank])
    
    
    best_loss = 100
    origin_grid = make_grid(test_data, 4, 4)
    origin_grid.save(f"video_sample/origin.png")
    
    # load check_points 튕겨서 내가 만든거
    save_path = 'checkpoint.pth'
    path_exist = os.path.exists(save_path)
    epochs = 1000
    '''
    if path_exist:
        checkpoint = torch.load('checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epochs -= int(checkpoint['epoch'])
        best_loss= checkpoint['loss']
        print(f'Load Best loss: {best_loss}')
    '''
    
    for epoch in range(epochs):
        # train
        print(f"{epoch}th epoch training...")
        train_sampler.set_epoch(epoch)
        loss_total = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            data = batch['img_input'].to(opts.local_rank)
            optimizer.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optimizer.step()
            loss_total += loss
        train_avg_loss = loss_total/len(train_dataloader)
        print(f"train_loss: {train_avg_loss}, lr: {scheduler.get_last_lr()}")
        loss_total = 0
        scheduler.step()
        # eval
        if train_avg_loss < best_loss:
            best_loss = train_avg_loss
            with torch.no_grad():
                x = diffusion.sample_input(test_data.to(opts.local_rank))
            imgs_grid = make_grid(x, 4, 4)
            imgs_grid.save(f"video_sample/sample_{epoch}.png")
            '''
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
            # 다른 필요한 정보들도 추가로 저장 가능
            }, save_path)
            '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser('vgg11 cifar training', parents=[get_args_parser()])
    opts = parser.parse_args()
    main(opts)
