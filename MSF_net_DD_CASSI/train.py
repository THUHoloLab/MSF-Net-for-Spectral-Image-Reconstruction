import torch
from torchvision.utils import save_image
import os
import time
from get_coded_aperture import get_coded_aperture
from args import parse_args
from load_dataset import DatasetFromFolder, load_test_dataset
from get_metrics import evaluate_hsi_metrics
from my_model import MSF_net_fusion


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def add_gaussian_noise(image, std=0.01, clip=(0., 1.), seed=None, mode="train"):
    if mode == "test" and seed is not None:
        gen = torch.Generator(device=image.device).manual_seed(seed)
        noise = torch.normal(0.0, std, size=image.shape, device=image.device, generator=gen)
    else:
        noise = torch.normal(0.0, std, size=image.shape, device=image.device)
    return torch.clamp(image + noise, *clip)

def train_model(args):
    for path in [args.net_save_dir,args.image_save_dir]:
        os.makedirs(path,exist_ok=True)
    CUDA_DEVICE=0
    torch.cuda.empty_cache()
    mask_cassi=get_coded_aperture(args.image_size,args.spectral_channel)
    mask_cassi = torch.tensor(mask_cassi).float()
    mask_cassi = mask_cassi.cuda(CUDA_DEVICE)
    model = MSF_net_fusion(innput_channels=1, spectral_channels=args.spectral_channel).cuda(CUDA_DEVICE)
    test_hsi = load_test_dataset(args.test_dir)
    test_hsi = test_hsi.cuda(CUDA_DEVICE)
    test_measurement_cassi = torch.sum(test_hsi * mask_cassi, dim=1, keepdim=True,) / args.spectral_channel
    test_measurement=torch.cat((test_measurement_cassi,0.25*mask_cassi.expand(test_measurement_cassi.shape[0], -1, -1, -1)), dim=1).cuda(CUDA_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    train_dataset = DatasetFromFolder(args.train_dir, args.image_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batchSize,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory)
    mse_loss = torch.nn.MSELoss().cuda(CUDA_DEVICE)
    start = time.time()
    model.train()

    for epoch in range(args.n_epoch):
        for train_hsi_batch in train_loader:
            train_hsi_batch = train_hsi_batch.cuda(CUDA_DEVICE)
            measurement_cassi = torch.sum(train_hsi_batch * mask_cassi, dim=1, keepdim=True) / args.spectral_channel
            optimizer.zero_grad()
            model_out = model(torch.cat([measurement_cassi,0.25*mask_cassi.expand(train_hsi_batch.shape[0], -1, -1, -1)],dim=1))
            loss = mse_loss(model_out, train_hsi_batch)
            loss.backward()
            optimizer.step()
        if epoch % args.save_freq == 0:
            save_dict = {
                "epoch": epoch,
                "args": args,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(args.net_save_dir, 'GPU%d_model_epoch_%d.pth' % (CUDA_DEVICE, epoch)))
        if epoch % args.log_freq == 0:
            model.eval()
            with torch.no_grad():
                test_hsi_pred = model(test_measurement)
                loss_test_value = mse_loss(test_hsi_pred, test_hsi).to("cpu").numpy()
                metrics_value = evaluate_hsi_metrics(test_hsi_pred, test_hsi)
                loss_train_value = loss.detach().to("cpu").numpy()
                save_image(test_hsi_pred[2, 15, :, :], args.image_save_dir + "/GPU%d_%d.png" % (CUDA_DEVICE, epoch),
                           normalize=True)
                print('epoch:', epoch, 'L2:', loss_train_value, loss_test_value,
                      'PSNR:', metrics_value['PSNR'], 'SSIM:', metrics_value['SSIM'], 'SAM:', metrics_value['SAM'])
                model.train()
    end = time.time()
    print(end - start)


def main():
    args = parse_args()
    train_model(args)


if __name__ == '__main__':
    main()
