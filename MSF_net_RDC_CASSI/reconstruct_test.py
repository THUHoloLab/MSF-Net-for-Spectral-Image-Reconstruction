import torch
import os
from get_coded_aperture import get_coded_aperture
import hdf5storage as hdf5
from args import parse_args
from load_dataset import load_test_dataset
from my_model import MSF_net_fusion

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = parse_args()
    CUDA_DEVICE = 0
    epoch_of_model = 2500
    torch.cuda.empty_cache()
    mask_CASSI = get_coded_aperture(args.image_size, args.spectral_channel)
    mask_CASSI = torch.tensor(mask_CASSI).float()
    mask_gray = torch.ones_like(mask_CASSI)
    mask_CASSI = mask_CASSI.cuda(CUDA_DEVICE)
    mask_gray = mask_gray.cuda(CUDA_DEVICE)
    model = MSF_net_fusion(innput_channels=2, spectral_channels=29).cuda(CUDA_DEVICE)
    test_hsi = load_test_dataset(args.test_dir)
    test_hsi = test_hsi.cuda(CUDA_DEVICE)
    test_measurement_gray = torch.sum(test_hsi * mask_gray, dim=1, keepdim=True) / args.spectral_channel
    test_measurement_CASSI = torch.sum(test_hsi * mask_CASSI, dim=1, keepdim=True) / args.spectral_channel
    test_measurement = torch.cat((test_measurement_gray, test_measurement_CASSI,
                                  0.25 * mask_CASSI.expand(test_measurement_gray.shape[0], -1, -1, -1)), dim=1).cuda(
        CUDA_DEVICE)
    checkpoint = torch.load(os.path.join(args.net_save_dir, 'GPU%d_model_epoch_%d.pth' % (CUDA_DEVICE, epoch_of_model)))
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.eval()
    with torch.no_grad():
        testHSI_pred = model(test_measurement)
        testHSI_pred = torch.squeeze(testHSI_pred)
        testHSI_pred = {"HSI_pred_test": testHSI_pred.cpu().numpy()}
        hdf5.savemat('reconstruction.mat', testHSI_pred)


if __name__ == '__main__':
    main()
