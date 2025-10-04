from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np


def evaluate_hsi_metrics(G_img, gt_img):
    G_img = G_img.cpu().numpy().transpose(0, 2, 3, 1)
    gt_img = gt_img.cpu().numpy().transpose(0, 2, 3, 1)
    PSNR = 0.0
    SSIM = 0.0
    for i in range(G_img.shape[0]):
        for j in range(G_img.shape[3]):  # G_img.shape[3]是通道数
            # PSNR的计算是各个通道的PSNR取平均
            PSNR = psnr(gt_img[i, :, :, j], G_img[i, :, :, j], data_range=1.0) + PSNR
        SSIM = ssim(gt_img[i, :, :, :], G_img[i, :, :, :], channel_axis=-1, data_range=1.0) + SSIM
    SSIM = SSIM / G_img.shape[0]  # 平均到每张图片
    PSNR = PSNR / G_img.shape[0] / G_img.shape[3]  # 平均到每张图片的每个通道
    SAM = compare_sam(gt_img, G_img)
    metrics = {'PSNR': PSNR, 'SSIM': SSIM, 'SAM': SAM}
    return metrics

def compare_sam(x_true, x_pred):
    sam_deg = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for num_batch in range(x_true.shape[0]):
        sum_sam = 0
        num = 0
        for x in range(x_true.shape[1]):
            for y in range(x_true.shape[2]):
                tmp_pred = x_pred[num_batch, x, y].ravel()
                tmp_true = x_true[num_batch, x, y].ravel()
                if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                    sum_sam += np.arccos(
                        np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                    num += 1
        sam_deg += sum_sam / num
    sam_deg = sam_deg / x_true.shape[0]
    return sam_deg
