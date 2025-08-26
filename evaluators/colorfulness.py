import numpy as np
import cv2
import torch


class ColorfulEvaluator:

    @torch.no_grad()
    def __call__(self, 
                img=None, 
                src_img=None,
                **kwargs):
        res = dict()
        if img.size != src_img.size:
            img = img.resize(src_img.size)

        res['colorfulness'] = self.image_colorfulness(img)
        return res
    

    def image_colorfulness(self, img):
        img = np.asarray(img, dtype=np.float32)
        img = img / 255.
        # split the image into its respective RGB components
        (R, G, B) = cv2.split(img)
        # compute rg = R - G
        rg = np.absolute(R - G)
        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)
        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))
        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)