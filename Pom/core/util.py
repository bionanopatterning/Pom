import numpy as np



def image_to_boxes(image, boxsize=128, overlap=0.5, normalize=None):
    w, h = image.shape[0:2]
    pad_w = 0
    pad_h = 0
    if not boxsize == w:
        pad_w = boxsize - (w % boxsize)
    if not boxsize == h:
        pad_h = boxsize - (h % boxsize)
    stride = int(boxsize * (1.0 - overlap))
    boxes = list()
    image = np.pad(image, ((0, pad_w), (0, pad_h), (0, 0)), mode='symmetric')
    for x in range(0, w + pad_w - boxsize + 1, stride):
        for y in range(0, h + pad_h - boxsize + 1, stride):
            box = image[x:x + boxsize, y:y + boxsize, :]
            boxes.append(box)

    boxes = np.array(boxes)
    if normalize is not None:
        for j, n in enumerate(normalize):
            if n:
                for k in range(boxes.shape[0]):
                    _img = boxes[k, :, :, j]
                    _img -= np.mean(_img)
                    _img /= np.std(_img)
                    boxes[k, :, :, j] = _img

    return boxes, (w, h), (pad_w, pad_h), stride


def boxes_to_image(boxes, imgsize, padding, stride):
    pad_w, pad_h = padding
    w, h = imgsize
    boxsize, _, n_features = boxes[0].shape
    out_image = np.zeros((w + pad_w, h + pad_h, n_features))
    count = np.zeros((w + pad_w, h + pad_h), dtype=int)

    i = 0
    for x in range(0, w + pad_w - boxsize + 1, stride):
        for y in range(0, h + pad_h - boxsize + 1, stride):
            out_image[x:x + boxsize, y:y + boxsize, :] += boxes[i]
            count[x:x + boxsize, y:y + boxsize] += 1
            i += 1
    c_mask = count == 0
    count[c_mask] = 1
    for k in range(n_features):
        out_image[:, :, k] /= count
    out_image = out_image[:w, :h, :]
    return out_image


def bin_vol(A, b=2):
    i, j, k = A.shape
    return A[:i//b * b, :j//b * b, :k//b * b].reshape((i//b, b, j//b, b, k//b, b)).mean(5).mean(3).mean(1)


def bin_img(A, b=2):
    i, j = A.shape
    return A[:i//b * b, :j//b * b].reshape((i//b, b, j//b, b)).mean(3).mean(1)



