import torch
import clip
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch.optim as optim
import numpy as np
from datetime import datetime
import scipy.spatial
import time
import random
import os
from tqdm import tqdm

TIMESTAMP = "{0:%Y_%m_%d}".format(datetime.now())

tic = time.time()
seed = 42
cross_seed = 0
print("Setting experiment random seed to %d", seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
BATCH_SIZE = 128
EPOCH = 10

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
print("device %s" % device)


def i2t(sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            # print(inds, i, index, npts)
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def cosine_similarity_tensor(image_features, text_features):
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    return logits_per_image


def cosine_similarity_numpy(image_features, text_features):
    image_features = image_features / np.linalg.norm(x=image_features, ord=2, axis=1)
    text_features = text_features / np.linalg.norm(x=text_features, ord=2, axis=1)
    logit_scale = np.exp(np.log(1 / 0.07))
    logits_per_image = logit_scale * image_features @ text_features.T
    return logits_per_image


print("load model ...... ")
model, preprocess = clip.load("/data/ftpdir/www/clip/model_cache/ViT-B-32.pt", device=device,
                              jit=False)  # Must set jit=False for training
print("end load model ...... ")
print("load data ...... ")
train_image_path = "/data/ftpdir/www/clip/dataset/train_local.txt"
train_txt_path = "/data/ftpdir/www/clip/dataset/train_caps.txt"
dev_image_path = "/data/ftpdir/www/clip/dataset/dev_local.txt"
dev_txt_path = "/data/ftpdir/www/clip/dataset/dev_caps.txt"
test_1k_image_path = "/data/ftpdir/www/clip/dataset/test_local.txt"
test_1k_txt_path = "/data/ftpdir/www/clip/dataset/test_caps.txt"
test_5k_image_path = "/data/ftpdir/www/clip/dataset/testall_local.txt"
test_5k_txt_path = "/data/ftpdir/www/clip/dataset/testall_caps.txt"
train_image, train_txt, dev_image, dev_txt, test_1k_image, test_1k_txt, test_5k_image, test_5k_txt = [], [], [], [], [], [], [], []

with open(train_image_path, "r") as f:
    for line in f:
        for i in range(5):
            train_image.append(line.replace("\n", ""))
    f.close()
with open(train_txt_path, "r") as f:
    for line in f:
        train_txt.append(line.replace("\n", ""))
    f.close()

with open(dev_image_path, "r") as f:
    for line in f:
        dev_image.append(line.replace("\n", ""))
    f.close()
with open(dev_txt_path, "r") as f:
    for line in f:
        dev_txt.append(line.replace("\n", ""))
    f.close()

with open(test_1k_image_path, "r") as f:
    for line in f:
        test_1k_image.append(line.replace("\n", ""))
    f.close()
with open(test_1k_txt_path, "r") as f:
    for line in f:
        test_1k_txt.append(line.replace("\n", ""))
    f.close()

with open(test_5k_image_path, "r") as f:
    for line in f:
        test_5k_image.append(line.replace("\n", ""))
    f.close()
with open(test_5k_txt_path, "r") as f:
    for line in f:
        test_5k_txt.append(line.replace("\n", ""))
    f.close()


class image_caption_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.caption = clip.tokenize(
            list_txt)  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        caption = self.caption[idx]
        return image, caption


# use your own data
train_dataset = image_caption_dataset(train_image, train_txt)
dev_dataset = image_caption_dataset(dev_image, dev_txt)
test_1k_dataset = image_caption_dataset(test_1k_image, test_1k_txt)
test_5k_dataset = image_caption_dataset(test_5k_image, test_5k_txt)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Define your own dataloader
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)  # Define your own dataloader
test_1k_dataloader = DataLoader(test_1k_dataset, batch_size=BATCH_SIZE)  # Define your own dataloader
test_5k_dataloader = DataLoader(test_5k_dataset, batch_size=BATCH_SIZE)  # Define your own dataloader
print("end load data ...... ")


# https://github.com/openai/CLIP/issues/57
# 解决混合精度的问题，改为32
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6,
                       weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

feature_file_path = "/data/ftpdir/www/clip/test/test_result/" + TIMESTAMP
if not os.path.isdir(feature_file_path):
    os.mkdir(feature_file_path)

# add your own code to track the training progress.
# train
for epoch in range(EPOCH):
    print("start train in epoch %d " % epoch)
    model.train()
    total_loss = 0
    tr_step_num = int(len(train_dataloader))
    train_batch_gen = iter(train_dataloader)
    for batch_idx in tqdm(range(0, tr_step_num), desc='1st loop'):
        optimizer.zero_grad()

        images, texts = next(train_batch_gen)

        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        temp_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        temp_loss.backward()
        total_loss = total_loss + temp_loss

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    print("end epoch train total loss" % total_loss)

    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f"/data/ftpdir/www/clip/model_checkpoint/model_10.pt")  # just change to your preferred folder/filename
    except:
        print("torch save model error")
    ## test5k

    with torch.no_grad():
        model.eval()
        print("start test 1k dev dataset")
        dev_images_feature = []
        dev_texts_feature = []
        all_images_feature_1k = []
        all_texts_feature_1k = []
        for dev_ind, dev_batch_1k in enumerate(dev_dataloader):
            dev_images, dev_texts = dev_batch_1k
            dev_images = dev_images.to(device)
            dev_texts = dev_texts.to(device)

            dev_images_feat = model.encode_image(dev_images)
            dev_texts_feat = model.encode_text(dev_texts)

            dev_images_feature.append(dev_images_feat.cpu().numpy())
            dev_texts_feature.append(dev_texts_feat.cpu().numpy())

        dev_images_feat_np = np.concatenate(dev_images_feature)
        dev_texts_feat_np = np.concatenate(dev_texts_feature)

        dev_images_feat_np = dev_images_feat_np[::5]
        image_to_text_dev = scipy.spatial.distance.cdist(dev_images_feat_np, dev_texts_feat_np, 'cosine')

        i2t_dev = i2t(image_to_text_dev)
        t2i_dev = t2i(image_to_text_dev)

        print("i2t in 1k dev dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % i2t_dev)
        print("t2i in 1k dev dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % t2i_dev)

        print("start test 1k dev dataset")
        for ind, test_batch_1k in enumerate(test_1k_dataloader):
            test_images_1k, test_texts_1k = test_batch_1k
            test_images_1k = test_images_1k.to(device)
            test_texts_1k = test_texts_1k.to(device)

            images_feature = model.encode_image(test_images_1k)
            texts_feature = model.encode_text(test_texts_1k)

            all_images_feature_1k.append(images_feature.cpu().numpy())
            all_texts_feature_1k.append(texts_feature.cpu().numpy())

        all_images_feature_1k = np.concatenate(all_images_feature_1k)
        all_texts_feature_1k = np.concatenate(all_texts_feature_1k)

        all_images_feature_1k = all_images_feature_1k[::5]
        image_to_text_1k = scipy.spatial.distance.cdist(all_images_feature_1k, all_texts_feature_1k, 'cosine')
        # image_to_text_1k = cosine_similarity_numpy(all_images_feature_1k, all_texts_feature_1k)[::5]

        i2t_1k = i2t(image_to_text_1k)
        t2i_1k = t2i(image_to_text_1k)

        print("i2t in 1k test dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % i2t_1k)
        print("t2i in 1k test dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % t2i_1k)

        new_img_feat_file = "/data/ftpdir/www/clip/test/test_result/" + TIMESTAMP + "/images_features_1k.npy"
        new_txt_feat_file = "/data/ftpdir/www/clip/test/test_result/" + TIMESTAMP + "/texts_features_1k.npy"

        np.save(new_img_feat_file, all_images_feature_1k)
        np.save(new_txt_feat_file, all_texts_feature_1k)

# region
# with torch.no_grad():
#     model.eval()
#     print("start test 1k test dataset")
#     all_images_feature_1k = []
#     all_texts_feature_1k = []

#     for ind, test_batch_1k in enumerate(test_1k_dataloader):

#         test_images_1k, test_texts_1k = test_batch_1k
#         test_images_1k = test_images_1k.to(device)
#         test_texts_1k = test_texts_1k.to(device)

#         images_feature = model.encode_image(test_images_1k)
#         texts_feature = model.encode_text(test_texts_1k)


#         all_images_feature_1k.append(images_feature.cpu().numpy())
#         all_texts_feature_1k.append(texts_feature.cpu().numpy())

#     print("get 1k test dataset features ")

#     all_images_feature_1k = np.concatenate(all_images_feature_1k)
#     all_texts_feature_1k = np.concatenate(all_texts_feature_1k)

#     all_images_feature_1k = all_images_feature_1k[::5]
#     image_to_text_1k = scipy.spatial.distance.cdist(all_images_feature_1k, all_texts_feature_1k, 'cosine')
#     # image_to_text_1k = cosine_similarity_numpy(all_images_feature_1k, all_texts_feature_1k)[::5]

#     i2t_1k = i2t(image_to_text_1k)
#     t2i_1k = t2i(image_to_text_1k)

#     print("i2t in 1k test dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % i2t_1k)
#     print("t2i in 1k test dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % t2i_1k)

#     new_img_feat_file = "/data/ftpdir/www/clip/test/test_result/"+TIMESTAMP+"/images_features_1k.npy"
#     new_txt_feat_file = "/data/ftpdir/www/clip/test/test_result/"+TIMESTAMP+"/texts_features_1k.npy"

#     np.save(new_img_feat_file, all_images_feature_1k)
#     np.save(new_txt_feat_file, all_texts_feature_1k)
# endregion

with torch.no_grad():
    model.eval()
    print("start test 5k test dataset")
    all_images_feature_5k = []
    all_texts_feature_5k = []

    for ind, test_batch_5k in enumerate(test_5k_dataloader):
        test_images_5k, test_texts_5k = test_batch_5k
        test_images_5k = test_images_5k.to(device)
        test_texts_5k = test_texts_5k.to(device)

        images_feature = model.encode_image(test_images_5k)
        texts_feature = model.encode_text(test_texts_5k)

        all_images_feature_5k.append(images_feature.cpu().numpy())
        all_texts_feature_5k.append(texts_feature.cpu().numpy())

    print("get 5k test dataset features ")

    all_images_feature_5k = np.concatenate(all_images_feature_5k)
    all_texts_feature_5k = np.concatenate(all_texts_feature_5k)

    all_images_feature_5k = all_images_feature_5k[::5]
    image_to_text_5k = scipy.spatial.distance.cdist(all_images_feature_5k, all_texts_feature_5k, 'cosine')
    # image_to_text_5k = cosine_similarity_numpy(all_images_feature_5k, all_texts_feature_5k)[::5]

    i2t_5k = i2t(image_to_text_5k)
    t2i_5k = t2i(image_to_text_5k)

    print("i2t in 5k test dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % i2t_5k)
    print("t2i in 5k test dataset is: R1 %.2f, R5 %.2f, R10 %.2f, medr %.2f, meanr %.2f \n" % t2i_5k)

    new_img_feat_file = "/data/ftpdir/www/clip/test/test_result/" + TIMESTAMP + "/images_features_5k.npy"
    new_txt_feat_file = "/data/ftpdir/www/clip/test/test_result/" + TIMESTAMP + "/texts_features_5k.npy"
    np.save(new_img_feat_file, all_images_feature_5k)
    np.save(new_txt_feat_file, all_texts_feature_5k)


