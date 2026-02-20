import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


# ==========================================================
# ===================== TRAIN LOADER ======================
# ==========================================================

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):

        self.rgb_dir = rgb_dir
        self.ps = img_options['patch_size']

        self.samples = []

        scene_folders = sorted(os.listdir(rgb_dir))

        for scene in scene_folders:
            scene_path = os.path.join(rgb_dir, scene)

            if not os.path.isdir(scene_path):
                continue

            files = os.listdir(scene_path)

            clean_imgs = sorted([f for f in files if "-C-" in f])
            rain_imgs  = sorted([f for f in files if "-R-" in f])

            for c_img in clean_imgs:
                prefix = c_img.split("-C-")[0]

                matches = [r for r in rain_imgs if r.startswith(prefix)]

                for r_img in matches:
                    self.samples.append((
                        os.path.join(scene_path, r_img),
                        os.path.join(scene_path, c_img)
                    ))

        if len(self.samples) == 0:
            raise RuntimeError(f"\n❌ No GT-RAIN training pairs found in {rgb_dir}\n")

        print(f"✅ GT-RAIN Training pairs loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        rain_path, clean_path = self.samples[index]

        rain_img  = TF.to_tensor(Image.open(rain_path).convert('RGB'))
        clean_img = TF.to_tensor(Image.open(clean_path).convert('RGB'))

        ps = self.ps
        _, h, w = rain_img.shape

        if ps is not None:

            # Pad if image smaller than patch
            if h < ps or w < ps:
                pad_h = max(ps - h, 0)
                pad_w = max(ps - w, 0)

                rain_img  = TF.pad(rain_img, (0, 0, pad_w, pad_h), padding_mode='reflect')
                clean_img = TF.pad(clean_img, (0, 0, pad_w, pad_h), padding_mode='reflect')

                _, h, w = rain_img.shape

            # Random crop
            rr = random.randint(0, h - ps)
            cc = random.randint(0, w - ps)

            rain_img  = rain_img[:, rr:rr+ps, cc:cc+ps]
            clean_img = clean_img[:, rr:rr+ps, cc:cc+ps]

        return clean_img, rain_img


# ==========================================================
# ===================== VALIDATION ========================
# ==========================================================

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):

        self.rgb_dir = rgb_dir
        self.ps = img_options['patch_size']

        self.samples = []

        scene_folders = sorted(os.listdir(rgb_dir))

        for scene in scene_folders:
            scene_path = os.path.join(rgb_dir, scene)

            if not os.path.isdir(scene_path):
                continue

            files = os.listdir(scene_path)

            clean_imgs = sorted([f for f in files if "-C-" in f])
            rain_imgs  = sorted([f for f in files if "-R-" in f])

            for c_img in clean_imgs:
                prefix = c_img.split("-C-")[0]

                matches = [r for r in rain_imgs if r.startswith(prefix)]

                for r_img in matches:
                    self.samples.append((
                        os.path.join(scene_path, r_img),
                        os.path.join(scene_path, c_img)
                    ))

        if len(self.samples) == 0:
            raise RuntimeError(f"\n❌ No GT-RAIN validation pairs found in {rgb_dir}\n")

        print(f"✅ GT-RAIN Validation pairs loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        rain_path, clean_path = self.samples[index]

        rain_img  = TF.to_tensor(Image.open(rain_path).convert('RGB'))
        clean_img = TF.to_tensor(Image.open(clean_path).convert('RGB'))

        ps = self.ps

        if ps is not None:
            rain_img  = TF.center_crop(rain_img, (ps, ps))
            clean_img = TF.center_crop(clean_img, (ps, ps))

        filename = os.path.splitext(os.path.basename(clean_path))[0]

        return clean_img, rain_img, filename


# ==========================================================
# ===================== TEST ONLY =========================
# ==========================================================

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):

        self.inp_filenames = sorted([
            os.path.join(inp_dir, x)
            for x in os.listdir(inp_dir)
            if is_image_file(x)
        ])

        if len(self.inp_filenames) == 0:
            raise RuntimeError(f"\n❌ No test images found in {inp_dir}\n")

        print(f"✅ Test samples loaded: {len(self.inp_filenames)}")

    def __len__(self):
        return len(self.inp_filenames)

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.basename(path_inp))[0]

        inp = TF.to_tensor(Image.open(path_inp).convert('RGB'))

        return inp, filename
