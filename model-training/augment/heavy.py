import albumentations as A
import cv2


class HeavyAugment:
    def __init__(self, img_size: int = 512):
        self.img_size = img_size
        self.transform = A.Compose(
            [
                A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_AREA),
                # Geometric transforms - more aggressive
                A.RandomRotate90(p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.Affine(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.7),
                # A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                # A.OpticalDistortion(distort_limit=0.3, p=0.5),
                # Color and staining variations - more aggressive
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                        A.RGBShift(
                            r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0
                        ),
                        A.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                        ),
                    ],
                    p=0.8,
                ),
                # Quality variations - more aggressive
                A.OneOf(
                    [
                        A.GaussNoise(p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, 7), p=1.0),
                        A.MedianBlur(blur_limit=5, p=1.0),
                        A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=1.0),
                    ],
                    p=0.5,
                ),
                # Tissue artifacts and variations
                A.OneOf(
                    [
                        A.CoarseDropout(
                            num_holes_range=(4, 8),
                            hole_height_range=(16, 32),
                            hole_width_range=(16, 32),
                            p=1.0,
                        ),
                        A.GridDropout(
                            ratio=0.3,
                            unit_size_range=(32, 64),
                            holes_number_xy=(4, 4),
                            shift_xy=(0, 0),
                            random_offset=True,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                # Additional histopathology-specific augmentations
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                    ],
                    p=0.3,
                ),
                # Normalize and convert to tensor
                A.Normalize(),
                A.ToTensorV2(),
            ],
            p=1.0,
        )

    def __call__(self, image, mask):
        transformed = self.transform(image=image, mask=mask)
        return {"image": transformed["image"], "mask": transformed["mask"]}

    def get_transform(self):
        return self.transform
