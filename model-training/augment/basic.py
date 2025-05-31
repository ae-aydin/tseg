import albumentations as A
import cv2


class BasicAugment:
    def __init__(self, img_size: int = 512):
        self.img_size = img_size
        self.transform = A.Compose(
            [
                A.Resize(self.img_size, self.img_size, interpolation=cv2.INTER_AREA),
                A.Normalize(),
                A.ToTensorV2(),
            ]
        )

    def __call__(self, image, mask):
        transformed = self.transform(image=image, mask=mask)
        return {"image": transformed["image"], "mask": transformed["mask"]}

    def get_transform(self):
        return self.transform
