"""Module with custom image transforms implementations."""
import cv2


class Resize:
    """Custom resize transformation."""

    def __init__(self, size: tuple[int, int] = (320, 64)):
        """Resize transformation insance.

        Args:
            - size: output image size
        """
        self.size = size

    def __call__(self, item) -> dict:
        """Apply resizing.

        Args:
            - item: Dict with keys "image", "seq", "seq_len", "text".

        Returns:
            Dict with image resized to self.size.
        """
        interpolation = (
            cv2.INTER_AREA if self.size[0] < item["image"].shape[1] else cv2.INTER_LINEAR
        )
        item["image"] = cv2.resize(item["image"], self.size, interpolation=interpolation)

        return item
