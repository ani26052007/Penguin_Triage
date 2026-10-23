"""Image handling between the simulator camera and the policy (fixture repo for the report tests)."""

CAMERA_EXTRINSICS = {"front": [0.5, 0.0, 0.4], "wrist": [0.0, 0.0, 0.1]}
TARGET_SIZE = (224, 224)


def normalize_image(img):
    """Scale 0..255 to 0..1."""
    return img / 255.0


def flip_image(img):
    """Flip an image top to bottom."""
    return img[::-1]


class FrameBuffer:
    def push(self, frame):
        self.last = frame

    def latest(self):
        return self.last
