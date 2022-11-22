from retinaface import RetinaFace
import numpy as np
from typing import Iterable


class FaceDetectionError(Exception):
    pass


def find_mid_eye(img: np.ndarray):
    '''Find the mid point between the eyes of the face on the image {img}.

    Args:
        img (np.ndarray): An image representing a single face.
    Returns:
        mid_eye (np.ndarray): The image coordinates for the eye mid point.
    Raises:
        FaceDetectionError: Is risen if none or several faces are detected.
    '''
    faces = RetinaFace.detect_faces(img)
    if len(faces) == 0:
        raise FaceDetectionError("No face detected.")
    if len(faces) >= 2:
        raise FaceDetectionError("Too many faces detected.")
    return np.mean([
        np.array(faces['face_1']['landmarks']['left_eye']),
        np.array(faces['face_1']['landmarks']['right_eye'])],
        axis=0
    )


def align_eyes(imgs: Iterable[np.ndarray], mid_eyes: np.ndarray):
    '''Crop face images so that their eyes align.

    Args:
        imgs (Iterable[np.ndarray]): A collection of face images
        mid_eyes (np.ndarray): The coordinate of the eye midpoint for each image.

    Returns:
        cropped_imgs (np.ndarray): The corresponding cropped images.
    '''
    imshape = np.array(imgs[0].shape[:2][::-1])

    dist_mid_eye = mid_eyes.mean(axis=0)
    shifts = dist_mid_eye[None, :] - mid_eyes

    bound_low = shifts.max(axis=0)
    bound_low_to_high = (imshape + shifts.min(axis=0) - bound_low).astype(int)

    cropped_imgs = []
    for img, shift in zip(imgs, shifts):
        x_crop_low, y_crop_low = (bound_low - shift).astype(int)
        x_crop_high = x_crop_low + bound_low_to_high[0]
        y_crop_high = y_crop_low + bound_low_to_high[1]

        cropped_imgs.append(img[y_crop_low:y_crop_high, x_crop_low:x_crop_high])
    return cropped_imgs


def main(imgs: Iterable[np.ndarray]):
    if len(imgs) <= 1:
        return imgs.copy()

    shape = imgs[0].shape
    for img in imgs:
        if img.shape != shape:
            raise ValueError(f"Incompatible shapes {img.shape} and {shape}")

    mid_eyes = np.array([find_mid_eye(img) for img in imgs])
    return align_eyes(imgs, mid_eyes)
