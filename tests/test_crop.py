from __future__ import annotations

import numpy as np

from ps1_hood.capture.crop import clean_frame, crop_image, detect_chrome_margins, ui_mask


def test_detects_dark_bottom_bar() -> None:
    img = np.full((200, 320, 3), 120, dtype=np.uint8)
    img[185:, :] = 10
    top, right, bottom, left = detect_chrome_margins(img)
    assert bottom >= 10
    assert top < 8
    cropped = crop_image(img)
    assert cropped.shape[0] < 200
    assert cropped.shape[1] <= 320


def test_does_not_eat_the_photo() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(40, 220, size=(240, 320, 3), dtype=np.uint8)
    cropped = crop_image(img, top_frac=0.0, bottom_frac=0.0, auto=True)
    assert cropped.shape[0] >= 180
    assert cropped.shape[1] >= 250


def test_ui_mask_kills_copyright_and_compass() -> None:
    rng = np.random.default_rng(2)
    img = rng.integers(50, 180, size=(200, 320, 3), dtype=np.uint8)
    img[188:, :] = 12
    mask = ui_mask(img)
    assert mask[195, 160] == 255
    assert mask[20, 20] == 255  # search / address corner
    # middle of the photograph stays usable
    assert mask[100, 160] == 0


def test_clean_frame_inpaints_and_returns_mask() -> None:
    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    img[110:, :] = 8
    cleaned, mask = clean_frame(img, auto=True, inpaint=True)
    assert cleaned.shape[0] < 120
    assert mask.shape == cleaned.shape[:2]
    assert int(mask[-1].mean()) == 255
