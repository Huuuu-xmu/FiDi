"""Generate tiny illustrative file-format examples, never clinical data."""

from pathlib import Path

from PIL import Image, ImageDraw


OUTPUT = Path(__file__).resolve().parent / "synthetic"
SIZE = (600, 448)


def make_pair(name: str, glottis_label: int, left_label: int, right_label: int, gap: int) -> None:
    image = Image.new("RGB", SIZE, (31, 28, 34))
    mask = Image.new("L", SIZE, 0)
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    left = [(115, 120), (290 - gap, 165), (290 - gap, 280), (115, 330)]
    right = [(485, 120), (310 + gap, 165), (310 + gap, 280), (485, 330)]
    center = [(290 - gap, 167), (310 + gap, 167), (310 + gap, 278), (290 - gap, 278)]

    for polygon, color, label in (
        (left, (172, 110, 120), left_label),
        (right, (176, 117, 126), right_label),
        (center, (15, 13, 19), glottis_label),
    ):
        image_draw.polygon(polygon, fill=color)
        mask_draw.polygon(polygon, fill=label)

    image.save(OUTPUT / f"{name}_rgb.png")
    mask.save(OUTPUT / f"{name}_gt.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    make_pair("synthetic_closed", glottis_label=1, left_label=2, right_label=3, gap=2)
    make_pair("synthetic_open", glottis_label=4, left_label=5, right_label=6, gap=36)


if __name__ == "__main__":
    main()
