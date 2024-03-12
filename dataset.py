from PIL import Image, ImageDraw
import numpy as np


def superimpose_object(background, foreground, position, blend_edge_probability):
    """
    Superimposes an object onto a background image.
    :param background: Background image.
    :param foreground: Foreground image.
    :param position: A tuple (x, y) indicating where the top left corner of the object should be placed.
    :param blend_edge_probability: Probability of blending the edges of the object into the background.
    """

    bg_width, bg_height = background.size
    obj_width, obj_height = foreground.size
    if obj_width + position[0] > bg_width or obj_height + position[1] > bg_height:
        raise ValueError("Object exceeds background dimensions at the provided position.")

    if np.random.rand() < blend_edge_probability:
        draw = ImageDraw.Draw(foreground)
        edge_width = int(min(obj_width, obj_height) * 0.05)  # Edge width is 5% of the smaller dimension
        for i in range(edge_width):
            alpha = int(255 * (1 - i / edge_width))
            draw.rectangle([i, i, obj_width - i, obj_height - i], outline=(0, 0, 0, alpha))

    object_array = np.array(foreground)
    transparent_indices = np.all(object_array[..., :3] == 0, axis=-1)
    object_array[transparent_indices, 3] = 0
    transparent_indices = np.all(object_array[..., :3] == 255, axis=-1)
    object_array[transparent_indices, 3] = 0
    object_img = Image.fromarray(object_array)

    background.paste(object_img, position, object_img)

    return background


if __name__ == "__main__":
    background = Image.open('./data/backgrounds/ermelinda-martin-Bu_9GlQe8uI-unsplash.jpg').convert('RGBA')
    foreground = Image.open('./data/foregrounds/banana.jpg').convert('RGBA')
    result = superimpose_object(background, foreground, (50, 50), 0.5)
    result.show()
