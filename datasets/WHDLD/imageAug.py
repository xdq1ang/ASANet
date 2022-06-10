import Augmentor

# class Augmentor2(Augmentor):
#     def Pip




p = Augmentor.Pipeline(r"WHDLD\water_image_png")
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
p.ground_truth(r"WHDLD\water_label")
# Add operations to the pipeline as normal:
p.rotate(probability=1, max_left_rotation=25, max_right_rotation=25)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.9)
p.flip_top_bottom(probability=0.5 )
p.skew_tilt(probability=0.5,magnitude=0.1)
p.skew_corner(probability=0.5,magnitude=0.1)
p.sample(5000)

