"""
"""
import kwplot
import kwimage
# import numpy as np
plt = kwplot.autoplt()
kwplot.autompl()


# lhs1 = kwimage.draw_text_on_image(None, 'k', color='kitware_blue')
lhs = kwimage.draw_text_on_image(None, 'nd', color='kitware_red', fontScale=10, thickness=30)
rhs = kwimage.draw_text_on_image(None, 'sa', color='kitware_orange', fontScale=10, thickness=30)
top = kwimage.stack_images([lhs, rhs], axis=1)
bot = kwimage.draw_text_on_image(None, 'mpler', color='kitware_orange', fontScale=10, thickness=30)
source_img = kwimage.stack_images([top, bot], axis=1)

phi = 1.618033988
w = h = 1 - (1 / phi)
x = 0.1
y = 0.02

w /= 2
h /= 2.5

box01 = kwimage.Box.coerce([x, y, w, h], format='xywh')
box1 = box01.scale(max(source_img.shape[0:2][::-1])).quantize()
sl = box1.to_slice()
chip_img = source_img[sl]


sf = .25
small_box = box1.scale(sf)
# chip_img = kwimage.imresize(chip_img, scale=3)

box2 = box1.translate((-box1.tl_x, -box1.tl_y))

bg_canvas = kwimage.imresize(source_img, scale=sf)
bg_canvas = small_box.draw_on(bg_canvas, color='kitware_green', thickness=2)

# chip_img = box1.to_relative_mask().to_multi_polygon()[0]
zoomed_canvas = box2.draw_on(chip_img, color='kitware_green', thickness=2)

canvas, transforms = kwimage.stack_images([bg_canvas, zoomed_canvas], return_info=True, pad=10)

pts1 = small_box.corners()
pts2 = box2.corners()

pts1 = kwimage.warp_points(transforms[0].params, pts1)
pts2 = kwimage.warp_points(transforms[1].params, pts2)

canvas = kwimage.draw_line_segments_on_image(canvas, pts1, pts2, color='kitware_green')

# poly1 = kwimage.Mask.coerce((lhs.sum(axis=2) > 0).astype(np.uint8)).to_multi_polygon()
# poly2 = kwimage.Mask.coerce((rhs.sum(axis=2) > 0).astype(np.uint8)).to_multi_polygon()
# poly1 = poly1.simplify(1)
# poly2 = poly2.simplify(1)

# poly2 = poly2.translate((0, poly1.to_box().br_y))
# box1 = poly1.to_box().to_polygon()
# box2 = poly2.to_box().to_polygon()

# box1.union(box2).to_box().scale(1.1, about='center').draw(fill=False, facecolor=None, setlim=1)

# poly1.draw(color='kitware_blue')
# poly2.draw(color='kitware_green')

# ax = plt.gca()
# ax.invert_yaxis()
# ax.set_aspect('equal')

# fig = plt.gcf()
# img = kwplot.render_figure_to_image(fig)
kwimage.imwrite('ndsampler_logo.png', canvas)
kwplot.imshow(canvas)
