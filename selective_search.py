import skimage.io
import skimage.data
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as pts


# img = skimage.data.astronaut()
img = skimage.io.imread('./images/fisheries.jpg')
img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=128 * 128)

print(len(regions))

plt.imshow(img)

for region in regions:
    # excluding regions smaller than 5000 pixels
    if region['size'] < 2000:
        continue
    r = region['rect']
    rectangle = pts.Rectangle((r[0], r[1]), r[2], r[3], fill=None, edgecolor='red')
    plt.gca().add_patch(rectangle)

plt.show()
