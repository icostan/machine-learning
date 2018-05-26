import json
import functools

def for_image(filename, fish):
    for a in for_fish(fish):
        if a['filename'] == filename:
            if len(a['annotations']) > 0:
                return a['annotations'][0]
            else:
                return None

@functools.lru_cache(maxsize=None)
def for_fish(fish='LAG'):
    fish = fish.lower()
    path = 'annotations/{fish}_labels.json'.format(fish=fish)
    with open(path) as data_file:
        annotations = json.load(data_file)
    return annotations
