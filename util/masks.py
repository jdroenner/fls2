__author__ = 'Johannes Droenner'

import scipy

# This function generates masks where everything exept the given name is masked, The elements corresponding to the name are NOT masked!
def create_elevation_unmasks(elevation, name_list=["L","S"]):
    list = []

    if "All" in name_list:
        list.append(("A", 0))

    if "L" in name_list:
        list.append(("L", elevation < 0))

    if "S" in name_list:
        list.append(("S", elevation >= 0))

    if "LPC" in name_list:
        list.append(("LPC", buffer_mask_and(elevation < 0)))

    if "SMC" in name_list:
        list.append(("SMC", buffer_mask_or(elevation >= 0)))

    return dict(list)


# this function generates masks for predefined combinations of Land/Sea and Day/Night
def create_light_unmasks(sza, mask_names=["D","N"]):
    named_masks = []

    if "B" in mask_names:
        named_masks.append(("B", sza > 9000))

    if "D" in mask_names:
        named_masks.append(("D", ~(sza < 90.0)))

    if "N" in mask_names:
        named_masks.append(("N", ~(sza >= 90.0)))

    return dict(named_masks)


def create_combined_unmasks(dict_1, dict_2):
    list = dict_1.items()+dict_2.items()

    for key_1 in dict_1.keys():
        mask_1 = dict_1[key_1]

        for key_2 in dict_2.keys():
            mask_2 = dict_2[key_2]

            key = key_1 + key_2
            mask = mask_1 | mask_2

            list.append((key, mask))

    return dict(list)

def buffer_mask_or(mask):
    size_y, size_x = mask.shape
    #out = scipy.empty_like(mask, dtype=bool)
    out = scipy.empty(mask.shape, dtype=bool)

    for x in range(0, size_x-1):
        for y in range(0, size_y-1):
            out[y,x] = mask[y-1,x-1] or mask[y-1,x] or mask[y-1,x+1] or mask[y,x-1] or mask[y,x] or mask[y,x+1] or mask[y+1,x-1] or mask[y+1,x] or mask[y+1,x+1]

    return out

def buffer_mask_and(mask):
    size_y, size_x = mask.shape
    #out = scipy.empty_like(mask, dtype=bool)
    out = scipy.empty(mask.shape, dtype=bool)

    for x in range(0, size_x-1):
        for y in range(0, size_y-1):
            out[y,x] = mask[y-1,x-1] and mask[y-1,x] and mask[y-1,x+1] and mask[y,x-1] and mask[y,x] and mask[y,x+1] and mask[y+1,x-1] and mask[y+1,x] and mask[y+1,x+1]

    return out
