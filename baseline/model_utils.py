import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

def increase_brightness(img, alpha=0.2):
    height, width, _ = img.shape
    white_img = np.zeros([height,width,3],dtype=np.uint8)
    white_img.fill(255) # or img[:] = 255

    dst = cv2.addWeighted(img, alpha , white_img, 1-alpha, 0)
    return dst

def increase_brightness_except(img, bbox_ls, alpha=0.2):
    height, width, _ = img.shape
    white_img = np.zeros([height,width,3],dtype=np.uint8)
    white_img.fill(255) # or img[:] = 255

    output_img = cv2.addWeighted(img, alpha , white_img, 1-alpha, 0)

    for x1, y1, x2, y2 in bbox_ls:
        output_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return output_img

def extract_single_object(img, mask, alpha=0.8):
    # Ensure mask is binary (0 or 1)
    mask = mask.astype(bool)
    mask = np.logical_not(mask)
    
    # Create a white image of the same size as the input image
    height, width, _ = img.shape
    white_img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Apply mask to the white image
    masked_white_img = np.where(mask, white_img, img)
    
    # Blend the original image with the masked white image
    output_img = cv2.addWeighted(img, 1-alpha, masked_white_img, alpha, 0)
    
    # output_path = "/home/nvelingker/LASER/VidVRD-II/debug/test.png"  # or test.jpg, etc.

    # # 'output_img' is the image you want to save
    # cv2.imwrite(output_path, output_img)

    return output_img

def crop_image_contain_bboxes(img, bbox_ls, data_id):
    """
    Crops `img` so that it contains all bounding boxes in `bbox_ls`.
    
    Args:
        img (np.ndarray): Image array, shape (H, W, C) or (H, W).
        bbox_ls (list): List of bounding boxes in normalized coords [0..1].
            Each bbox can be:
              - A dict with keys ['x1','y1','x2','y2'], or
              - A list/tuple [x1, y1, x2, y2].
            Here, x1 < x2, y1 < y2, and 0 <= x1, x2, y1, y2 <= 1.
        data_id (str): An identifier for error/debug messages.

    Returns:
        np.ndarray: The cropped image that contains all bounding boxes.
    """
    import numpy as np

    H, W = img.shape[:2]  # Get image height and width

    all_bx1, all_bx2 = [], []
    all_by1, all_by2 = [], []

    for bbox in bbox_ls:
        # 1. Retrieve [x1, y1, x2, y2]
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        else:
            x1, y1, x2, y2 = bbox  # e.g. [x1, y1, x2, y2]

        # 2. Scale normalized floats by image width/height
        #    Then round or int() to get pixel coordinates.
        x1 = int(round(x1 * W))
        x2 = int(round(x2 * W))
        y1 = int(round(y1 * H))
        y2 = int(round(y2 * H))

        # 3. Sort so x1 < x2 and y1 < y2
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # 4. Clamp each coordinate to stay within image boundaries
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        all_bx1.append(x1)
        all_bx2.append(x2)
        all_by1.append(y1)
        all_by2.append(y2)

    # 5. Compute the "union" bounding box containing all bboxes
    union_x1 = min(all_bx1)
    union_x2 = max(all_bx2)
    union_y1 = min(all_by1)
    union_y2 = max(all_by2)

    # 6. Sanity-check that we have a valid box
    if union_x2 <= union_x1 or union_y2 <= union_y1:
        raise ValueError(
            f"Image bbox issue for data_id={data_id}: "
            f"union_x1={union_x1}, union_x2={union_x2}, "
            f"union_y1={union_y1}, union_y2={union_y2}"
        )

    # 7. Finally, crop and return
    ret =  img[union_y1:union_y2, union_x1:union_x2]
    
    output_path = "/home/nvelingker/LASER/VidVRD-II/debug/test2.png"  # or test.jpg, etc.

    # 'output_img' is the image you want to save
    cv2.imwrite(output_path, ret)
    
    return ret


        
def extract_object_subject(img, red_mask, blue_mask, alpha=0.5, white_alpha=0.8):
    # Ensure the masks are binary (0 or 1)
    red_mask = red_mask.astype(bool)
    blue_mask = blue_mask.astype(bool)
    non_masked_area = ~(red_mask | blue_mask)
    
    # Split the image into its color channels (B, G, R)
    b, g, r = cv2.split(img)
    
    # Adjust the red channel based on the red mask
    r = np.where(red_mask[:, :, 0], np.clip(r + (255 - r) * alpha, 0, 255), r).astype(np.uint8)

    # Adjust the blue channel based on the blue mask
    b = np.where(blue_mask[:, :, 0], np.clip(b + (255 - b) * alpha, 0, 255), b).astype(np.uint8)

    # Merge the channels back together
    output_img = cv2.merge((b, g, r))
    
    white_img = np.full_like(output_img, 255, dtype=np.uint8)
    output_img = np.where(non_masked_area, cv2.addWeighted(output_img, 1 - white_alpha, white_img, white_alpha, 0), output_img)

    return output_img

def increase_brightness_draw_outer_edge(img, bbox_ls, alpha=0.2, colormap_name='Set1', thickness=2):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().astype(np.uint8)
    else: 
        img = img.astype(np.uint8)
    height, width, _ = img.shape
    white_img = np.zeros([height,width,3],dtype=np.uint8)
    white_img.fill(255) # or img[:] = 255

    output_img = cv2.addWeighted(img, alpha , white_img, 1-alpha, 0)
    colormap = plt.colormaps[colormap_name]

    for bbox_id, (x1, y1, x2, y2) in enumerate(bbox_ls):
        output_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        color =  [c * 255 for c in mpl.colors.to_rgb(colormap(bbox_id))]
        # print(f"color: {color}")
        output_img = cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)

    return torch.tensor(output_img, dtype=torch.float32)

