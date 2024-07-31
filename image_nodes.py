import torch
import numpy as np
import time
from PIL import Image
import cv2
import hashlib
import folder_paths
import os
from torchvision.transforms import ToPILImage
import json


BIGMIN = -(2**53 - 1)
BIGMAX = 2**53 - 1

def pil2tensor(image,mask=False):
    return torch.from_numpy(np.array(image).astype(np.float32) / (1.0 if mask else 255.0)).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(image.mul(255.0).clamp(0,255).cpu().numpy().astype(np.uint8))

class ImageToOpticalFlow:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": (["R","G","B"],{"default":"R"}),
                "x_scale": (
                    "FLOAT",
                    {"default": 256.0, "min": -10000.0, "max": 10000.0, "step": 0.01},
                ),
                "x_offset": (
                    "FLOAT",
                    {"default": -128.0, "min": -10000.0, "max": 10000.0, "step": 0.01},
                ),
                "y": (["R","G","B"],{"default":"G"}),
                "y_scale": (
                    "FLOAT",
                    {"default": 256.0, "min": -10000.0, "max": 10000.0, "step": 0.01},
                ),
                "y_offset": (
                    "FLOAT",
                    {"default": -128.0, "min": -10000.0, "max": 10000.0, "step": 0.01},
                ),
               
            },
        }

    RETURN_TYPES = ("OPTICAL_FLOW",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "exec"

    CATEGORY = "image"

    def exec( self, image, x, x_scale, x_offset,y,y_scale,y_offset ):
        lut = {"R":0,"G":1,"B":2}
        indices = torch.tensor([lut[x],lut[y]],dtype=int,device=image.device)
        flow = image.clone()[:,:,:,indices]
        flow[:,:,:,0] *= x_scale
        flow[:,:,:,0] += x_offset          
        flow[:,:,:,1] *= y_scale
        flow[:,:,:,1] += y_offset          
        return (flow.cpu().numpy(), )
    


class ShiftMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "x_offset": (
                    "INT",
                    {"default": 0, "min": -10000, "max": 10000, "step": 1},
                ),
                "y_offset": (
                    "INT",
                    {"default": 0, "min": -10000, "max": 10000, "step": 1},
                ),
               
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "exec"

    CATEGORY = "image"

    def exec( self, mask, x_offset,y_offset ):
        if len(mask.size())==3:
            mask = torch.roll(mask.clone(), shifts=( y_offset, x_offset), dims=(1, 2))
        else:
            mask = torch.roll(mask.clone(), shifts=( y_offset, x_offset), dims=(0, 1))
        return(mask,)
        
class SlitScan:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "mask_time_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 256.0, "step": 0.01},
                ),
                "t_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01},
                ),
                "time_depth": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 256.0, "step": 0.01},
                ),
                "blend_power": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 32.0, "step": 0.1},
                ),
                "wrap_t": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "exec"

    CATEGORY = "image"

    def exec(
        self, images, masks, mask_time_factor, t_offset, time_depth, blend_power, wrap_t
    ):
        if type(images) == list:
            images = torch.cat(images, dim=0)
        if type(masks) == list:
            masks = torch.cat(masks, dim=0)

        if masks.size(1) != images.size(1) or masks.size(2) != images.size(2):
            masks = torch.nn.functional.interpolate(
                masks.permute(0, 3, 1, 2),
                (images.size(1), images.size(2)),
                mode="bicubic",
            ).permute(0, 2, 3, 1)

        masks = (masks.clone() * time_depth) + t_offset
        if mask_time_factor > 0:
            for i in range(len(masks)):
                f = i / (len(masks) - 1) * mask_time_factor
                masks[i] += f

        if wrap_t:
            masks = (((masks % 1.0) + 1.0) % 1.0) * (images.size(0) - 1)
        else:
            masks = masks.clamp(0.0, 1.0) * (images.size(0) - 1)

        lower = masks.floor()
        blend = (masks - lower).unsqueeze(-1)

        if blend_power > 1.0:
            i1 = blend < 0.5
            i2 = blend >= 0.5
            blend[i1] = 0.5 * (2.0 * blend[i1]).pow(blend_power)
            blend[i2] = 1.0 - 0.5 * (2.0 * (1.0 - blend[i2])).pow(blend_power)

        lower = lower.long().unsqueeze(-1)

        result = []
        for i in range(len(masks)):
            for j in range(images.size(0) - 1):
                if j == 0:
                    merged = images[0].clone()
                indices = (lower[i] == j).repeat(1, 1, 3)
                m = images[j] * (1.0 - blend[i]) + images[j + 1] * blend[i]
                merged[indices] = m[indices]
            result.append(merged.unsqueeze(0))
        result = torch.cat(result, dim=0)
        return (result,)
        
      
class DistanceMap:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "xy_mode": (["max","min","median"],)
            },
        }

    RETURN_TYPES = ("MASK","INT","INT")
    RETURN_NAMES = ("image","x","y")
    FUNCTION = "exec"

    CATEGORY = "generator"

    def exec( self, mask, xy_mode ):

        image = mask[0].mul(255.0).clamp(0,255).cpu().numpy().astype(np.uint8)

        ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) 
          
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) 
        dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX) 
        if xy_mode == "max":
            index = np.argmax(dist_output)
        elif xy_mode == "min":
            index = np.argmin(dist_output)
        elif xy_mode == "median":
            index = np.argsort(dist_output.flatten())[len(dist_output.flatten())//2]
            print("index",index)
        height, width = dist_output.shape
        y, x = np.unravel_index(index, (height, width))
        dist_output = pil2tensor(dist_output,True)
        return(dist_output,x,y )           
        
        
class TemporalBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frames_backward": ("INT",{"default": 8, "min": 0, "max": 16384, "step": 1},),
                "frames_forward": ("INT",{"default": 0, "min": 0, "max": 16384, "step": 1},),
                "falloff": ("FLOAT",{"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "exec"

    CATEGORY = "generator"

    def exec( self, images, frames_backward, frames_forward, falloff ):

        result = images.clone()
        for i in range(len(images)):
            indices = [i]
            factors = [1.0]
            factor = 1.0
            for j in range(i-1,i-frames_backward,-1):
                factor *= falloff
                if factor>0.0 and j > -1:
                    indices.append(j)
                    factors.append(factor)
            factor = 1.0
            for j in range(i+1,i+frames_forward):
                factor *= falloff
                if factor>0.0 and j < len(images):
                    indices.append(j)
                    factors.append(factor)
            indices = torch.tensor(indices,dtype=int,device=images.device)
            factors = torch.tensor(factors,dtype=images.dtype,device=images.device).view(-1,1,1,1)
            result[i] = (images[indices]*factors).sum(axis=0) / factors.sum()     
            
        return(result, )           

        
        
class PreviewMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
            },
            
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "exec"
    CATEGORY = "mask"

    def exec(self, masks):
        
        if len(masks.size())==2:
            masks = masks.unsqueeze(0)
        
        results = []
            
        for i in range(len(masks)):            
            preview = ToPILImage()(masks[i])
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    "", folder_paths.get_temp_directory(), preview.height, preview.width
                )
            )
            file = "qtmp_" + str(int(time.time() * 1000)) + ".jpg"
            preview.save(os.path.join(full_output_folder, file))
            results.append({"filename": file, "subfolder": subfolder, "type": "temp"})

        return {"ui": {"images": results}}         
        

class CoordinatesFromMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "percentage": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_distance_from_edge": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "max_points": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1}),
                "seed": ("INT", {"default": 1234, "min": BIGMIN, "max": BIGMAX, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "MASK")
    RETURN_NAMES = ("coordinates", "point_mask")
    FUNCTION = "exec"

    CATEGORY = "generator"

    def exec(self, mask, percentage, min_distance_from_edge, max_points, seed):
        # Convert mask to numpy array
        image = mask[0].mul(255.0).clamp(0,255).cpu().numpy().astype(np.uint8)

        # Threshold the image
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # Compute distance transform
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Filter points based on min_distance_from_edge
        valid_points = np.argwhere(dist >= min_distance_from_edge)

        # Determine number of points to return
        num_points = min(int(len(valid_points) * percentage), max_points)

        # Sample points
        selected_points = self.sample_points(valid_points, num_points, dist.shape, abs(seed))

        # Convert to list of dictionaries
        coordinates = [{"x": int(point[1]), "y": int(point[0])} for point in selected_points]

        # Convert to JSON string
        json_coordinates = json.dumps(coordinates)

        # Create point mask
        point_mask = np.zeros_like(image)
        for point in selected_points:
            point_mask[point[0], point[1]] = 255

        # Convert point_mask to tensor
        point_mask_tensor = torch.from_numpy(point_mask).float() / 255.0
        point_mask_tensor = point_mask_tensor.unsqueeze(0)  # Add batch dimension

        return (json_coordinates, point_mask_tensor)

    def sample_points(self, points, k, shape, seed ):
        if len(points) <= k:
            return points

        result = []
        grid_size = max(1, int(np.sqrt(shape[0] * shape[1] / k) / 2))
        grid = {}

        def get_cell(point):
            return int(point[0] / grid_size), int(point[1] / grid_size)

        def get_neighbors(cell):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    yield (cell[0] + i, cell[1] + j)

        def is_valid(point):
            cell = get_cell(point)
            for neighbor in get_neighbors(cell):
                if neighbor in grid:
                    if np.linalg.norm(point - grid[neighbor]) < grid_size:
                        return False
            return True

        # Shuffle points for randomness
        rng = np.random.default_rng(seed)
        rng.shuffle(points)

        # First pass: Poisson disk sampling
        for point in points:
            if len(result) >= k:
                break
            if is_valid(point):
                result.append(point)
                grid[get_cell(point)] = point

        # Second pass: Fill remaining points if necessary
        if len(result) < k:
            remaining = k - len(result)
            # Create a boolean mask for points not in result
            mask = np.ones(len(points), dtype=bool)
            for point in result:
                mask[np.where((points == point).all(axis=1))[0][0]] = False
            # Select from remaining points
            additional_indices = rng.choice(np.where(mask)[0], size=remaining, replace=False)
            additional_points = points[additional_indices]
            result.extend(additional_points)

        return np.array(result)
