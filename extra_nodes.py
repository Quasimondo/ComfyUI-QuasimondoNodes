import torch

class ImageBlendMaskBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mask": ("IMAGE",),
                "blend_percentage": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_blend_mask"

    CATEGORY = "image"

    def image_blend_mask(self, image_a, image_b, mask, blend_percentage):
        #print("image_a",image_a.size())
        #print("image_b",image_b.size())
        #print("mask",mask.size())
        
        if len(image_a)!=len(image_b):
            raise ValueError(f"Image_Blend_Mask_Batch: image_a and image_b must have same batch size")
        
        if len(mask)!=1 and len(mask)!=len(image_b):
            raise ValueError(f"Image_Blend_Mask_Batch: mask must either be a single one or have same batch size as images")
        
        mask = (mask[:,:,:,0] * 0.299 + mask[:,:,:,1] * 0.587 + mask[:,:,:,2] * 0.114).unsqueeze(-1)
        
        if mask.size(1)!=image_a.size(1) or mask.size(2)!=image_a.size(2):
            mask = torch.nn.functional.interpolate(mask.permute(0,3,1,2),(image_a.size(1),image_a.size(2)),mode='bicubic').permute(0,2,3,1)
            
        
        if len(mask) < len(image_a):
            mask = mask.clone().repeat(len(image_a))
        
        
        #print("mask after",mask.size())
        
        result = image_a * (1.0-(mask*blend_percentage)) + image_b*mask*blend_percentage
        
        return (result, )
