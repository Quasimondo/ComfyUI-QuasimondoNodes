import numpy as np
import torch
from .colorTransfer import hist_match_rgb
BIGMAX = (2**53-1)


def normalize_image(image):
    for c in range(image.size(-1)):
        image[:,:,c] -= image[:,:,c].min()
        image[:,:,c] /= image[:,:,c].max()
    
def generate_images(mode:str, imgtype:str="RGB", image_count: int = 1, width: int = 512, height: int = 512, seed:int=0, normalize:bool = True):
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    channels = 3
    if imgtype=="RGBA":
        channels = 4
    elif imgtype=="Mask":
        channels = 1

    if mode == "Noise":
        images = torch.rand((image_count,height,width,channels), generator=generator, device="cpu", dtype=torch.float32)
    elif mode == "Gaussian Noise":
        images = torch.randn((image_count,height,width,channels), generator=generator, device="cpu", dtype=torch.float32)
   
    if normalize:
        for i in range(image_count):
            normalize_image(images[i])

    if channels==1:
        images = images.squeeze(-1)
                
    return (images, image_count, width, height)
    
def generate_perlin(imgtype:str="RGB", image_count: int = 1, width: int = 512, height: int = 512, seed:int=0, octaves:int=3, res:int=8, persistence:float = 0.5, phase:float = 0.0, normalize:bool = True):
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    channels = 3
    if imgtype=="RGBA":
        channels = 4
    elif imgtype=="Mask" or imgtype=="RGB (grey)":
        channels = 1

    images = torch.zeros((image_count,height,width,channels),device="cpu", dtype=torch.float32)
    padded_width = int(np.ceil(width / res) * res)
    padded_height = int(np.ceil(height / res) * res)     
    print(f"{padded_width=} {padded_height=}" )
    octaves = int(min(np.log2(min(padded_width,padded_height))-2,octaves))
    
    for i in range(image_count):
        for c in range(channels):
            images[i,:,:,c] = rand_perlin_2d_octaves((padded_height,padded_width),(res,res),generator,octaves,persistence, phase)[:height,:width]
            
    if normalize:
        for i in range(image_count):
            normalize_image(images[i])

    if imgtype=="RGB (grey)":
        images = torch.cat([images,images,images],dim=-1)    

    if channels==1:
        images = images.squeeze(-1)
                
    return (images, image_count, width, height)    
    
#from: https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57    
def rand_perlin_2d(shape, res, generator, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3, phase=0.0):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (int(np.ceil(shape[0] / res[0])), int(np.ceil(shape[1] / res[1])))
    
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
    angles = 2*np.pi*(torch.rand(res[0]+1, res[1]+1, generator=generator)+phase)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
    
    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
    
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return np.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(shape, res, generator, octaves=1, persistence=0.5, phase=0.0):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        if amplitude==0.0:
            break
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]), generator, phase=phase)
        frequency *= 2
        amplitude *= persistence
    return noise    

class RandomImageGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Noise", "Gaussian Noise"],),
                "image_count": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "channels": (["RGB", "RGBA", "Mask"],),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "normalize" : ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT","INT","INT")
    RETURN_NAMES = ("IMAGE","image_count","width","height")
    
    FUNCTION = "generate_images"
    CATEGORY = "image/generators"

    def generate_images(self,mode,image_count,width, height,channels, random_seed, normalize):
        return generate_images(mode, channels, image_count,width,height,random_seed, normalize)
        
        
class ImageNoiseGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "black_mix": ("INT", {"default": 0, "max": 20, "min": 0, "step": 1}),
                "brightness": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.0, "step": 0.01}),
                "same_seed_for_batch":("BOOLEAN",{"default":False})
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    
    FUNCTION = "generate_images"
    CATEGORY = "image/generators"

    def generate_images(self,image, seed, black_mix, brightness, same_seed_for_batch ):
        np.random.seed(seed&0xFFFFFFFF)
        
        osize = image.size()
        result = image.clone().reshape(image.size(0),-1,image.size(-1))
        for i in range(len(image)):
            if same_seed_for_batch:
                np.random.seed(seed&0xFFFFFFFF)
            result[i] = result[i,np.random.permutation(result.size(1))]
            for j in range(black_mix):
                indices = np.random.choice(np.arange(result.size(1)),int(result.size(1)/2),replace=False)
                result[i][indices] = 0.0
                
        result = result.reshape(osize) * brightness  
     
        
        
        return (result,)  
        
class PerlinNoiseGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_count": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "channels": (["RGB", "RGBA", "RGB (grey)","Mask"],),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "octaves": ("INT", {"default": 3, "min": 1, "max": 13, "step": 1}),
                "res": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),                
                "persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.001}),
                "phase": ("FLOAT", {"default": 0., "min": 0., "max": 1.0, "step": 0.001}),
                "normalize" : ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT","INT","INT")
    RETURN_NAMES = ("IMAGE","image_count","width","height")
    
    FUNCTION = "generate_perlin"
    CATEGORY = "image/generators"

    def generate_perlin(self,image_count,width, height, channels, random_seed, octaves, res, persistence, phase, normalize):
        return generate_perlin(channels, image_count,width,height,random_seed, octaves, res, persistence, phase, normalize)        
    
NODE_CLASS_MAPPINGS = {
    "Random Image Generator": RandomImageGenerator,
    "Perlin Noise Generator": PerlinNoiseGenerator
}
