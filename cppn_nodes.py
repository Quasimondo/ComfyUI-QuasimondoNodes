from .cppn.models import CPPN
import torch
import numpy as np

BIGMAX = (2**53-1)


def get_coordinates(dim_x, dim_y, scale=1.0, x_offset=0.0, y_offset=0.0, batch_size=1):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = dim_x * dim_y
    x_range = scale * (np.arange(dim_x)+ x_offset - (dim_x - 1) / 2.0) / (dim_x - 1) / 0.5
    y_range = scale * (np.arange(dim_y)+ y_offset - (dim_y - 1) / 2.0) / (dim_y - 1) / 0.5
    x_mat = np.matmul(np.ones((dim_y, 1)), x_range.reshape((1, dim_x)))
    y_mat = np.matmul(y_range.reshape((dim_y, 1)), np.ones((1, dim_x)))
    r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()


class CPPNGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        activations = ["Tanh","Sigmoid","Sin","ReLU","ELU","Softplus","Modulo"]
    
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "width": ("INT", {"default": 1024, "min": 1, "max":8192, "step":1}),
                "height": ("INT", {"default": 1024, "min": 1, "max":8192, "step":1}),
                "framecount": ("INT", {"default": 1, "min": 1, "max":512, "step":1}),
                "dim_z": ("INT", {"default": 32, "min": 1, "max":512, "step":1}),
                "channels_in": ("INT", {"default": 32, "min": 1, "max":512, "step":1}),  
                "layers": ("INT", {"default": 3, "min": 0, "max":16, "step":1}),  
                "first_activation": (activations,),
                "middle_activations": (activations,),
                "last_activation": (activations,{"default":"Sigmoid"}),
                "scale": ("FLOAT", {"default": 1.0, "min": -1024.0, "max":1024.0, "step":0.01}),
                "x_offset": ("FLOAT", {"default": 0.0, "min": -100000000.0, "max":1000000000.0, "step":0.1}),
                "y_offset": ("FLOAT", {"default": 0.0, "min": -1000000000.0, "max":1000000000.0, "step":0.1}),
                "z_offset": ("FLOAT", {"default": 0.0, "min": -1000000000.0, "max":1000000000.0, "step":0.0001}),
                "mean":  ("FLOAT", {"default": 0.0, "min": -1000.0, "max":1000.0, "step":0.0001}),
                "std":  ("FLOAT", {"default": 1.0, "min": 0.0, "max":1000.0, "step":0.0001}),  
                "bias_mean":  ("FLOAT", {"default": 0.0, "min": -1000.0, "max":1000.0, "step":0.0001}),
                "bias_std":  ("FLOAT", {"default": 1.0, "min": 0.0, "max":1000.0, "step":0.0001}), 
                "zero_bias": ("BOOLEAN", {"default": True}),   
                "output_factor": ("FLOAT", {"default": 1.0, "min": -100000000.0, "max":1000000000.0, "step":0.1}),
                "output_offset": ("FLOAT", {"default": 0.0, "min": -1000000000.0, "max":1000000000.0, "step":0.1}),
                "mode": (["Grayscale","RGB"],),
                "normalize": ("BOOLEAN", {"default": False}),   
            },"optional": { 
                "images":("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "exec"

    CATEGORY = "images"

    def exec(self, seed, width, height, framecount, dim_z, channels_in, layers, 
             first_activation, middle_activations, last_activation, scale, x_offset, y_offset, 
             z_offset, mean, std, bias_mean, bias_std, zero_bias, output_factor, output_offset, 
             mode, normalize, images=None ):     
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        dim_c = 1 if mode == "Grayscale" else 3
        
        generator = torch.Generator()
        generator.manual_seed(seed)

        model = CPPN(dim_z, dim_c, channels_in, layers, first_activation, middle_activations, last_activation, mean, std, bias_mean, bias_std, zero_bias, generator ).to(device)

        x, y, r = get_coordinates(width,height, scale, x_offset, y_offset)
        print("x",x.size())
        print("y",y.size())
        print("r",r.size())
        
        x, y, r = x.to(device), y.to(device), r.to(device)

        z = torch.randn(1, dim_z, generator = generator).to(device) + z_offset
        tscale = torch.ones((width * height, 1)).to(device)
        z_scaled = torch.matmul(tscale, z)

        result = model(z_scaled, x, y, r)
        result = result.view(-1, width, height, dim_c).mul(output_factor).add(output_offset).cpu()

        if normalize:
            result -= result.min()
            m = result.max()
            if m!= 0.0:
                result /= m

        if dim_c==1:
            result = result.repeat(1,1,1,3)    


        print("result",result.size())
        return (result,)        
        

