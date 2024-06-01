from .gl_nodes import CustomShader, SpringMesh
from .queue_nodes import VideoQueueManager, FolderQueueManager
from .random_image_generator import RandomImageGenerator, PerlinNoiseGenerator, ImageNoiseGenerator
from .extra_nodes import ImageBlendMaskBatch

NODE_CLASS_MAPPINGS = {
     "Custom Shader": CustomShader,
     "Spring Mesh": SpringMesh,
     "Video Queue Manager": VideoQueueManager,
     "Folder Queue Manager": FolderQueueManager,
     "Random Image Generator": RandomImageGenerator,
     "Perlin Noise Generator": PerlinNoiseGenerator,   
     "Image Noise Generator": ImageNoiseGenerator,
     "Image Blend by Mask (Batch)": ImageBlendMaskBatch,
}

__all__ = ['NODE_CLASS_MAPPINGS']
