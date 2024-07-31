from .gl_nodes import (
     CustomShader, 
     SpringMesh
)

from .queue_nodes import (
     VideoQueueManager, 
     FolderQueueManager
)

from .random_image_generator import (
     RandomImageGenerator, 
     PerlinNoiseGenerator, 
     ImageNoiseGenerator
)

from .extra_nodes import (
     ImageBlendMaskBatch, 
     ColorMatch
)

from .cppn_nodes import CPPNGenerator

from .image_nodes import (
    SlitScan,
    ImageToOpticalFlow,
    DistanceMap,
    ShiftMask,
    PreviewMask,
    TemporalBlur,
    CoordinatesFromMask
)

NODE_CLASS_MAPPINGS = {
     "Custom Shader": CustomShader,
     "Spring Mesh": SpringMesh,
     "Video Queue Manager": VideoQueueManager,
     "Folder Queue Manager": FolderQueueManager,
     "Random Image Generator": RandomImageGenerator,
     "Perlin Noise Generator": PerlinNoiseGenerator,   
     "Image Noise Generator": ImageNoiseGenerator,
     "Image Blend by Mask (Batch)": ImageBlendMaskBatch,
     "Color Match": ColorMatch,
     "CPPN Generator": CPPNGenerator,
     "Slit Scan": SlitScan,
     "Image to Optical Flow": ImageToOpticalFlow,
     "Distance Map":DistanceMap,
     "Shift Mask":ShiftMask,
     "Preview Mask": PreviewMask,
     "Temporal Blur":TemporalBlur,
     "Coordinates From Mask":CoordinatesFromMask,
}

__all__ = ['NODE_CLASS_MAPPINGS']
