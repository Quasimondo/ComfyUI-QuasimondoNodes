from .gl_nodes import CustomShader, SpringMesh
from .queue_nodes import VideoQueueManager, FolderQueueManager

NODE_CLASS_MAPPINGS = {
     "Custom Shader": CustomShader,
     "Spring Mesh": SpringMesh,
     "Video Queue Manager": VideoQueueManager,
     "Folder Queue Manager": FolderQueueManager,
}

__all__ = ['NODE_CLASS_MAPPINGS']
