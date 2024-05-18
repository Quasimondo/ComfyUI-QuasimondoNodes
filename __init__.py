from .gl_nodes import CustomShader, SpringMesh

NODE_CLASS_MAPPINGS = {
     "Custom Shader": CustomShader,
     "Spring Mesh": SpringMesh
}

__all__ = ['NODE_CLASS_MAPPINGS']
