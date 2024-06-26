# ComfyUI-QuasimondoNodes
A collection of various custom nodes for ComfyUI (Work in progress)

## What is it?

Nodes I wrote mostly for myself since I find it often quicker to write my own solution than trying to find an existing one. I am pretty sure that I will be reinventing the wheel here quite often.
To be used with [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) as custom nodes.

### Updates
* June 28th 2024: Added option to use many Shadertoy (https://www.shadertoy.com/) shaders without conversion
* June 21st 2024: Added `CPPN Generator`
* June 1st 2024: Added `Video Queue Manager` and `Folder Queue Manager`

### Nodes

* `Custom Shader`: This allows you to integrate your own OpenGL shaders. Even allows to generate animations. (uses the moderngl library)
* `Spring Mesh`: This node is still very much work in progress - the idea is to distort a mesh using motion maps or flows  (uses the moderngl library)
* `Video Queue Manager`: This node allows to process an entire folder containing videos one-by-one and frame-by-frame when used in conjunction with auto queue.
* `Folder Queue Manager`: This node allows to process an entire folder sub-folders one-by-one in conjunction with auto queue.
* `Image Blend Mask (Batch)`: This is just a simple replacement for Image Blend Mask that also works with batches
* `CPPN Generator`: A node based on Hardmaru's Compositional Pattern-Producing Networks (CPPN)

## Install

1. Enter ComfyUI's Python Environment by running `.\.venv\Scripts\activate` from ComfyUI's root directory.
2. Clone this repo into ComfyUI's `custom_nodes` directory by entering the directory and running: `git clone git@github.com:Quasimondo/ComfyUI-QuasimondoNodes.git ComfyUI-QuasimondoNodes`.
3. Enter the `ComfyUI-QuasimondoNodes` directory.
4. Run `pip install -r .\requirements.txt` to install this project's dependencies.
5. Start ComfyUI as normal.


