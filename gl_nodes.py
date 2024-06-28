import numpy as np
import time
import os
import torch
from PIL import Image
import moderngl
import numbers
import struct
from torchvision.transforms import PILToTensor
import comfy.utils


def convert_from_shadertoy(code):
    code = "#define fragCoord gl_FragCoord.xy\n"+"#define iMouse vec4(0.0)\n"+code
    code = code.replace("void mainImage( out vec4 fragColor, in vec2 fragCoord )","void main()")
    code = code.replace("fragColor","gl_FragColor")
    code = code.replace("iResolution","u_resolution")
    code = code.replace("iTime","u_time")
    code = code.replace("iFrame","u_frame")
     
    print(code)
    
    return code
    
class CustomShader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "output_size": (
                    ["Custom", "texture_0", "texture_1", "texture_2", "texture_3"],
                ),
                "width": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "v0": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.00001,
                        "round": 0.00001,
                    },
                ),
                "v1": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.00001,
                        "round": 0.00001,
                    },
                ),
                "v2": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.00001,
                        "round": 0.00001,
                    },
                ),
                "v3": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.00001,
                        "round": 0.00001,
                    },
                ),
                "v4": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.00001,
                        "round": 0.00001,
                    },
                ),
                "v5": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.00001,
                        "round": 0.00001,
                    },
                ),
                "fragment_code": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "void main() {\n\tvec2 uv_text = gl_FragCoord.xy/u_resolution.xy;\n\tgl_FragColor = texture(texture_0,uv_text);\n}",
                    },
                ),
                "sampling": (["LINEAR", "NEAREST"],),
                "border": (
                    ["REPEAT/REPEAT", "CLAMP/CLAMP", "CLAMP/REPEAT", "REPEAT/CLAMP"],
                ),
                "timefactor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "shadertoy": (
                    "BOOLEAN",{"default":False}
                ),
            },
            "optional": {
                "texture_0": ("IMAGE",),
                "mask_0": ("MASK",),
                "texture_1": ("IMAGE",),
                "mask_1": ("MASK",),
                "texture_2": ("IMAGE",),
                "mask_2": ("MASK",),
                "texture_3": ("IMAGE",),
                "mask_3": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "mask", "help")
    # OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"

    def run(
        self,
        frames,
        output_size,
        width,
        height,
        v0,
        v1,
        v2,
        v3,
        v4,
        v5,
        fragment_code,
        sampling,
        border,
        timefactor,
        shadertoy,
        texture_0=None,
        mask_0=None,
        texture_1=None,
        mask_1=None,
        texture_2=None,
        mask_2=None,
        texture_3=None,
        mask_3=None,
    ):
    
        if shadertoy:
            fragment_code = convert_from_shadertoy(fragment_code)    
    

        t0 = None
        if not texture_0 is None:
            width0 = texture_0[0].size(1)
            height0 = texture_0[0].size(0)
            channels0 = texture_0[0].size(2)
            device0 = texture_0[0].device
            if not mask_0 is None:
                t0 = torch.cat((texture_0[0], mask_0[0].unsqueeze(-1)), dim=-1)
            else:
                t0 = torch.cat(
                    (
                        texture_0[0],
                        torch.ones(
                            (height0, width0, 1), dtype=texture_0.dtype, device=device0
                        ),
                    ),
                    dim=-1,
                )

        t1 = None
        if not texture_1 is None:
            width2 = texture_1[0].size(1)
            height2 = texture_1[0].size(0)
            channels2 = texture_1[0].size(2)
            device2 = texture_1[0].device

            if not mask_1 is None:
                t1 = torch.cat((texture_1[0], mask_1[0].unsqueeze(-1)), dim=-1)
            else:
                t1 = torch.cat(
                    (
                        texture_1[0],
                        torch.ones(
                            (height2, width2, 1), dtype=texture_1.dtype, device=device2
                        ),
                    ),
                    dim=-1,
                )

        t2 = None
        if not texture_2 is None:
            width3 = texture_2[0].size(1)
            height3 = texture_2[0].size(0)
            channels3 = texture_2[0].size(2)
            device3 = texture_2[0].device
            if not mask_2 is None:
                t2 = torch.cat((texture_2[0], mask_2[0].unsqueeze(-1)), dim=-1)
            else:
                t2 = torch.cat(
                    (
                        texture_2[0],
                        torch.ones(
                            (height3, width3, 1), dtype=texture_2.dtype, device=device3
                        ),
                    ),
                    dim=-1,
                )

        t3 = None
        if not texture_3 is None:
            width4 = texture_3[0].size(1)
            height4 = texture_3[0].size(0)
            channels4 = texture_3[0].size(2)
            device4 = texture_3[0].device
            if not mask_3 is None:
                t3 = torch.cat((texture_3[0], mask_3[0].unsqueeze(-1)), dim=-1)
            else:
                t3 = torch.cat(
                    (
                        texture_3[0],
                        torch.ones(
                            (height4, width4, 1), dtype=texture_3.dtype, device=device4
                        ),
                    ),
                    dim=-1,
                )

        ctx = moderngl.create_context(standalone=True, backend="egl")

        if output_size == "texture_0" and not t0 is None:
            width = width0
            height = height0
        elif output_size == "texture_1" and not t1 is None:
            width = width2
            height = height2
        elif output_size == "texture_2" and not t2 is None:
            width = width3
            height = height3
        elif output_size == "texture_3" and not t3 is None:
            width = width4
            height = height4
        fbo = ctx.simple_framebuffer((width, height), components=4)
        fbo.use()

        sampleMode = ctx.NEAREST if sampling == "NEAREST" else ctx.LINEAR

        vc = """
            #version 330
            in vec2 vert;
            
            void main() {
                gl_Position = vec4(vert, 0.0, 1.0);
            }
            """

        fc = (
            """
            #version 330
            precision mediump float;
            
            """
            + ("uniform sampler2D texture_0;" if not t0 is None else "")
            + """
            """
            + ("uniform sampler2D texture_1;" if not t1 is None else "")
            + """
            """
            + ("uniform sampler2D texture_2;" if not t2 is None else "")
            + """
            """
            + ("uniform sampler2D texture_3;" if not t3 is None else "")
            + """            
            
            uniform float v0;
            uniform float v1;
            uniform float v2;
            uniform float v3;
            uniform float v4;
            uniform float v5;
            
            uniform vec2 u_resolution;
            uniform float u_time;
            uniform int u_frame;
            """
            + fragment_code
        )

        prog = ctx.program(vertex_shader=vc, fragment_shader=fc)

        texture_coordinates = [0, 1, 1, 1, 0, 0, 1, 0]
        world_coordinates = [-1, -1, 1, -1, -1, 1, 1, 1]
        render_indices = [0, 1, 2, 1, 2, 3]

        vbo = ctx.buffer(struct.pack("8f", *world_coordinates))
        ibo = ctx.buffer(struct.pack("6I", *render_indices))

        vao_content = [
            (vbo, "2f", "vert"),
        ]

        vao = ctx.vertex_array(prog, vao_content, ibo)
        try:
            prog["texture_0"] = 0
        except:
            pass

        try:
            prog["texture_1"] = 1
        except:
            pass

        try:
            prog["texture_2"] = 2
        except:
            pass

        try:
            prog["texture_3"] = 3
        except:
            pass

        ptt = PILToTensor()
        filtered = []
        masks = []

        if not t0 is None:
            data = list(t0.flatten().float().cpu().numpy().astype(np.float32))
            num_frags = width0 * height0 * 4
            data = struct.pack(f"{num_frags}f", *data)

            texture = ctx.texture((width0, height0), 4, data=data, dtype="f4")
            texture.filter = sampleMode, sampleMode
            texture.repeat_x = border.split("/")[0] == "REPEAT"
            texture.repeat_y = border.split("/")[1] == "REPEAT"
            texture.use(location=0)

        if not t1 is None:
            data = list(t1.flatten().float().cpu().numpy().astype(np.float32))
            num_frags = width2 * height2 * 4
            data = struct.pack(f"{num_frags}f", *data)

            texture2 = ctx.texture((width2, height2), 4, data=data, dtype="f4")
            texture2.filter = sampleMode, sampleMode
            texture2.repeat_x = border.split("/")[0] == "REPEAT"
            texture2.repeat_y = border.split("/")[1] == "REPEAT"
            texture2.use(location=1)

        if not t2 is None:
            data = list(t2.flatten().float().cpu().numpy().astype(np.float32))
            num_frags = width3 * height3 * 4
            data = struct.pack(f"{num_frags}f", *data)

            texture3 = ctx.texture((width3, height3), 4, data=data, dtype="f4")
            texture3.filter = sampleMode, sampleMode
            texture3.repeat_x = border.split("/")[0] == "REPEAT"
            texture3.repeat_y = border.split("/")[1] == "REPEAT"
            texture3.use(location=2)

        if not t3 is None:
            data = list(t3.flatten().float().cpu().numpy().astype(np.float32))
            num_frags = width4 * height4 * 4
            data = struct.pack(f"{num_frags}f", *data)

            texture4 = ctx.texture((width4, height4), 4, data=data, dtype="f4")
            texture4.filter = sampleMode, sampleMode
            texture4.repeat_x = border.split("/")[0] == "REPEAT"
            texture4.repeat_y = border.split("/")[1] == "REPEAT"
            texture4.use(location=3)


        pbar = comfy.utils.ProgressBar(frames)

        for i in range(frames):
            if comfy.utils.PROGRESS_BAR_ENABLED:
                pbar.update_absolute(i + 1, frames)
        
            try:
                prog["u_time"].value = (i / frames)*timefactor
            except:
                pass
                
            try:
                prog["u_frame"].value = i
            except:
                pass

            try:
                prog["v0"].value = (
                    v0 if isinstance(v0, numbers.Number) else v0[min(i, len(v0) - 1)]
                )
            except Exception as e:
                print(e)
                pass
            try:
                prog["v1"].value = (
                    v1 if isinstance(v1, numbers.Number) else v1[min(i, len(v1) - 1)]
                )
            except Exception as e:
                print(e)
                pass
            try:
                prog["v2"].value = (
                    v2 if isinstance(v2, numbers.Number) else v2[min(i, len(v2) - 1)]
                )
            except Exception as e:
                print(e)
                pass
            try:
                prog["v3"].value = (
                    v3 if isinstance(v3, numbers.Number) else v3[min(i, len(v3) - 1)]
                )
            except Exception as e:
                print(e)
                pass
            try:
                prog["v4"].value = (
                    v4 if isinstance(v4, numbers.Number) else v4[min(i, len(v4) - 1)]
                )
            except Exception as e:
                print(e)
                pass
            try:
                prog["v5"].value = (
                    v5 if isinstance(v5, numbers.Number) else v5[min(i, len(v5) - 1)]
                )
            except Exception as e:
                print(e)
                pass

            try:
                prog["u_resolution"].value = [width, height]
            except:
                pass
            ctx.clear(0.0, 0.0, 0.0, 0.0)
            vao.render(mode=moderngl.TRIANGLES)

            data = fbo.read(components=4)
            image = Image.frombytes("RGBA", fbo.size, data)
            image = ptt(image)
            image = image.permute(1, 2, 0).float().mul(1.0 / 255.0)

            filtered.append(image[:, :, :3].unsqueeze(0))
            masks.append(image[:, :, 3].squeeze().unsqueeze(0))

        help = """
Predefined variables in shader:

    uniform float u_time;        // frame time 0.0 ... 1.0
    uniform vec2 u_resolution;   // output size in pixels
    
    uniform sampler2D texture_0; // input texture_0 if connected
    uniform sampler2D texture_1; // input texture_1 if connected
    uniform sampler2D texture_2; // input texture_2 if connected
    uniform sampler2D texture_3; // input texture_3 if connected    
            
    uniform float v0; 
    uniform float v1;
    uniform float v2;
    uniform float v3;
    uniform float v4;
    uniform float v5;
    
If masks are connected they get merged with their corresponding texture

Use gl_FragColor = vec4(...); to generate output.
    
"""

        return (torch.cat(filtered, dim=0), torch.cat(masks, dim=0), help)


class SpringMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture": ("IMAGE",),
                "frames": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "output_size": (["Custom", "texture", "motion_map"],),
                "width": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 65536}),
                "cols": ("INT", {"default": 64, "min": 1, "max": 2048}),
                "rows": ("INT", {"default": 64, "min": 1, "max": 2048}),
                "offset_sampling": (["RG", "GR", "BG", "GB", "RB", "BR"],),
                "offset_normalize": ("BOOLEAN", {"default": True}),
                "offset_zero": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "offset_scale_x": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "round": 0.0001,
                    },
                ),
                "offset_scale_y": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "round": 0.0001,
                    },
                ),
                "spring_constant_nb": (
                    "FLOAT",
                    {
                        "default": 0.333,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "spring_constant_grid": (
                    "FLOAT",
                    {
                        "default": 0.333,
                        "min": -65536.0,
                        "max": 65536.0,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "time_step": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.000001,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.001,
                    },
                ),
                "t_steps": ("INT", {"default": 1, "min": 1, "max": 500}),
                "cumulative": ("BOOLEAN", {"default": True}),
                "sampling": (["LINEAR", "NEAREST"],),
                "border": (
                    ["REPEAT/REPEAT", "CLAMP/CLAMP", "CLAMP/REPEAT", "REPEAT/CLAMP"],
                ),
                "show_vertices": ("BOOLEAN", {"default": False}),
            },
            "optional": {"motion_map": ("IMAGE",), "mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    # OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"

    def calculate_forces(self, points, connections, resting_lengths, spring_constant):
        forces = np.zeros_like(points)
        for i, (p1, p2) in enumerate(connections):
            length = np.linalg.norm(points[p1] - points[p2])
            if length > 0:
                force = spring_constant * (length - resting_lengths[i])
                forces[p1] += force * (points[p2] - points[p1]) / length
                forces[p2] += force * (points[p1] - points[p2]) / length
        return np.nan_to_num(forces)

    def calculate_grid_forces(self, points, grid, forces, spring_constant):

        for i in range(len(points)):
            length = np.linalg.norm(points[i] - grid[i])
            if length >= 0:
                force = spring_constant * length
                forces[i] += force * (grid[i] - points[i]) / length

        return np.nan_to_num(forces)

    def euler_method(
        self,
        points,
        grid,
        connections,
        resting_lengths,
        spring_constant_nb=0.1,
        spring_constant_grid=0.1,
        t=50,
        time_step=0.1,
    ):
        velocities = np.zeros_like(points)
        for _ in range(t):
            forces = self.calculate_forces(
                points, connections, resting_lengths, spring_constant_nb
            )
            forces = self.calculate_grid_forces(
                points, grid, forces, spring_constant_grid
            )
            velocities += forces * time_step
            points += velocities * time_step
        return points

    def run(
        self,
        texture,
        frames,
        output_size,
        width,
        height,
        cols,
        rows,
        offset_sampling,
        offset_normalize,
        offset_zero,
        offset_scale_x,
        offset_scale_y,
        spring_constant_nb,
        spring_constant_grid,
        time_step,
        t_steps,
        cumulative,
        sampling,
        border,
        show_vertices,
        motion_map=None,
        mask=None,
    ):

        t0 = None
        width0 = texture[0].size(1)
        height0 = texture[0].size(0)
        channels0 = texture[0].size(2)
        device0 = texture[0].device
        if not mask is None:
            t0 = torch.cat((texture[0], mask[0].unsqueeze(-1)), dim=-1)
        else:
            t0 = torch.cat(
                (
                    texture[0],
                    torch.ones(
                        (height0, width0, 1), dtype=texture.dtype, device=device0
                    ),
                ),
                dim=-1,
            )

        ctx = moderngl.create_context(standalone=True, backend="egl")

        if output_size == "texture" and not t0 is None:
            width = width0
            height = height0

        elif output_size == "motion_map" and not motion_map is None:
            width = motion_map[0].size(1)
            height = motion_map[0].size(0)

        fboaa = ctx.simple_framebuffer((width, height), components=4, samples=8)
        fboaa.use()
        fbo = ctx.simple_framebuffer((width, height), components=4)

        sampleMode = ctx.NEAREST if sampling == "NEAREST" else ctx.LINEAR

        vertex_shader = """
            #version 330
            in vec3 in_vert;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                gl_Position = vec4(in_vert, 1.0);
                v_uv = in_uv;
            }
        """

        if show_vertices:

            fragment_shader = """
                #version 330
                precision mediump float;
                
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0);
                }
             """
        else:
            fragment_shader = """
                #version 330
                precision mediump float;
                
                uniform sampler2D texture_0;
                
                in vec2 v_uv;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(texture_0, v_uv);
                }
             """

        prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        vertices = []
        uv_coordinates = []
        render_indices = []
        for row in range(rows):
            y = row / (rows - 1)
            for col in range(cols):
                x = col / (cols - 1)
                vertices += [x * 2.0 - 1.0, y * 2.0 - 1.0]
                uv_coordinates += [x, y]
                if col < cols - 1 and row < rows - 1:
                    index = row * cols + col
                    render_indices += [
                        index,
                        index + 1,
                        index + cols + 1,
                        index,
                        index + cols + 1,
                        index + cols,
                    ]

        vertices = np.array(vertices, dtype="f4")

        v2d = vertices.copy().reshape(-1, 2)
        v2d_t = v2d.copy()
        connections = []
        for row in range(rows - 1):
            for col in range(cols - 1):
                index = row * cols + col
                connections.append([index, index + 1])
                connections.append([index, index + cols])
                connections.append([index, index + cols + 1])
                connections.append([index + 1, index + cols])
        connections = np.array(connections)
        resting_lengths = np.linalg.norm(
            v2d[connections[:, 0]] - v2d[connections[:, 1]], axis=1
        )

        uv_coordinates = np.array(uv_coordinates, dtype="f4")
        uv_buffer = ctx.buffer(uv_coordinates)

        if not motion_map is None:
            if type(motion_map) is list:
                motion_map = torch.cat([m.unsqueeze(0) for m in motion_map], dim=0)

            motion_map = motion_map.clone()

            distortion = uv_coordinates.copy().reshape(rows * cols, 2)
            distortion[:, 1] *= motion_map.size(1)
            distortion[:, 0] *= motion_map.size(2)
            distortion = (distortion + 0.5).astype(int)
            distortion[:, 1] = distortion[:, 1].clip(0, motion_map.size(1) - 1)
            distortion[:, 0] = distortion[:, 0].clip(0, motion_map.size(2) - 1)
            distortion = distortion[:, 1] * motion_map.size(2) + distortion[:, 0]
            offset_channels = np.array(
                [
                    [0, 1, 2]["RGB".index(offset_sampling[1])],
                    [0, 1, 2]["RGB".index(offset_sampling[0])],
                ]
            )
            o_s = np.array(
                [
                    [
                        offset_scale_x
                        / (motion_map.size(1) if offset_normalize else 1.0),
                        offset_scale_y
                        / (motion_map.size(2) if offset_normalize else 1.0),
                    ]
                ]
            )
            v2d += o_s * (
                (
                    motion_map[0, :, :, offset_channels]
                    .view(-1, 2)[distortion]
                    .cpu()
                    .numpy()
                )
                - offset_zero
            )
            v2d = self.euler_method(
                v2d,
                v2d_t,
                connections,
                resting_lengths,
                spring_constant_nb=spring_constant_nb,
                spring_constant_grid=spring_constant_grid,
                t=t_steps,
                time_step=time_step,
            )
            vertices = v2d.flatten()

        vbo = ctx.buffer(vertices)

        render_indices = np.array(render_indices, dtype="i4")
        index_buffer = ctx.buffer(render_indices)

        if show_vertices:
            vao_content = [
                vbo.bind("in_vert", layout="2f"),
            ]
        else:
            vao_content = [
                vbo.bind("in_vert", layout="2f"),
                uv_buffer.bind("in_uv", layout="2f"),
            ]

        vao = ctx.vertex_array(
            program=prog, content=vao_content, index_buffer=index_buffer
        )
        try:
            prog["texture_0"] = 0
            data = list(t0.flatten().float().cpu().numpy().astype(np.float32))
            num_frags = width0 * height0 * 4
            data = struct.pack(f"{num_frags}f", *data)

            texture = ctx.texture((width0, height0), 4, data=data, dtype="f4")
            texture.filter = sampleMode, sampleMode
            texture.repeat_x = border.split("/")[0] == "REPEAT"
            texture.repeat_y = border.split("/")[1] == "REPEAT"
            texture.use(location=0)
        except:
            pass

        # ctx.enable_only(moderngl.NOTHING)
        if show_vertices:
            ctx.wireframe = True
        ptt = PILToTensor()
        filtered = []
        masks = []
        for i in range(frames):
            ctx.clear(0.0, 0.0, 0.0, 0.0)
            vao.render(mode=moderngl.TRIANGLES)

            ctx.copy_framebuffer(dst=fbo, src=fboaa)
            data = fbo.read(components=4)

            image = Image.frombytes("RGBA", fbo.size, data)
            image = ptt(image)
            image = image.permute(1, 2, 0).float().mul(1.0 / 255.0)

            filtered.append(image[:, :, :3].unsqueeze(0))
            masks.append(image[:, :, 3].squeeze().unsqueeze(0))

            if frames > 1:
                if i < motion_map.size(0):
                    distortion = uv_coordinates.copy().reshape(rows * cols, 2)
                    distortion[:, 1] *= motion_map.size(1)
                    distortion[:, 0] *= motion_map.size(2)
                    distortion = (distortion + 0.5).astype(int)
                    distortion[:, 1] = distortion[:, 1].clip(0, motion_map.size(1) - 1)
                    distortion[:, 0] = distortion[:, 0].clip(0, motion_map.size(2) - 1)
                    distortion = (
                        distortion[:, 1] * motion_map.size(2) + distortion[:, 0]
                    )
                    offset_channels = np.array(
                        [
                            [0, 1, 2]["RGB".index(offset_sampling[1])],
                            [0, 1, 2]["RGB".index(offset_sampling[0])],
                        ]
                    )
                    o_s = np.array(
                        [
                            [
                                offset_scale_x
                                / (motion_map.size(1) if offset_normalize else 1.0),
                                offset_scale_y
                                / (motion_map.size(2) if offset_normalize else 1.0),
                            ]
                        ]
                    )

                    if not cumulative:
                        v2d = v2d_t.copy()
                    v2d += o_s * (
                        (
                            motion_map[i, :, :, offset_channels]
                            .view(-1, 2)[distortion]
                            .cpu()
                            .numpy()
                        )
                        - offset_zero
                    )

                v2d = self.euler_method(
                    v2d,
                    v2d_t,
                    connections,
                    resting_lengths,
                    spring_constant_nb=spring_constant_nb,
                    spring_constant_grid=spring_constant_grid,
                    t=t_steps,
                    time_step=time_step,
                )
                vbo.write(v2d.flatten())

        return (torch.cat(filtered, dim=0), torch.cat(masks, dim=0))
