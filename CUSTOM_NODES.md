# Custom Node Documentation

## CPPN Nodes

### CPPNGenerator

**Category:** `images`

**Purpose:** Generates images using a Compositional Pattern Producing Network (CPPN). This node allows for the creation of complex, abstract patterns and textures.

**Input Types:**

*   `seed`: INT (default: 0, min: 0, max: very large) - Seed for random number generation.
*   `width`: INT (default: 1024, min: 1, max: 8192) - Width of the output image.
*   `height`: INT (default: 1024, min: 1, max: 8192) - Height of the output image.
*   `framecount`: INT (default: 1, min: 1, max: 512) - Number of frames to generate (for animations).
*   `dim_z`: INT (default: 32, min: 1, max: 512) - Dimension of the latent vector z.
*   `channels_in`: INT (default: 32, min: 1, max: 512) - Number of input channels for the network.
*   `layers`: INT (default: 3, min: 0, max: 16) - Number of layers in the CPPN.
*   `first_activation`: STRING (options: "Tanh", "Sigmoid", "Sin", "ReLU", "ELU", "Softplus", "Modulo") - Activation function for the first layer.
*   `middle_activations`: STRING (options: "Tanh", "Sigmoid", "Sin", "ReLU", "ELU", "Softplus", "Modulo") - Activation function for the middle layers.
*   `last_activation`: STRING (options: "Tanh", "Sigmoid", "Sin", "ReLU", "ELU", "Softplus", "Modulo", default: "Sigmoid") - Activation function for the last layer.
*   `scale`: FLOAT (default: 1.0, min: -1024.0, max: 1024.0) - Scale factor for the coordinates.
*   `x_offset`: FLOAT (default: 0.0) - X-offset for the coordinates.
*   `y_offset`: FLOAT (default: 0.0) - Y-offset for the coordinates.
*   `z_offset`: FLOAT (default: 0.0) - Offset for the latent vector z.
*   `mean`: FLOAT (default: 0.0) - Mean for weight initialization.
*   `std`: FLOAT (default: 1.0) - Standard deviation for weight initialization.
*   `bias_mean`: FLOAT (default: 0.0) - Mean for bias initialization.
*   `bias_std`: FLOAT (default: 1.0) - Standard deviation for bias initialization.
*   `zero_bias`: BOOLEAN (default: True) - If true, biases are initialized to zero.
*   `output_factor`: FLOAT (default: 1.0) - Factor to multiply the output by.
*   `output_offset`: FLOAT (default: 0.0) - Offset to add to the output.
*   `mode`: STRING (options: ["Grayscale", "RGB"]) - Output image mode.
*   `normalize`: BOOLEAN (default: False) - If true, normalizes the output image.
*   `images` (optional): IMAGE - Optional input images. (The script implies this is an input, but its usage in `exec` is not immediately clear without further context of the ComfyUI environment)

**Return Types:**

*   `images`: IMAGE - The generated image(s).

**Functionality:**

The `CPPNGenerator` node uses a neural network (CPPN) to generate images. The network takes coordinates (x, y, radius) and a latent vector (z) as input and outputs pixel values. The structure and behavior of the CPPN can be controlled by various parameters like the number of layers, activation functions, and initialization parameters for weights and biases. It can generate single images or sequences of frames.

## Extra Nodes

### ImageBlendMaskBatch

**Category:** `image`

**Purpose:** Blends two batches of images using a mask and a blend percentage.

**Input Types:**

*   `image_a`: IMAGE - The first batch of images (background).
*   `image_b`: IMAGE - The second batch of images (foreground).
*   `mask`: IMAGE - A mask image (or batch of masks) to define the blend areas. Grayscale values are derived from RGB.
*   `blend_percentage`: FLOAT (default: 0.5, min: 0.0, max: 1.0) - The opacity of `image_b` in the masked areas. 0.0 means only `image_a` is visible, 1.0 means `image_b` fully replaces `image_a` in the masked region, scaled by the mask's intensity.

**Return Types:**

*   `IMAGE`: IMAGE - The resulting batch of blended images.

**Functionality:**

This node performs a masked blend between two image batches. The mask is converted to grayscale. If the mask's dimensions don't match the images, it's resized using bicubic interpolation. If a single mask is provided for a batch of images, it's cloned for each image. The blending formula is effectively `image_a * (1.0 - (mask_intensity * blend_percentage)) + image_b * mask_intensity * blend_percentage`.

### ColorMatch

**Category:** `image`

**Purpose:** Adjusts the color histogram of an image (or batch of images) to match that of a reference image (or batch).

**Input Types:**

*   `image`: IMAGE - The input image(s) whose colors will be adjusted.
*   `reference_image`: IMAGE - The reference image(s) from which to source the color histogram.
*   `blend_factor`: FLOAT (default: 1.0, min: 0.0, max: 2.0) - How much of the color matching to apply. 1.0 applies the full match, 0.0 applies none (original image), values between 0 and 1 blend between original and matched, and values > 1 can exaggerate the effect.

**Return Types:**

*   `IMAGE`: IMAGE - The image(s) with adjusted colors.

**Functionality:**

This node uses histogram matching (`hist_match_rgb` from `colorTransfer.py`) to transfer the color characteristics of the `reference_image` to the `image`. If a single reference image is provided for a batch of input images, it's cloned. The `blend_factor` allows for controlling the intensity of the color matching effect, blending the original image with the color-matched version.

## GL Nodes

### CustomShader

**Category:** `image/postprocessing`

**Purpose:** Applies a custom GLSL fragment shader to input textures, allowing for a wide range of GPU-accelerated image processing effects.

**Input Types:**

*   `frames`: INT (default: 1, min: 1, max: 9999) - Number of frames to render.
*   `output_size`: STRING (options: ["Custom", "texture_0", "texture_1", "texture_2", "texture_3"]) - Determines the output resolution. "Custom" uses `width` and `height` inputs; otherwise, it uses the dimensions of the specified input texture.
*   `width`: INT (default: 1024, min: 1, max: 65536) - Output width if `output_size` is "Custom".
*   `height`: INT (default: 1024, min: 1, max: 65536) - Output height if `output_size` is "Custom".
*   `v0` - `v5`: FLOAT (default: 0.0) - Six user-definable float values passed as uniforms to the shader. These can be animated if an array/list of floats is passed.
*   `fragment_code`: STRING (multiline, default: basic texture lookup) - The GLSL fragment shader code.
*   `sampling`: STRING (options: ["LINEAR", "NEAREST"]) - Texture sampling mode for input textures.
*   `border`: STRING (options: ["REPEAT/REPEAT", "CLAMP/CLAMP", "CLAMP/REPEAT", "REPEAT/CLAMP"]) - Texture wrap mode for input textures (X/Y).
*   `timefactor`: FLOAT (default: 1.0) - Multiplier for the `u_time` uniform.
*   `shadertoy`: BOOLEAN (default: False) - If true, attempts to convert Shadertoy-specific variables/functions to standard GLSL.
*   `texture_0` (optional): IMAGE - First input texture.
*   `mask_0` (optional): MASK - Mask for `texture_0`. If provided, `texture_0` becomes RGBA with the mask as the alpha channel.
*   `texture_1` (optional): IMAGE - Second input texture.
*   `mask_1` (optional): MASK - Mask for `texture_1`.
*   `texture_2` (optional): IMAGE - Third input texture.
*   `mask_2` (optional): MASK - Mask for `texture_2`.
*   `texture_3` (optional): IMAGE - Fourth input texture.
*   `mask_3` (optional): MASK - Mask for `texture_3`.

**Return Types:**

*   `images`: IMAGE - The rendered image(s) from the shader.
*   `mask`: MASK - The alpha channel output from the shader, if applicable.
*   `help`: STRING - A help string listing available uniforms and basic usage.

**Functionality:**

This node sets up a WebGL (via Moderngl) environment to run a user-provided fragment shader. It can take up to four textures (and corresponding masks) as inputs. The shader has access to several predefined uniforms:
    *   `uniform float u_time;`        // Time, typically 0.0 to 1.0 over `frames`, scaled by `timefactor`.
    *   `uniform vec2 u_resolution;`   // Output resolution in pixels.
    *   `uniform int u_frame;`         // Current frame number.
    *   `uniform sampler2D texture_0;` // Input texture_0 (if connected).
    *   `uniform sampler2D texture_1;` // Input texture_1 (if connected).
    *   `uniform sampler2D texture_2;` // Input texture_2 (if connected).
    *   `uniform sampler2D texture_3;` // Input texture_3 (if connected).
    *   `uniform float v0;` ... `v5;`  // User-defined float values.
The output is `gl_FragColor`. If `shadertoy` is enabled, it performs some basic replacements (e.g., `iResolution` to `u_resolution`).

### SpringMesh

**Category:** `image/postprocessing`

**Purpose:** Deforms an input texture using a 2D spring-mass mesh simulation. The mesh's movement can be influenced by a motion map.

**Input Types:**

*   `texture`: IMAGE - The input texture to be warped.
*   `frames`: INT (default: 1, min: 1, max: 9999) - Number of simulation frames to render.
*   `output_size`: STRING (options: ["Custom", "texture", "motion_map"]) - Determines output resolution.
*   `width`: INT (default: 1024) - Output width if `output_size` is "Custom".
*   `height`: INT (default: 1024) - Output height if `output_size` is "Custom".
*   `cols`: INT (default: 64, min: 1, max: 2048) - Number of columns in the simulation mesh.
*   `rows`: INT (default: 64, min: 1, max: 2048) - Number of rows in the simulation mesh.
*   `offset_sampling`: STRING (options: ["RG", "GR", "BG", "GB", "RB", "BR"]) - Specifies which channels of the `motion_map` control X and Y offsets.
*   `offset_normalize`: BOOLEAN (default: True) - If true, normalizes offset values from `motion_map` based on its dimensions.
*   `offset_zero`: FLOAT (default: 0.0, min: -1.0, max: 1.0) - Zero point for offset calculation from `motion_map`.
*   `offset_scale_x`: FLOAT (default: 0.1) - Scale factor for X-offsets from `motion_map`.
*   `offset_scale_y`: FLOAT (default: 0.1) - Scale factor for Y-offsets from `motion_map`.
*   `spring_constant_nb`: FLOAT (default: 0.333) - Spring constant for connections between neighboring mesh points.
*   `spring_constant_grid`: FLOAT (default: 0.333) - Spring constant for connections of mesh points to their original grid positions.
*   `time_step`: FLOAT (default: 0.5, min: 0.000001, max: 1.0) - Time step for the Euler physics simulation.
*   `t_steps`: INT (default: 1, min: 1, max: 500) - Number of simulation steps per frame.
*   `cumulative`: BOOLEAN (default: True) - If true, motion map effects accumulate over frames; otherwise, the mesh resets based on the motion map each frame.
*   `sampling`: STRING (options: ["LINEAR", "NEAREST"]) - Texture sampling for the input `texture`.
*   `border`: STRING (options: ["REPEAT/REPEAT", ...]) - Texture wrap mode for `texture`.
*   `show_vertices`: BOOLEAN (default: False) - If true, renders the mesh vertices instead of the texture.
*   `motion_map` (optional): IMAGE - An image used to displace mesh vertices. Sampled according to `offset_sampling`.
*   `mask` (optional): MASK - Optional mask for the input `texture`.

**Return Types:**

*   `images`: IMAGE - The deformed image(s).
*   `mask`: MASK - The deformed mask (alpha channel of the texture).

**Functionality:**

This node creates a grid of points (a mesh) defined by `rows` and `cols`. This mesh is used to sample the input `texture`. The mesh points are simulated as masses connected by springs.
There are two types of spring forces:
1.  Between neighboring points in the mesh (`spring_constant_nb`).
2.  Between each point and its original, undeformed position in the grid (`spring_constant_grid`).
If a `motion_map` is provided, its pixel values (selected by `offset_sampling`) are used to further displace the mesh points. The simulation (Euler method) runs for `t_steps` per output frame. The `cumulative` option controls whether displacements from the `motion_map` are applied fresh each frame or added to the previous state. The final deformed mesh is then used to render the input `texture`.

## Image Nodes

### ImageToOpticalFlow

**Category:** `image`

**Purpose:** Converts selected channels of an input image into a 2-channel optical flow format.

**Input Types:**

*   `image`: IMAGE - The input image batch.
*   `x`: STRING (options: ["R", "G", "B"], default: "R") - Channel to use for the X component of the flow.
*   `x_scale`: FLOAT (default: 256.0) - Scale factor for the X component.
*   `x_offset`: FLOAT (default: -128.0) - Offset for the X component.
*   `y`: STRING (options: ["R", "G", "B"], default: "G") - Channel to use for the Y component of the flow.
*   `y_scale`: FLOAT (default: 256.0) - Scale factor for the Y component.
*   `y_offset`: FLOAT (default: -128.0) - Offset for the Y component.

**Return Types:**

*   `flow`: OPTICAL_FLOW - The resulting optical flow data (as a NumPy array).

**Functionality:**

This node takes an image and extracts two of its color channels (e.g., Red for X, Green for Y) to create a 2D optical flow field. The selected channels are scaled and offset according to the input parameters. The output is a NumPy array suitable for use with nodes expecting optical flow data.

### ShiftMask

**Category:** `image`

**Purpose:** Shifts a mask by specified integer offsets.

**Input Types:**

*   `mask`: MASK - The input mask (can be a single mask or a batch).
*   `x_offset`: INT (default: 0) - The amount to shift the mask horizontally.
*   `y_offset`: INT (default: 0) - The amount to shift the mask vertically.

**Return Types:**

*   `mask`: MASK - The shifted mask.

**Functionality:**

This node performs a simple roll operation on the input mask tensor. The shift is done with wrap-around (circular shift). It handles both 2D (single mask) and 3D (batch of masks) tensors.

### SlitScan

**Category:** `image`

**Purpose:** Creates a slit-scan effect by sampling pixels from different frames of an image sequence based on mask values.

**Input Types:**

*   `images`: IMAGE - A batch of images (frames of a sequence).
*   `masks`: MASK - A batch of masks corresponding to the images, or a single mask.
*   `mask_time_factor`: FLOAT (default: 1.0) - A factor applied to the frame index when calculating temporal offset from masks.
*   `t_offset`: FLOAT (default: 0.0) - A global temporal offset applied to mask values.
*   `time_depth`: FLOAT (default: 1.0) - Scales the mask values to control the range of frames sampled.
*   `blend_power`: FLOAT (default: 1.0, min: 1.0, max: 32.0) - Controls the blending curve between sampled frames. Higher values create sharper transitions.
*   `wrap_t`: BOOLEAN (default: True) - If true, time (frame index) wraps around; otherwise, it's clamped.

**Return Types:**

*   `images`: IMAGE - The resulting slit-scan image batch.

**Functionality:**

The `SlitScan` node generates an output image sequence where each pixel in a frame can be sourced from a different frame in the input `images` sequence. The choice of source frame is determined by the corresponding `masks`.
Mask values are processed: `processed_mask = (mask_value * time_depth) + t_offset + (current_frame_index / total_frames * mask_time_factor)`.
This `processed_mask` value (either wrapped or clamped between 0 and 1) is then scaled by `images.size(0) - 1` to get a frame index.
Blending occurs between the floor and ceil of this frame index, with `blend_power` affecting the transition. If mask dimensions don't match image dimensions, masks are resized.

### DistanceMap

**Category:** `generator`

**Purpose:** Generates a distance transform map from a binary mask and finds coordinates of a point based on distance criteria.

**Input Types:**

*   `mask`: MASK - The input mask (expects a single mask from the batch).
*   `xy_mode`: STRING (options: ["max", "min", "median"]) - Determines how to select the output (x, y) coordinates:
    *   `max`: Point furthest from a zero-value pixel.
    *   `min`: Point closest to a zero-value pixel (likely on an edge if mask is not all white).
    *   `median`: Point with the median distance value.

**Return Types:**

*   `image`: MASK - The distance transform image (normalized 0-1).
*   `x`: INT - X-coordinate of the selected point.
*   `y`: INT - Y-coordinate of the selected point.

**Functionality:**

Takes the first mask from the input batch, converts it to a binary image (threshold at 127). It then computes the Euclidean distance transform (OpenCV `distanceTransform` with `DIST_L2` and `DIST_MASK_PRECISE`). The resulting distance map is normalized to the 0-1 range. Based on `xy_mode`, it finds a point (x, y) corresponding to the maximum, minimum, or median distance value in the map.

### TemporalBlur

**Category:** `generator` (Should likely be `image/filter` or `image/temporal`)

**Purpose:** Applies a temporal blur to a sequence of images by averaging frames with a falloff.

**Input Types:**

*   `images`: IMAGE - The input image sequence (batch).
*   `frames_backward`: INT (default: 8, min: 0) - Number of previous frames to include in the blur.
*   `frames_forward`: INT (default: 0, min: 0) - Number of subsequent frames to include in the blur.
*   `falloff`: FLOAT (default: 0.5, min: 0.0, max: 1.0) - Factor by which the contribution of adjacent frames is reduced. A falloff of 1.0 means equal weight (box blur), 0.0 means only the current frame is used (no blur).

**Return Types:**

*   `image`: IMAGE - The temporally blurred image sequence.

**Functionality:**

For each frame in the input `images` sequence, this node calculates an average of itself, `frames_backward` previous frames, and `frames_forward` subsequent frames. The contribution of each neighboring frame is weighted by `falloff` raised to the power of its distance from the current frame. For example, frame `i-1` has weight `falloff`, frame `i-2` has weight `falloff^2`, and so on. The final blurred frame is a weighted average.

### PreviewMask

**Category:** `mask` (Utility Node)

**Purpose:** Saves input masks as temporary image files and displays them in the ComfyUI interface.

**Input Types:**

*   `masks`: MASK - The input masks to be previewed.

**Return Types:** None (Output Node)

**Functionality:**

This is an output node. It takes a batch of masks (or a single 2D mask), converts each to a PIL Image, and saves it as a temporary JPG file in the ComfyUI temp directory. The UI then displays these images. The filenames are prefixed with `qtmp_` followed by a timestamp.

### CoordinatesFromMask

**Category:** `generator`

**Purpose:** Extracts a list of (x, y) coordinates from a mask based on distance from edges and desired density, using a form of Poisson disk sampling.

**Input Types:**

*   `mask`: MASK - The input mask (uses the first mask in the batch).
*   `percentage`: FLOAT (default: 0.1, min: 0.0, max: 1.0) - Target percentage of valid points to select, relative to points meeting `min_distance_from_edge`.
*   `min_distance_from_edge`: INT (default: 5, min: 0, max: 100) - Minimum distance a point must be from a zero-value pixel (edge) in the mask to be considered.
*   `max_points`: INT (default: 100, min: 1, max: 1000) - Maximum number of coordinate points to return.
*   `seed`: INT (default: 1234) - Seed for the random number generator used in sampling.

**Return Types:**

*   `coordinates`: STRING - A JSON string representing a list of dictionaries, e.g., `[{"x": x1, "y": y1}, {"x": x2, "y": y2}, ...]`.
*   `point_mask`: MASK - A new mask with white pixels at the selected coordinates.

**Functionality:**

1.  Takes the first mask from the batch and thresholds it.
2.  Computes a distance transform to find how far each white pixel is from the nearest black pixel (edge).
3.  Filters points that are at least `min_distance_from_edge` away from an edge.
4.  Determines the number of points to select: `min(int(len(valid_points) * percentage), max_points)`.
5.  Uses a custom `sample_points` method to select points. This method aims for a distribution similar to Poisson disk sampling:
    *   It shuffles the valid points.
    *   It iterates through points, adding a point if it's not too close to already selected points in a grid-based system (controlled by `grid_size` which is related to `k`).
    *   If not enough points are selected in the first pass, it randomly picks more from the remaining valid points.
6.  The selected (y, x) coordinates are formatted into a JSON string.
7.  A new mask is generated with white pixels at the selected coordinates.

## Queue Nodes

### VideoQueueManager

**Category:** `queue`

**Purpose:** Manages a queue of video files from a specified path. It iterates through videos and their frames based on a global index and batch size, allowing processing of large video datasets frame by frame or in batches.

**Input Types:**

*   `index`: INT (default: 0, min: 0) - The global frame index (or batch index if `batch_size` > 1) across all videos in the path.
*   `path`: STRING - The directory path containing video files.
*   `batch_size`: INT (default: 1, min: 1) - Number of frames per batch. The `index` input is effectively a batch index.
*   `string_decimals`: INT (default: 4, min: 0, max: 16) - Number of digits for zero-padding the `current_frame_string` output.

**Return Types:**

*   `full_path`: STRING - The full path to the current video file.
*   `filename`: STRING - The filename of the current video.
*   `current_frame`: INT - The starting frame number within the current video for the current batch/index.
*   `current_frame_string`: STRING - The `current_frame` number formatted as a zero-padded string.
*   `total_frames`: INT - The total number of frames in the current video file.

**Functionality:**

The `VideoQueueManager` scans a directory for video files (common suffixes like mp4, avi, gif, etc.). It sorts these files alphanumerically.
It then determines which video file and which specific frame (or starting frame of a batch) corresponds to the global `index`, considering the `batch_size`.
For example, if `batch_size` is 1, and the first video has 100 frames and the second has 50:
- `index` 0 returns frame 0 of video 1.
- `index` 99 returns frame 99 of video 1.
- `index` 100 returns frame 0 of video 2.
If `batch_size` is 10, and the first video has 100 frames:
- `index` 0 returns starting frame 0 of video 1.
- `index` 9 returns starting frame 90 of video 1.
- `index` 10 returns starting frame 0 of video 2 (assuming video 2 is next).
If no video/frame corresponds to the index (e.g., index is too high), it returns empty strings and -1 for frame numbers.

### FolderQueueManager

**Category:** `queue`

**Purpose:** Manages a queue of subdirectories within a specified path.

**Input Types:**

*   `index`: INT (default: 0, min: 0) - The index of the folder to select from the sorted list of subdirectories.
*   `path`: STRING - The directory path containing the subdirectories.

**Return Types:**

*   `full_path`: STRING - The full path to the selected subdirectory.
*   `foldername`: STRING - The name of the selected subdirectory.

**Functionality:**

This node lists all items in the given `path`, sorts them alphanumerically, and filters for directories. It then returns the full path and name of the directory at the specified `index`. If the index is out of bounds, it returns empty strings.
