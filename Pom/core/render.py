from Pom.core.opengl_classes import *
import Pom.core.config as cfg
from Pom.core.config import project_configuration, feature_library
import json
import glfw
from skimage import measure
from scipy.ndimage import label, binary_dilation, gaussian_filter1d, gaussian_filter

# TODO: add ray-traced volume rendering
# TODO: changed the zmin to 1e3, see if that doesn't clip certain objects!
PIXEL_SCALE = 950


class Renderer:
    def __init__(self, image_size=512):
        if not glfw.init():
            raise Exception("GLFW initialization failed.")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(10, 10, "Offscreen Render", None, None)
        glfw.make_context_current(self.window)
        glEnable(GL_MULTISAMPLE)
        self.surface_model_shader = Shader(os.path.join(cfg.root, "shaders", "se_surface_model_shader.glsl"))
        self.edge_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_edge_detect.glsl"))
        self.depth_mask_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_mask_shader.glsl"))
        self.ray_trace_shader = Shader(os.path.join(cfg.root, "shaders", "raytrace_volume.glsl"))
        self.ndc_img_shader = Shader(os.path.join(cfg.root, "shaders", "ndc_image.glsl"))

        self.style = 0

        self.image_size = image_size
        self.texture3d = glGenTextures(1, GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, self.texture3d)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.scene_fbo = FrameBuffer(width=self.image_size, height=self.image_size, texture_format="rgba32f")
        self.scene_fbo_b = FrameBuffer(width=self.image_size, height=self.image_size, texture_format="rgba32f")
        self.depth_fbo_a = FrameBuffer(width=self.image_size, height=self.image_size, texture_format="rgba32f")
        self.depth_fbo_b = FrameBuffer(width=self.image_size, height=self.image_size, texture_format="rgba32f")
        self.volume_fbo = FrameBuffer(width=self.image_size, height=self.image_size, texture_format="rgba32f")
        self.box_va = VertexArray(attribute_format="xyz")
        self.box_va_shape = (0, 0, 0)
        self.ndc_screen_va = VertexArray(attribute_format="xy")
        self.ndc_screen_va.update(VertexBuffer([-1, -1, 1, -1, 1, 1, -1, 1]), IndexBuffer([0, 1, 2, 0, 2, 3]))

        # TODO: make it possible to use style = 0, or 1, or 2, for different styles.
        self.RENDER_SILHOUETTES = True
        self.RENDER_SILHOUETTES_ALPHA = 0.5
        self.RENDER_SILHOUETTES_THRESHOLD = 0.025

        self.camera = Camera3D(self.image_size)
        self.camera.on_update()
        self.volume_fbo_active = False
        self.light = Light3D()
        self.ambient_strength = 0.75
        self.background_colour = (1.0, 1.0, 1.0, 0.0)

    @staticmethod
    def poll_gl_states():
        # List of capabilities to check
        capabilities = [
            (GL_BLEND, 'GL_BLEND'),
            (GL_CULL_FACE, 'GL_CULL_FACE'),
            (GL_DEPTH_TEST, 'GL_DEPTH_TEST'),
            (GL_DITHER, 'GL_DITHER'),
            (GL_POLYGON_OFFSET_FILL, 'GL_POLYGON_OFFSET_FILL'),
            (GL_SAMPLE_ALPHA_TO_COVERAGE, 'GL_SAMPLE_ALPHA_TO_COVERAGE'),
            (GL_SAMPLE_COVERAGE, 'GL_SAMPLE_COVERAGE'),
            (GL_SCISSOR_TEST, 'GL_SCISSOR_TEST'),
            (GL_STENCIL_TEST, 'GL_STENCIL_TEST'),
            (GL_MULTISAMPLE, 'GL_MULTISAMPLE'),
            # Add more capabilities as needed
        ]
        # Poll and print the state of each capability
        for cap, cap_name in capabilities:
            print(cap, cap_name)
            state = glIsEnabled(cap)
            print(f'{cap_name}: {"Enabled" if state else "Disabled"}')

    def delete(self):
        glfw.terminate()

    def render(self, renderables_list):
        self.volume_fbo_active = False
        # render surface models first, volumes second.
        m_surfaces = [s for s in renderables_list if isinstance(s, SurfaceModel)]
        self.render_surface_models(m_surfaces)

        m_volumes = [v for v in renderables_list if isinstance(v, VolumeModel)]
        if not m_volumes:
            return

        self.render_depth_masks(m_volumes[0].data.shape)
        # depth to start sampling at is now in fbo b
        # depth to stop sampling at is now in fbo a

        # ray trace the volumes one by one.
        self.volume_fbo.bind()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        Z, X, Y = m_volumes[0].data.shape
        self.ray_trace_shader.bind()
        self.ray_trace_shader.uniformmat4("ipMat", self.camera.ipmat)
        self.ray_trace_shader.uniformmat4("ivMat", self.camera.ivmat)
        self.ray_trace_shader.uniformmat4("pMat", self.camera.pmat)
        self.ray_trace_shader.uniform1f("near", self.camera.clip_near)
        self.ray_trace_shader.uniform1f("far", self.camera.clip_far)
        self.ray_trace_shader.uniform2f("viewportSize", (self.image_size, self.image_size))
        self.ray_trace_shader.uniform1f("pixelSize", PIXEL_SCALE / m_volumes[0].data.shape[1])
        self.ray_trace_shader.uniform1i("Z", Z)
        self.ray_trace_shader.uniform1i("Y", Y)
        self.ray_trace_shader.uniform1i("X", X)
        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self.depth_fbo_b.depth_texture_renderer_id)
        glActiveTexture(GL_TEXTURE0 + 2)
        glBindTexture(GL_TEXTURE_2D, self.depth_fbo_a.depth_texture_renderer_id)
        glBindImageTexture(3, self.volume_fbo.texture.renderer_id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F)

        for v in m_volumes:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_3D, self.texture3d)
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RED, v.data.shape[1], v.data.shape[2], v.data.shape[0], 0, GL_RED, GL_FLOAT, v.data.flatten())
            self.ray_trace_shader.uniform3f("C", v.colour)
            glDispatchCompute(self.image_size // 32, self.image_size // 32, 1)
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
            glFinish()
        self.volume_fbo_active = True
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.scene_fbo.bind()
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.volume_fbo.texture.renderer_id)
        self.ndc_img_shader.bind()
        self.ndc_screen_va.bind()
        glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.ndc_screen_va.unbind()
        self.ndc_img_shader.unbind()
        # combine the isosurface image and the volume images

    def render_surface_models(self, surface_models):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.scene_fbo.bind()
        self.surface_model_shader.bind()
        self.surface_model_shader.uniformmat4("vpMat", self.camera.matrix)
        self.surface_model_shader.uniform3f("viewDir", self.camera.get_view_direction())
        self.surface_model_shader.uniform3f("lightDir", self.light.vec)
        self.surface_model_shader.uniform1f("ambientStrength", self.ambient_strength)
        self.surface_model_shader.uniform1f("lightStrength", self.light.strength)
        self.surface_model_shader.uniform3f("lightColour", self.light.colour)
        self.surface_model_shader.uniform1i("style", self.style)
        glEnable(GL_DEPTH_TEST)
        alpha_sorted_surface_models = sorted(surface_models, key=lambda x: x.alpha, reverse=True)
        for s in alpha_sorted_surface_models:
            self.surface_model_shader.uniform4f("color", [*s.colour, s.alpha])
            for blob in s.blobs.values():
                if blob.complete and not blob.hide:
                    blob.va.bind()
                    glDrawElements(GL_TRIANGLES, blob.va.indexBuffer.getCount(), GL_UNSIGNED_INT, None)
                    blob.va.unbind()
        self.surface_model_shader.unbind()
        glDisable(GL_DEPTH_TEST)


        if len(alpha_sorted_surface_models) > 0:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.scene_fbo.framebufferObject)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.scene_fbo_b.framebufferObject)
            glBlitFramebuffer(0, 0, self.image_size, self.image_size, 0, 0, self.image_size, self.image_size, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
            glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo.framebufferObject)
            self.edge_shader.bind()
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.scene_fbo_b.depth_texture_renderer_id)
            self.edge_shader.uniform1f("threshold", self.RENDER_SILHOUETTES_THRESHOLD)
            self.edge_shader.uniform1f("edge_alpha", self.RENDER_SILHOUETTES_ALPHA)
            self.edge_shader.uniform1f("zmin", self.camera.clip_near)
            self.edge_shader.uniform1f("zmax", self.camera.clip_far)
            self.ndc_screen_va.bind()
            glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.ndc_screen_va.unbind()
            self.edge_shader.unbind()

    def render_depth_masks(self, vol_size):
        if self.box_va_shape != vol_size:
            self.box_va_shape = vol_size
            render_pixel_size = PIXEL_SCALE / vol_size[1]
            w = vol_size[1] / 2 * render_pixel_size
            h = vol_size[2] / 2 * render_pixel_size
            d = vol_size[0] / 2 * render_pixel_size
            vertices = [-w, h, d,
                        w, h, d,
                        w, -h, d,
                        -w, -h, d,
                        -w, h, -d,
                        w, h, -d,
                        w, -h, -d,
                        -w, -h, -d]
            indices = [0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 0, 4, 7, 7, 3, 0, 5, 1, 2, 2, 6, 5, 4, 0, 1, 1, 5, 4, 3, 7, 6, 6, 2, 3]
            self.box_va.update(VertexBuffer(vertices), IndexBuffer(indices))


        # read scene depth
        self.scene_fbo_b.bind()
        data = glReadPixels(0, 0, self.image_size, self.image_size, GL_DEPTH_COMPONENT, GL_FLOAT)
        scene_depth = np.frombuffer(data, dtype=np.float32).reshape(self.image_size, self.image_size)

        # render the provisional stop depth
        self.depth_fbo_a.bind()
        glClearDepth(0.0)
        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GREATER)
        glDepthMask(GL_TRUE)

        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", self.camera.matrix)
        self.box_va.bind()
        glDrawElements(GL_TRIANGLES, self.box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.box_va.unbind()
        self.depth_mask_shader.unbind()

        # find the actual stop depth: minimum(scene_depth, stop_depth) and write to fbo a depth texture.
        self.depth_fbo_a.bind()
        data = glReadPixels(0, 0, self.image_size, self.image_size, GL_DEPTH_COMPONENT, GL_FLOAT)
        stop_depth = np.frombuffer(data, dtype=np.float32).reshape(self.image_size, self.image_size)
        stop_depth = np.minimum(scene_depth, stop_depth)
        glBindTexture(GL_TEXTURE_2D, self.depth_fbo_a.depth_texture_renderer_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.image_size, self.image_size, 0, GL_DEPTH_COMPONENT, GL_FLOAT, stop_depth.flatten())
        self.depth_fbo_a.unbind((0, 0, self.image_size, self.image_size))

        # render the start depth
        self.depth_fbo_b.bind()
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_CULL_FACE)
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)


        self.depth_mask_shader.bind()
        self.depth_mask_shader.uniformmat4("vpMat", self.camera.matrix)
        self.box_va.bind()
        glDrawElements(GL_TRIANGLES, self.box_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
        self.box_va.unbind()

        self.depth_mask_shader.unbind()
        glFinish()

        self.depth_fbo_b.unbind((0, 0, self.image_size, self.image_size))
        glDepthFunc(GL_LESS)
        glClearDepth(1.0)

    def new_image(self):
        self.scene_fbo.bind()
        glClearColor(*self.background_colour)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.scene_fbo_b.bind()
        glClearColor(*self.background_colour)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.volume_fbo.bind()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def get_image(self):
        self.scene_fbo.bind()
        glBindTexture(GL_TEXTURE_2D, self.scene_fbo.texture.renderer_id)
        data = glReadPixels(0, 0, self.image_size, self.image_size, GL_RGBA, GL_FLOAT)
        image = np.frombuffer(data, dtype=np.float32).reshape((self.image_size, self.image_size, 4))
        image = np.flip(image, axis=0)[:, :, :3] * 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image


class VolumeModel:
    def __init__(self, data, feature_definition, pixel_size):
        self.data = data
        self.data = gaussian_filter1d(data, sigma=3.0, axis=0)
        self.data[0, :, :] = 0
        self.data[:, 0, :] = 0
        self.data[:, :, 0] = 0
        self.data[-1, :, :] = 0
        self.data[:, -1, :] = 0
        self.data[:, :, -1] = 0
        self.title = feature_definition.title
        self.colour = feature_definition.colour
        self.pixel_size = pixel_size

    def delete(self):
        pass


class SurfaceModel:
    def __init__(self, data, feature_definition, pixel_size):
        self.data = data
        if self.data.dtype == np.float32:
            self.data = gaussian_filter1d(self.data, sigma=3.0, axis=0)
        self.data[0, :, :] = 0
        self.data[-1, :, :] = 0
        self.data[:, 0, :] = 0
        self.data[:, -1, :] = 0
        self.data[:, :, 0] = 0
        self.data[:, :, -1] = 0
        z_margin = int(self.data.shape[0] * project_configuration["z_margin_summary"])

        self.data[:z_margin, :, :] = 0
        self.data[-z_margin:, :, :] = 0

        self.colour = feature_definition.colour
        self.level = feature_definition.level

        if self.data.dtype == np.float32:
            self.level /= 255.0

        self.dust = feature_definition.dust
        self.alpha = feature_definition.render_alpha

        self.render_pixel_size = PIXEL_SCALE / self.data.shape[1]
        self.true_pixel_size = pixel_size / 10.0
        if self.true_pixel_size < 0.1:
            self.dust = 0.0
        self.blobs = dict()

        self.generate_model()

    def hide_dust(self):
        for i in self.blobs:
            self.blobs[i].hide = self.blobs[i].volume < self.dust

    def generate_model(self):
        data = self.data
        origin = 0.5 * np.array(self.data.shape) * self.render_pixel_size
        new_blobs = dict()

        labels, N = label(data >= self.level)
        Z, Y, X = np.nonzero(labels)
        for i in range(len(Z)):
            z = Z[i]
            y = Y[i]
            x = X[i]
            l = labels[z, y, x]
            if l not in new_blobs:
                new_blobs[l] = SurfaceModelBlob(data, self.level, self.render_pixel_size, origin, self.true_pixel_size)
            new_blobs[l].x.append(x)
            new_blobs[l].y.append(y)
            new_blobs[l].z.append(z)

        # 3: upload surface blobs one by one.
        for i in new_blobs:
            try:
                new_blobs[i].compute_mesh()
            except Exception:
                pass

        for i in self.blobs:
            self.blobs[i].delete()
        self.blobs = new_blobs
        self.hide_dust()

    def delete(self):
        for i in self.blobs:
            self.blobs[i].delete()


class SurfaceModelBlob:
    def __init__(self, data, level, render_pixel_size, origin, true_pixel_size=1.0):
        self.data = data
        self.level = level
        self.render_pixel_size = render_pixel_size
        self.true_pixel_size = true_pixel_size
        self.origin = origin
        self.x = list()
        self.y = list()
        self.z = list()
        self.volume = 0
        self.indices = list()
        self.vertices = list()
        self.normals = list()
        self.vao_data = list()
        self.va = VertexArray(attribute_format="xyznxnynz")
        self.va_requires_update = False
        self.complete = False
        self.hide = False

    def compute_mesh(self):
        self.volume = len(self.x) * self.true_pixel_size**3
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)

        rx = (np.amin(self.x), np.amax(self.x)+2)
        ry = (np.amin(self.y), np.amax(self.y)+2)
        rz = (np.amin(self.z), np.amax(self.z)+2)
        box = np.zeros((1 + rz[1]-rz[0] + 1, 1 + ry[1]-ry[0] + 1, 1 + rx[1]-rx[0] + 1))
        box[1:-1, 1:-1, 1:-1] = self.data[rz[0]:rz[1], ry[0]:ry[1], rx[0]:rx[1]]
        mask = np.zeros((1 + rz[1]-rz[0] + 1, 1 + ry[1]-ry[0] + 1, 1 + rx[1]-rx[0] + 1), dtype=bool)

        mx = self.x - rx[0] + 1
        my = self.y - ry[0] + 1
        mz = self.z - rz[0] + 1
        for x, y, z in zip(mx, my, mz):
            mask[z, y, x] = True
        mask = binary_dilation(mask, iterations=2)
        box *= mask
        vertices, faces, normals, _ = measure.marching_cubes(box, level=self.level)
        vertices += np.array([rz[0], ry[0], rx[0]])
        self.vertices = vertices[:, [2, 1, 0]]
        self.normals = normals[:, [2, 1, 0]]

        self.vertices *= self.render_pixel_size
        self.vertices -= np.array([self.origin[2], self.origin[1], self.origin[0]])
        self.vao_data = np.hstack((self.vertices, self.normals)).flatten()
        self.indices = faces.flatten()
        self.va_requires_update = True
        self.va.update(VertexBuffer(self.vao_data), IndexBuffer(self.indices, long=True))
        self.va_requires_update = False
        self.complete = True


    def delete(self):
        if self.va.initialized:
            if glIsBuffer(self.va.vertexBuffer.vertexBufferObject):
                glDeleteBuffers(1, [self.va.vertexBuffer.vertexBufferObject])
            if glIsBuffer(self.va.indexBuffer.indexBufferObject):
                glDeleteBuffers(1, [self.va.indexBuffer.indexBufferObject])
            if glIsVertexArray(self.va.vertexArrayObject):
                glDeleteVertexArrays(1, [self.va.vertexArrayObject])
            self.va.initialized = False


class Light3D:
    def __init__(self):
        self.colour = (1.0, 1.0, 1.0)
        self.vec = (0.0, 1.0, 0.0)
        self.yaw = 20.0
        self.pitch = 0.0
        self.strength = 0.5

    def compute_vec(self, dyaw=0, dpitch=0):
        # Calculate the camera forward vector based on pitch and yaw
        cos_pitch = np.cos(np.radians(self.pitch + dpitch))
        sin_pitch = np.sin(np.radians(self.pitch + dpitch))
        cos_yaw = np.cos(np.radians(self.yaw + dyaw))
        sin_yaw = np.sin(np.radians(self.yaw + dyaw))

        forward = np.array([-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw])
        self.vec = forward


class Camera3D:
    def __init__(self, image_size):
        self.view_matrix = np.eye(4)
        self.projection_matrix = np.eye(4)
        self.view_projection_matrix = np.eye(4)
        self.focus = np.zeros(3)
        self.pitch = project_configuration["camera_pitch"]
        self.yaw = project_configuration["camera_yaw"]
        self.distance = 1120.0
        self.clip_near = 1e2
        self.clip_far = 1e4
        self.projection_width = 1
        self.projection_height = 1
        self.set_projection_matrix(image_size, image_size)

    def set_projection_matrix(self, window_width, window_height):
        self.projection_width = window_width
        self.projection_height = window_height
        self.update_projection_matrix()

    def cursor_delta_to_world_delta(self, cursor_delta):
        self.yaw *= -1
        camera_right = np.cross([0, 1, 0], self.get_forward())
        camera_up = np.cross(camera_right, self.get_forward())
        self.yaw *= -1
        return cursor_delta[0] * camera_right + cursor_delta[1] * camera_up

    def get_forward(self):
        # Calculate the camera forward vector based on pitch and yaw
        cos_pitch = np.cos(np.radians(self.pitch))
        sin_pitch = np.sin(np.radians(self.pitch))
        cos_yaw = np.cos(np.radians(self.yaw))
        sin_yaw = np.sin(np.radians(self.yaw))

        forward = np.array([-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw])
        return forward

    @property
    def matrix(self):
        return self.view_projection_matrix

    @property
    def vpmat(self):
        return self.view_projection_matrix

    @property
    def ivpmat(self):
        return np.linalg.inv(self.view_projection_matrix)

    @property
    def pmat(self):
        return self.projection_matrix

    @property
    def vmat(self):
        return self.view_matrix

    @property
    def ipmat(self):
        return np.linalg.inv(self.projection_matrix)

    @property
    def ivmat(self):
        return np.linalg.inv(self.view_matrix)

    def on_update(self):
        self.update_projection_matrix()
        self.update_view_projection_matrix()

    def update_projection_matrix(self):
        aspect_ratio = self.projection_width / self.projection_height
        self.projection_matrix = Camera3D.create_perspective_matrix(60.0, aspect_ratio, self.clip_near, self.clip_far)
        self.update_view_projection_matrix()

    @staticmethod
    def create_perspective_matrix(fov, aspect_ratio, near, far):
        S = 1 / (np.tan(0.5 * fov / 180.0 * np.pi))
        f = far
        n = near

        projection_matrix = np.zeros((4, 4))
        projection_matrix[0, 0] = S / aspect_ratio
        projection_matrix[1, 1] = S
        projection_matrix[2, 2] = -f / (f - n)
        projection_matrix[3, 2] = -1
        projection_matrix[2, 3] = -f * n / (f - n)

        return projection_matrix

    def update_view_projection_matrix(self):
        eye_position = self.calculate_relative_position(self.focus, self.pitch, self.yaw, self.distance)
        self.view_matrix = self.create_look_at_matrix(eye_position, self.focus)
        self.view_projection_matrix = self.projection_matrix @ self.view_matrix

    def get_view_direction(self):
        eye_position = self.calculate_relative_position(self.focus, self.pitch, self.yaw, self.distance)
        focus_position = np.array(self.focus)
        view_dir = eye_position - focus_position
        view_dir /= np.sum(view_dir**2)**0.5
        return view_dir

    @staticmethod
    def calculate_relative_position(base_position, pitch, yaw, distance):
        cos_pitch = np.cos(np.radians(pitch))
        sin_pitch = np.sin(np.radians(pitch))
        cos_yaw = np.cos(np.radians(yaw))
        sin_yaw = np.sin(np.radians(yaw))

        forward = np.array([
            cos_pitch * sin_yaw,
            sin_pitch,
            -cos_pitch * cos_yaw
        ])
        forward = forward / np.linalg.norm(forward)

        relative_position = base_position + forward * distance

        return relative_position

    @staticmethod
    def create_look_at_matrix(eye, position):
        forward = Camera3D.normalize(position - eye)
        right = Camera3D.normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)

        look_at_matrix = np.eye(4)
        look_at_matrix[0, :3] = right
        look_at_matrix[1, :3] = up
        look_at_matrix[2, :3] = -forward
        look_at_matrix[:3, 3] = -np.dot(look_at_matrix[:3, :3], eye)
        return look_at_matrix

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

