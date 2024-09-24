from Pom.core.opengl_classes import *

import json
import glfw
from skimage import measure
from scipy.ndimage import label, binary_dilation, gaussian_filter
import Pom.core.config as cfg


def parse_feature_library(feature_library_path):
    with open(feature_library_path, 'r') as f:
        flist = json.load(f)
    feature_library = dict()
    for f in flist:
        feature_definition = FeatureLibraryFeature.from_dict(f)
        feature_library[feature_definition.title] = feature_definition
    return feature_library


class FeatureLibraryFeature:
    DEFAULT_COLOURS = [(66 / 255, 214 / 255, 164 / 255),
                       (255 / 255, 243 / 255, 0 / 255),
                       (255 / 255, 104 / 255, 0 / 255),
                       (255 / 255, 13 / 255, 0 / 255),
                       (174 / 255, 0 / 255, 255 / 255),
                       (21 / 255, 0 / 255, 255 / 255),
                       (0 / 255, 136 / 255, 266 / 255),
                       (0 / 255, 247 / 255, 255 / 255),
                       (0 / 255, 255 / 255, 0 / 255)]
    CLR_COUNTER = 0
    SORT_TITLE = "\n"

    def __init__(self):
        self.title = "New feature"
        self.colour = FeatureLibraryFeature.DEFAULT_COLOURS[FeatureLibraryFeature.CLR_COUNTER % len(FeatureLibraryFeature.DEFAULT_COLOURS)]
        self.box_size = 64
        self.brush_size = 10.0 # nm
        self.alpha = 1.0
        self.use = True
        self.dust = 1.0
        self.level = 128
        self.render_alpha = 1.0
        self.hide = False
        FeatureLibraryFeature.CLR_COUNTER += 1

    @property
    def rgb(self):
        r = (self.colour[0] * 255)
        g = (self.colour[1] * 255)
        b = (self.colour[2] * 255)
        return (r, g, b)

    def to_dict(self):
        return vars(self)

    @staticmethod
    def from_dict(data):
        ret = FeatureLibraryFeature()
        ret.title = data['title']
        ret.colour = data['colour']
        ret.box_size = data['box_size']
        ret.brush_size = data['brush_size']
        ret.alpha = data['alpha']
        ret.use = data['use']
        ret.dust = data['dust']
        ret.level = data['level']
        ret.render_alpha = data['render_alpha']
        ret.hide = data['hide']
        return ret


class Renderer:
    def __init__(self, style=0, image_size=512):
        if not glfw.init():
            raise Exception("GLFW initialization failed.")

        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(10, 10, "Offscreen Render", None, None)
        glfw.make_context_current(self.window)
        glEnable(GL_MULTISAMPLE)

        self.surface_model_shader = Shader(os.path.join(cfg.root, "shaders", "se_surface_model_shader.glsl"))
        self.edge_shader = Shader(os.path.join(cfg.root, "shaders", "se_depth_edge_detect.glsl"))
        self.style = 0

        self.image_size = image_size
        self.scene_fbo = FrameBuffer(width=self.image_size, height=self.image_size, texture_format="rgba32f")
        self.img_fbo = FrameBuffer(width=self.image_size, height=self.image_size, texture_format="rgba32f")

        self.ndc_screen_va = VertexArray(attribute_format="xy")
        self.ndc_screen_va.update(VertexBuffer([-1, -1, 1, -1, 1, 1, -1, 1]), IndexBuffer([0, 1, 2, 0, 2, 3]))

        # TODO: make it possible to use style = 0, or 1, or 2, for different styles.
        self.RENDER_SILHOUETTES = True
        self.RENDER_SILHOUETTES_ALPHA = 0.5
        self.RENDER_SILHOUETTES_THRESHOLD = 0.01

        self.camera = Camera3D(self.image_size)
        self.camera.on_update()

        self.light = Light3D()
        self.ambient_strength = 0.75
        self.background_colour = (1.0, 1.0, 1.0, 1.0)

    def delete(self):
        glfw.terminate()

    def render_surface_models(self, surface_models):
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

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if self.RENDER_SILHOUETTES and len(alpha_sorted_surface_models) > 0:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.scene_fbo.framebufferObject)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.img_fbo.framebufferObject)
            glBlitFramebuffer(0, 0, self.image_size, self.image_size, 0, 0, self.image_size, self.image_size, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
            glBindFramebuffer(GL_FRAMEBUFFER, self.scene_fbo.framebufferObject)
            self.edge_shader.bind()
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.img_fbo.depth_texture_renderer_id)
            self.edge_shader.uniform1f("threshold", self.RENDER_SILHOUETTES_THRESHOLD)
            self.edge_shader.uniform1f("edge_alpha", self.RENDER_SILHOUETTES_ALPHA)
            self.edge_shader.uniform1f("zmin", self.camera.clip_near)
            self.edge_shader.uniform1f("zmax", self.camera.clip_far)
            self.ndc_screen_va.bind()
            glDrawElements(GL_TRIANGLES, self.ndc_screen_va.indexBuffer.getCount(), GL_UNSIGNED_SHORT, None)
            self.ndc_screen_va.unbind()
            self.edge_shader.unbind()

    def new_image(self):
        self.scene_fbo.bind()
        glClearColor(*self.background_colour)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def get_image(self):
        glBindTexture(GL_TEXTURE_2D, self.scene_fbo.texture.renderer_id)
        data = glReadPixels(0, 0, self.image_size, self.image_size, GL_RGBA, GL_FLOAT)
        image = np.frombuffer(data, dtype=np.float32).reshape(self.image_size, self.image_size, 4)
        image = np.flip(image, axis=0)[:, :, :3] * 255
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image


class SurfaceModel:
    def __init__(self, data, feature_definition, pixel_size):
        self.data = data
        if self.data.dtype == np.float32:
            self.data = gaussian_filter(self.data, sigma=1)
        self.data[0, :, :] = 0
        self.data[-1, :, :] = 0
        self.data[:, 0, :] = 0
        self.data[:, -1, :] = 0
        self.data[:, :, 0] = 0
        self.data[:, :, -1] = 0
        n_slices = self.data.shape[0]
        self.data[:n_slices // 3, :, :] = 0
        self.data[-n_slices // 3:, :, :] = 0

        self.colour = feature_definition.colour
        self.level = feature_definition.level

        if self.data.dtype == np.float32:
            self.level /= 255.0

        self.dust = feature_definition.dust
        self.alpha = feature_definition.render_alpha

        self.render_pixel_size = 1000 / self.data.shape[1]
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
        self.pitch = -30.0
        self.yaw = 180.0
        self.distance = 1120.0
        self.clip_near = 1e-1
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

