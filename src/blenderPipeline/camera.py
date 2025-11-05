import bpy
class Camera:
    def __init__(self):
        bpy.ops.object.camera_add()
        self.camera = bpy.context.object
    def setPosition(self, position: tuple[float, float, float]):
        if self.camera:
            self.camera.location = position
    def setRotation(self, rotation: tuple[float, float, float]):
        if self.camera:
            self.camera.rotation_euler = rotation
    def setKeyframe(self, position: tuple[float, float, float], rotation: tuple[float, float, float], frame: int):
        self.setPosition(position)
        self.setRotation(rotation)
        self.camera.keyframe_insert(data_path="location", frame=frame)
        self.camera.keyframe_insert(data_path="rotation_euler", frame=frame)