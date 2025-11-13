import bpy
class Camera:
    def __init__(self):
        bpy.ops.object.camera_add()
        self.camera = bpy.context.object
        #TODO: set camera FOV unit to angle and set FOV to 39.6 degrees
        self.camera.data.lens_unit = 'FOV'
        self.FOV = 39.6
        self.camera.data.angle = self.FOV * (3.14159265 / 180.0)  # Convert degrees to radians
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