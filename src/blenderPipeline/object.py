import bpy


class Object:
    def __init__(self, obj: bpy.types.Object, class_id: int, class_name: str):
        self.obj = obj
        self.name = obj.name
        self.location = obj.location
        self.rotation = obj.rotation_euler
        self.scale = obj.scale
        self.class_id = class_id
        self.obj.pass_index = class_id
        self.obj["class_name"] = class_name

        # For resetting between batches:
        self.default_location = obj.location.copy()
        self.default_rotation = obj.rotation_euler.copy()
        self.default_scale = obj.scale.copy()
        self.setKeyframe(self.default_location, self.default_rotation, self.default_scale, frame=0)

    def setPosition(self, position: tuple[float, float, float]):
        self.obj.location = position

    def setRotation(self, rotation: tuple[float, float, float]):
        self.obj.rotation_euler = rotation

    def setScale(self, scale: tuple[float, float, float]):
        self.obj.scale = scale
    
    def clearPosition(self):
        self.obj.location = self.default_location
        self.obj.rotation_euler = self.default_rotation
        self.obj.scale = self.default_scale

    def setKeyframe(self, position: tuple[float, float, float], rotation: tuple[float, float, float], scale: tuple[float, float, float], frame: int):
        self.setPosition(position)
        self.setRotation(rotation)
        self.setScale(scale)
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        self.obj.keyframe_insert(data_path="scale", frame=frame)

        for fcurve in self.obj.animation_data.action.fcurves:
            fcurve.keyframe_points[-1].interpolation = 'CONSTANT'

    def setOnlyKeyframe(self, frame: int):
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        self.obj.keyframe_insert(data_path="scale", frame=frame)

        for fcurve in self.obj.animation_data.action.fcurves:
            fcurve.keyframe_points[-1].interpolation = 'CONSTANT'
    
    @property
    def obj_class(self):
        return self.obj.get("class_name", "Undefined")
    

# spawn a cube and make it the active object
bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0.0, 0.0, 0.0))
obj = bpy.context.active_object

obj = Object(obj, 1, "ExampleClass")
obj.setKeyframe((1, 2, 3), (0.5, 0.5, 0.5), (1, 1, 1), frame=10)
obj.setKeyframe((4, 5, 6), (1.0, 1.0, 1.0), (2, 2, 2), frame=20)

obj.clearPosition()
obj.setOnlyKeyframe(frame=30)




