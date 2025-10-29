import bpy

class Object:
    def __init__(self, object: bpy.types.Object):
        self.name = object.name
        self.location = object.location
        self.rotation = object.rotation_euler
        self.scale = object.scale