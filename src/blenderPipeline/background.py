import bpy
import random




class Background:
    def __init__(self, background: bpy.types.Object, limits: tuple[float, float, float, float]):
        self.obj = background
        self.name = background.name
        self.location = background.location
        self.rotation = background.rotation_euler
        self.scale = background.scale
        self.limits = limits # (xmin, xmax, ymin, ymax)
    
    def getRandomPosition(self) -> tuple[float, float, float]:
        xmin, xmax, ymin, ymax = self.limits
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        return (x, y)
    


# spawn a plane and make it the active object
bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, location=(0.0, 0.0, 0.0))
bg_obj = bpy.context.active_object
background = Background(bg_obj, limits=(-4.0, 4.0, -4.0, 4.0))

# Example of getting a random position within the background limits
random_position = background.getRandomPosition()
print(f"Random position on background: {random_position}")
