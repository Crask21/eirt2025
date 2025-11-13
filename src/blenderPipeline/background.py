import bpy
import random




class Background:
    def __init__(self, backgroundPath: str, limits: tuple[float, float, float, float]):
        if backgroundPath is not None:
            bpy.ops.sna.dgs_render_import_ply_bf139(filepath=backgroundPath)
            self.background = bpy.context.active_object    
            self.EnableCameraUpdates(self.background)

            self.UpdateActiveToView(self.background)
        # self.name = background.name
        # self.location = background.location
        # self.rotation = background.rotation_euler
        # self.scale = background.scale
        self.limits = limits # (xmin, xmax, ymin, ymax) #(-3,3,-5.2,5.2)

    
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
