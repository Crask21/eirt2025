import bpy
import random




class Background:
    def __init__(self, backgroundPath: str, limits: tuple[tuple[float, float, float, float]]):
        if backgroundPath is not None:
            # ------------------------------------ PLY ----------------------------------- #
            # bpy.ops.sna.dgs_render_import_ply_bf139(filepath=backgroundPath)
            # self.background = bpy.context.active_object    
            # self.EnableCameraUpdates(self.background)

            # self.UpdateActiveToView(self.background)
            # ----------------------------------- USD ----------------------------------- #
            bpy.ops.wm.usd_import(filepath=backgroundPath)
        # self.name = background.name
        # self.location = background.location
        # self.rotation = background.rotation_euler
        # self.scale = background.scale
        self.limits = limits # (xmin, xmax, ymin, ymax) #(-3,3,-5.2,5.2)
        print(f"Background limits set to: {self.limits}")

    
    def getRandomPosition(self) -> tuple[float, float, float]:
        if type(self.limits[0]) is not tuple:
            xmin, xmax, ymin, ymax = self.limits
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            return (x, y)
        else:
            random_limits = self.limits[random.randint(0, len(self.limits)-1)]
            xmin, xmax, ymin, ymax = random_limits
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            return (x, y)
    
    def is_within_limits(self, x: float, y: float) -> bool:
        if type(self.limits[0]) is not tuple:
            xmin, xmax, ymin, ymax = self.limits
            return xmin <= x <= xmax and ymin <= y <= ymax
        for limits in self.limits:
            xmin, xmax, ymin, ymax = limits
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False
    


# # spawn a plane and make it the active object
# bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, location=(0.0, 0.0, 0.0))
# bg_obj = bpy.context.active_object
# background = Background(bg_obj, limits=(-4.0, 4.0, -4.0, 4.0))

# # Example of getting a random position within the background limits
# random_position = background.getRandomPosition()
# print(f"Random position on background: {random_position}")

# background = Background("F:\\datasets\\eirt_background\\background01.usdc", limits=(-4.0, 4.0, -4.0, 4.0))