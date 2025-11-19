import bpy
import random




class Background:
    def __init__(self, backgroundPath: str, limits: tuple[float, float, float, float]):
        if backgroundPath is not None:
            bpy.ops.wm.open_mainfile(filepath=backgroundPath)
            
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
# background = Background("F:\\datasets\\eirt_background\\background01.blend", limits=(-4.0, 4.0, -4.0, 4.0))
# # Example of getting a random position within the background limits
# random_position = background.getRandomPosition()
# print(f"Random position on background: {random_position}")
