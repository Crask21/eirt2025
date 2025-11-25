import os
import sys
from os.path import dirname
sys.path.append(dirname(__file__))
from background import Background
from objectLoader import ObjectLoader
from backgroundLoader import BackgroundLoader
from object import Object
from camera import Camera
from light import Light
import random
import bpy
import math
import time
import csv


# This is only relevant on first run, on later reloads those modules
# are already in locals() and those statements do not do anything.

# import objectLoader
# print(f"[INFO] ObjectLoader file path: {objectLoader.__file__}")

objectsPath = "F:\\datasets\\eirt_objects"
backgroundPath = "F:\\datasets\\eirt_background\\background01.usdc"
savePath = "F:\\datasets\\eirt_output\\stationary_batch01"
enableCuda = True


class Batch:
    def __init__(self, objectsPerBatch: int = 1, objectsPerSample: int = 1, samples: int = 100, startFrame: int = 0): 
        # ----------------------------- sample parameters ---------------------------- #
        self.objectsPerBatch = objectsPerBatch
        if objectsPerSample > objectsPerBatch:
            self.objectsPerSample = objectsPerBatch
        else:
            self.objectsPerSample = objectsPerSample

        # -------------------------- loaders and scene setup ------------------------- #
        # self.background = Background(backgroundPath, limits=(-6, 6, -2.5, 2.5))
        self.background = Background(backgroundPath, limits=(-5.8, 5.8, -2.4, 2.4))
        self.objectLoader = ObjectLoader(DatabasePath=objectsPath, debug=False, includeGaussianSplatts=False)
        self.camera = Camera()
        self.light = Light(light_type='AREA', energy=15000, location=(0, 0, 20), rotation=(0, 0, 0), radius=50.0)

        # ----------------------------- spawn background ----------------------------- #

        # ------------------------------- spawn objects ------------------------------ #

        self.objects = []
        default_spawn_position = (0.0, 0.0, -10.0)
        total_dataset_objects = self.objectLoader.TotalObjects

        print(f"[INFO] Total objects in dataset: {total_dataset_objects}")
        for _ in range(self.objectsPerBatch):
            rand_index = random.randint(0, total_dataset_objects - 1)
            obj, class_name, class_id = self.objectLoader.CreateObject(rand_index)
            self.objects.append(Object(obj, class_id, class_name, spawn_position=default_spawn_position))


        # -------------------------- set scene and save path ------------------------- #
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU' if enableCuda else 'CPU'
        bpy.context.scene.render.resolution_x = 1080
        bpy.context.scene.render.resolution_y = 720
        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.frame_start = startFrame
        bpy.context.scene.frame_end = bpy.context.scene.frame_start + samples - 1

        tree = bpy.context.scene.node_tree
        for n in tree.nodes:
            tree.nodes.remove(n)

        # Render Layers node
        rl = tree.nodes.new('CompositorNodeRLayers')

        # # File Outputs
        composite_out = tree.nodes.new('CompositorNodeOutputFile')
        composite_out.base_path = os.path.join(savePath, "rgb")
        tree.links.new(rl.outputs['Image'], composite_out.inputs['Image'])

        # Enable depth pass
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        depth_out = tree.nodes.new('CompositorNodeOutputFile')
        # depth_out.format.color_depth = '16'

        depth_out.base_path = os.path.join(savePath, "depth")
        depth_out.format.file_format = "PNG"
        depth_out.format.color_depth = '16'
        tree.links.new(rl.outputs['Depth'], depth_out.inputs['Image'])

        # mask2_out = tree.nodes.new('CompositorNodeComposite')
        # tree.links.new(rl.outputs['IndexOB'], mask2_out.inputs['Image'])

        # bpy.context.scene.render.filepath = os.path.join(savePath, "mask2")
        # bpy.context.scene.render.image_settings.file_format = 'JPEG'

        mask_out = tree.nodes.new('CompositorNodeOutputFile')
        mask_out.base_path = os.path.join(savePath, "mask")
        # Connect RGB

        multiplier = tree.nodes.new('CompositorNodeMath')
        multiplier.operation = 'MULTIPLY'
        multiplier.inputs[1].default_value = 0.1  # Scale factor
        tree.links.new(rl.outputs['IndexOB'], multiplier.inputs[0])
        tree.links.new(multiplier.outputs['Value'], mask_out.inputs['Image'])

        multiplier2 = tree.nodes.new('CompositorNodeMath')
        multiplier2.operation = 'MULTIPLY'
        multiplier2.inputs[1].default_value = 0.05  # Scale factor
        tree.links.new(rl.outputs['Depth'], multiplier2.inputs[0])
        tree.links.new(multiplier2.outputs['Value'], depth_out.inputs['Image'])

        # Connect object index pass

        

        # ---------------------------------- example --------------------------------- #
        # obj, class_name, class_id = self.objectLoader.CreateObject(0, class_name="person")
        # self.objects.append(Object(obj, class_id, class_name, spawn
        # _position=default_spawn_position))
        # self.objects[0].setKeyframe(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), frame=10)
        # self.objects[0].setKeyframe(position=(2.0, 2.0, 0.0), rotation=(0.0, 0.0, 1.57), scale=(1.0, 1.0, 2.0), frame=20)
        # self.objects[0].clearPosition()
        # self.objects[0].setOnlyKeyframe(frame=30)

    def addObject(self, obj):
        self.objects.append(obj)

    def GenerateBatch(self):
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
            bpy.context.scene.frame_set(frame)
            self.GenerateSample()

    def GenerateSample(self):
        
        # Random camera position and rotation within background limits
        cam_x = random.uniform(self.background.limits[0], self.background.limits[1])
        cam_y = random.uniform(self.background.limits[2], self.background.limits[3])
        cam_z = 0.5  # Fixed height for simplicity
        cam_rot_x = 85/180 * 3.14159265  # Tilt down 80 degrees
        cam_rot_z = random.uniform(0, 2 * 3.14159265)  # Rotate around Z axis
        self.camera.setKeyframe(position=(cam_x, cam_y, cam_z), rotation=(cam_rot_x, 0.0, cam_rot_z), frame=bpy.context.scene.frame_current)
        
        randomObjs = random.sample(self.objects, self.objectsPerSample)

        # generate random transforms (r, alpha, theta) where (r,alpha) are polar coordinates in the XY plane originating from camera. Theta is rotation around Z axis.
        # Global (x,y) coordinates are constrained to be within map
        for obj in randomObjs:
            # TODO: ensure objects are within background limits and no collisions
            i = 0
            while True:
                r = random.uniform(1.5, 15.0)
                alpha = random.uniform(-self.camera.FOV/2, self.camera.FOV/2) * (3.14159265 / 180.0)  # Convert degrees to radians
                theta = random.uniform(0, 2*3.14159265)

                
                alpha_camera = alpha + self.camera.camera.rotation_euler[2]  # Adjust alpha based on camera rotation
                x = r * round(-math.sin(alpha_camera), 2) + self.camera.camera.location[0]
                y = r * round(math.cos(alpha_camera), 2) + self.camera.camera.location[1]
                z = 0.0  # Keep Z constant for simplicity
                # Check for collisions with other objects
                obj.setPosition((x, y, z))
                if (self.background.limits[0] <= x <= self.background.limits[1]) and (self.background.limits[2] <= y <= self.background.limits[3]) and not self.checkCollisions(obj):
                    obj.setKeyframe(position=(x, y, z), rotation=(0.0, 0.0, theta), scale=(1.0, 1.0, 1.0), frame=bpy.context.scene.frame_current)
                    break
                i += 1
                if i > 20:
                    print(f"[WARNING] Could not place object {obj.obj_class} without collisions after 20 attempts.")
                    obj.clearPosition()
                    break
        
        # Reset object positions for next frame
        for obj in randomObjs:
            obj.clearPosition()
            obj.setOnlyKeyframe(frame=bpy.context.scene.frame_current+1)


            # print(f"[DEBUG] alpha_camera: {alpha_camera}, camera rotation: {self.camera.camera.rotation_euler[2]}")
            # print(f"[DEBUG] camera pose: {self.camera.camera.location}, object polar coords (r, alpha): ({r}, {alpha}), object cartesian coords (x, y): ({x}, {y})")

    def GenerateStationarySceneSamples(self):
        
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)

        # generate random transforms (r, alpha, theta) where (r,alpha) are polar coordinates in the XY plane originating from camera. Theta is rotation around Z axis.
        # Global (x,y) coordinates are constrained to be within map
        for obj in self.objects:
            # TODO: ensure objects are within background limits and no collisions
            i = 0
            while True:
                x = random.uniform(self.background.limits[0], self.background.limits[1])
                y = random.uniform(self.background.limits[2], self.background.limits[3])
                z = 0.0  # Keep Z constant for simplicity
                theta = random.uniform(0, 2*3.14159265)
                # Check for collisions with other objects
                obj.setPosition((x, y, z))
                if not self.checkCollisions(obj):
                    obj.setKeyframe(position=(x, y, z), rotation=(0.0, 0.0, theta), scale=(1.0, 1.0, 1.0), frame=bpy.context.scene.frame_current)
                    break
                i += 1
                if i > 20:
                    print(f"[WARNING] Could not place object {obj.obj_class} without collisions after 20 attempts.")
                    obj.clearPosition()
                    break
        
        if not os.path.exists(savePath):
            os.makedirs(savePath)


        with open(os.path.join(savePath, "camera_positions.csv"), "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["frame", "cam_x", "cam_y", "cam_z", "cam_rot_x", "cam_rot_y", "cam_rot_z"])
            # set random camera and object positions for each frame while remaining colision free
            for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
                bpy.context.scene.frame_set(frame)

                while True:
                    # Random camera position and rotation within background limits
                    cam_x = random.uniform(self.background.limits[0], self.background.limits[1])
                    cam_y = random.uniform(self.background.limits[2], self.background.limits[3])
                    cam_z = 0.5  # Fixed height for simplicity
                    cam_rot_x = 85/180 * 3.14159265  # Tilt down 80 degrees
                    cam_rot_z = random.uniform(0, 2 * 3.14159265)  # Rotate around Z axis

                    collision = False
                    for obj2 in self.objects:
                        dist = ((cam_x - obj2.obj.location[0]) ** 2 + 
                            (cam_y - obj2.obj.location[1]) ** 2 + 
                            (cam_z - obj2.obj.location[2]) ** 2) ** 0.5
                        # print(f"[DEBUG] Checking collision between {obj1.obj_class} and {obj2.obj_class}: Distance = {dist}, Sum of Radii = {obj1.bounding_radius + obj2.bounding_radius}")
                        if dist < (1 + obj2.bounding_radius):
                            collision = True
                            break
                    if not collision:
                        break
                csvwriter.writerow([frame, cam_x, cam_y, cam_z, cam_rot_x, 0.0, cam_rot_z])
                self.camera.setKeyframe(position=(cam_x, cam_y, cam_z), rotation=(cam_rot_x, 0.0, cam_rot_z), frame=bpy.context.scene.frame_current)
        

    
    def checkCollisions(self, obj1: Object) -> bool:
        for obj2 in self.objects:
            if obj1 != obj2:
                dist = ((obj1.obj.location[0] - obj2.obj.location[0]) ** 2 + 
                    (obj1.obj.location[1] - obj2.obj.location[1]) ** 2 + 
                    (obj1.obj.location[2] - obj2.obj.location[2]) ** 2) ** 0.5
                # print(f"[DEBUG] Checking collision between {obj1.obj_class} and {obj2.obj_class}: Distance = {dist}, Sum of Radii = {obj1.bounding_radius + obj2.bounding_radius}")
                if dist < (obj1.bounding_radius + obj2.bounding_radius):
                    return True
        
        return False

batch = Batch(objectsPerBatch=7, objectsPerSample=5, samples=100, startFrame=0)
batch.GenerateStationarySceneSamples()
# batch.GenerateBatch()

# @bpy.app.handlers.persistent
# def on_scene_loaded(dummy):
#     print("[INFO] Scene loaded, initializing batch generation...")

# bpy.app.handlers.load_post.append(on_scene_loaded)
# bpy.ops.wm.open_mainfile(filepath=backgroundPath)
