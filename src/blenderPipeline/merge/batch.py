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
<<<<<<< HEAD
from print_utils import print_green, print_success, print_error, print_warning, print_info


=======
import csv
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784


# This is only relevant on first run, on later reloads those modules
# are already in locals() and those statements do not do anything.

# import objectLoader
# print(f"[INFO] ObjectLoader file path: {objectLoader.__file__}")

<<<<<<< HEAD
objectsPath = "objects"
backgroundPath = None
savePath = "C:\\Users\\andpo\\Documents\\EIRT\\eirt2025\\src\\blenderPipeline\\output"
=======
objectsPath = "F:\\datasets\\eirt_objects"
backgroundPath = "F:\\datasets\\eirt_background\\background01.usdc"
savePath = "F:\\datasets\\eirt_output\\stationary_batch01"
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784
enableCuda = True


class Batch:
    def __init__(self, objectsPerBatch: int = 1, objectsPerSample: int = 1, samples: int = 100, startFrame: int = 0): 
        # ---------------------- clean default blender scene --------------------- #
        print_info("Cleaning default Blender objects (Cube, Camera, Light)...")
        
        # Delete all default objects in the scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)
        
        # Also clean up orphaned data (meshes, materials, etc.)
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        
        for block in bpy.data.lights:
            if block.users == 0:
                bpy.data.lights.remove(block)
        
        for block in bpy.data.cameras:
            if block.users == 0:
                bpy.data.cameras.remove(block)
        
        print_success("Scene cleaned - all default objects removed")
        
        # ------------------------------ Render settings ----------------------------- #
        bpy.data.scenes["Scene"].render.engine = 'CYCLES'
        bpy.data.scenes["Scene"].cycles.device = 'GPU' if enableCuda else 'CPU'
        # ----------------------------- sample parameters ---------------------------- #
        self.objectsPerBatch = objectsPerBatch
        if objectsPerSample > objectsPerBatch:
            self.objectsPerSample = objectsPerBatch
        else:
            self.objectsPerSample = objectsPerSample

        # -------------------------- loaders and scene setup ------------------------- #
<<<<<<< HEAD
        self.background = Background(backgroundPath, limits=(-6, 6, -2.5, 2.5))
        self.objectLoader = ObjectLoader(DatabasePath=objectsPath, debug=False, includeGaussianSplatts=True)
=======
        # self.background = Background(backgroundPath, limits=(-6, 6, -2.5, 2.5))
        self.background = Background(backgroundPath, limits=(-5.8, 5.8, -2.4, 2.4))
        self.objectLoader = ObjectLoader(DatabasePath=objectsPath, debug=False, includeGaussianSplatts=False)
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784
        self.camera = Camera()
        self.light = Light(light_type='AREA', energy=15000, location=(0, 0, 20), rotation=(0, 0, 0), radius=50.0)

        # ----------------------------- spawn background ----------------------------- #

        # ------------------------------- spawn objects ------------------------------ #

        self.objects = []
        default_spawn_position = (0.0, 0.0, -10.0)
        total_dataset_objects = self.objectLoader.TotalObjects

        print_green(f"Total objects in dataset: {total_dataset_objects}")
        for _ in range(self.objectsPerBatch):
            rand_index = random.randint(0, total_dataset_objects - 1)
            obj, class_name, class_id = self.objectLoader.CreateObject(rand_index)
            self.objects.append(Object(obj, class_id, class_name, spawn_position=default_spawn_position))


        # -------------------------- set scene and save path ------------------------- #
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU' if enableCuda else 'CPU'
<<<<<<< HEAD
        # Enable denoising on GPU
        bpy.data.scenes["Scene"].cycles.denoising_use_gpu = True
        bpy.data.scenes["Scene"].cycles.use_fast_gi = True
        bpy.data.scenes["Scene"].render.compositor_device = 'GPU'

=======
        bpy.context.scene.render.resolution_x = 1080
        bpy.context.scene.render.resolution_y = 720
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784
        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.frame_start = startFrame
        bpy.context.scene.frame_end = bpy.context.scene.frame_start + samples - 1

        tree = bpy.context.scene.node_tree
        
        # Clear nodes safely - iterate in reverse to avoid index issues
        # Use while loop to avoid holding references during iteration
        while tree.nodes:
            tree.nodes.remove(tree.nodes[0])

        # Render Layers node
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = (0, 0)

        # # File Outputs
        composite_out = tree.nodes.new('CompositorNodeOutputFile')
        composite_out.base_path = os.path.join(savePath, "rgb")
<<<<<<< HEAD
        composite_out.location = (400, 200)
=======
        tree.links.new(rl.outputs['Image'], composite_out.inputs['Image'])
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784

        # Enable depth pass
        bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
        depth_out = tree.nodes.new('CompositorNodeOutputFile')
        # depth_out.format.color_depth = '16'

        depth_out.base_path = os.path.join(savePath, "depth")
<<<<<<< HEAD
        depth_out.location = (600, 0)
        tree.links.new(rl.outputs['Depth'], depth_out.inputs['Image'])

        mask2_out = tree.nodes.new('CompositorNodeComposite')
        mask2_out.location = (400, -400)
=======
        depth_out.format.file_format = "PNG"
        depth_out.format.color_depth = '16'
        tree.links.new(rl.outputs['Depth'], depth_out.inputs['Image'])

        # mask2_out = tree.nodes.new('CompositorNodeComposite')
        # tree.links.new(rl.outputs['IndexOB'], mask2_out.inputs['Image'])
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784

        # bpy.context.scene.render.filepath = os.path.join(savePath, "mask2")
        # bpy.context.scene.render.image_settings.file_format = 'JPEG'


        
        # Connect RGB

# ---------------------- Mask pipeline with Dilate/Erode --------------------- #
        # Add Dilate node for mask output
        dilate_mask = tree.nodes.new('CompositorNodeDilateErode')
        dilate_mask.mode = 'STEP'
        dilate_mask.distance = 10  # Positive for dilate
        dilate_mask.location = (400, -200)

        # Add Erode node for mask output
        erode_mask = tree.nodes.new('CompositorNodeDilateErode')
        erode_mask.mode = 'STEP'
        erode_mask.distance = -10  # Negative for erode
        erode_mask.location = (600, -200)

        # Multiply node for mask output
        multiplier = tree.nodes.new('CompositorNodeMath')
        multiplier.operation = 'MULTIPLY'
        multiplier.inputs[1].default_value = 0.1  # Scale factor
        multiplier.location = (800, -200)

        # Mask output node
        mask_out = tree.nodes.new('CompositorNodeOutputFile')
        mask_out.base_path = os.path.join(savePath, "mask")
        mask_out.location = (1000, -200)

        # Link nodes
        tree.links.new(rl.outputs['IndexOB'], dilate_mask.inputs['Mask'])
        tree.links.new(dilate_mask.outputs['Mask'], erode_mask.inputs['Mask'])
        tree.links.new(erode_mask.outputs['Mask'], multiplier.inputs[0])
        tree.links.new(multiplier.outputs['Value'], mask_out.inputs['Image'])

        # Add viewer node for debugging
        viewer_node = tree.nodes.new('CompositorNodeViewer')
        viewer_node.location = (400, 300)
        tree.links.new(rl.outputs['Image'], viewer_node.inputs['Image'])

        # Depth pipeline
        multiplier2 = tree.nodes.new('CompositorNodeMath')
        multiplier2.operation = 'MULTIPLY'
        multiplier2.inputs[1].default_value = 0.05  # Scale factor
        multiplier2.location = (400, 0)
        tree.links.new(rl.outputs['Depth'], multiplier2.inputs[0])
        tree.links.new(multiplier2.outputs['Value'], depth_out.inputs['Image'])

        # Connect object index pass

<<<<<<< HEAD
        # ---------------------------- GenerateSample test --------------------------- #
        print_green(f"Starting batch generation for frames {bpy.context.scene.frame_start} to {bpy.context.scene.frame_end}")
        
        # Process frames with progress tracking
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
            try:
                # Update scene frame
                bpy.context.scene.frame_set(frame)
                
                # Generate sample for this frame
                self.GenerateSample()
                
                # Print progress every 10 frames
                if frame % 10 == 0:
                    print(f"[INFO] Processed frame {frame}/{bpy.context.scene.frame_end}")
                    
            except Exception as e:
                print_error(f"Failed to generate sample for frame {frame}: {e}")
                # Continue to next frame instead of crashing
                continue
        
        print_success("Batch generation completed.") 
=======
        

>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784
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
    
    def update_3dgs_camera_views(self):
        """
        Update all 3DGS objects to match the current camera view.
        This is critical for proper compositing of Gaussian Splat renders.
        Ensures all 3DGS objects are configured before updating.
        """
        try:
            # First, ensure all 3DGS objects are properly configured
            has_3dgs = False
            for obj in self.objects:
                if hasattr(obj, 'is_3dgs') and obj.is_3dgs:
                    has_3dgs = True
                    # Lazy configuration - only configures once
                    obj.ensure_3dgs_configured()
            
            if has_3dgs:
                # Now update camera views for all configured 3DGS objects
                bpy.ops.sna.dgs_render_update_enabled_3dgs_objects_6d7f4('EXEC_DEFAULT')
                # print("[DEBUG] Updated 3DGS camera views")
        except Exception as e:
            print_warning(f"Could not update 3DGS camera views: {e}")

<<<<<<< HEAD
    def GenerateSample(self, placement_attempts = 50):
=======
    def GenerateBatch(self):
        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
            bpy.context.scene.frame_set(frame)
            self.GenerateSample()

    def GenerateSample(self):
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784
        
        # Random camera position and rotation within background limits
        cam_x = random.uniform(self.background.limits[0], self.background.limits[1])
        cam_y = random.uniform(self.background.limits[2], self.background.limits[3])
        cam_z = 0.5  # Fixed height for simplicity
        cam_rot_x = 85/180 * 3.14159265  # Tilt down 80 degrees
        cam_rot_z = random.uniform(0, 2 * 3.14159265)  # Rotate around Z axis
        self.camera.setKeyframe(position=(cam_x, cam_y, cam_z), rotation=(cam_rot_x, 0.0, cam_rot_z), frame=bpy.context.scene.frame_current)
        
        # IMPORTANT: Update 3DGS objects after camera move for proper compositing
        # self.update_3dgs_camera_views()
        
        # Create a copy of the list to avoid issues during iteration
        randomObjs = random.sample(self.objects[:], self.objectsPerSample)

        # generate random transforms (r, alpha, theta) where (r,alpha) are polar coordinates in the XY plane originating from camera. Theta is rotation around Z axis.
        # Global (x,y) coordinates are constrained to be within map
        # Use index-based iteration to avoid holding object references during operations
        for obj_idx in range(len(randomObjs)):
            # Re-fetch object reference at start of each iteration
            obj = randomObjs[obj_idx]
            
            # TODO: ensure objects are within background limits and no collisions
            i = 0
            placed = False
            while i < placement_attempts:
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
                    placed = True
                    break
                i += 1
            
            if not placed:
                print_warning(f"Could not place object {obj.obj_class} without collisions after {placement_attempts} attempts.")
                obj.clearPosition()
        
        # Reset object positions for next frame
        # CRITICAL: Use indices instead of direct object references to avoid crashes
        # when Blender re-allocates memory during operations
        for i in range(len(randomObjs)):
            try:
                # Re-fetch object each time to avoid stale references
                obj = randomObjs[i]
                obj.clearPosition()
                obj.setOnlyKeyframe(frame=bpy.context.scene.frame_current+1)
            except (ReferenceError, AttributeError) as e:
                print_warning(f"Could not reset object {i}: {e}")
                continue


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
        """Check if obj1 collides with any other object in the scene.
        
        Uses safe iteration and null checks to avoid Blender crashes.
        CRITICAL: Does not hold references to Blender data during iteration.
        """
        # Use indices instead of iterating over object references
        for i in range(len(self.objects)):
            try:
                obj2 = self.objects[i]
                if obj1 != obj2:
                    # Copy location values immediately to avoid accessing during operations
                    # This prevents crashes from stale references
                    loc1 = (obj1.obj.location[0], obj1.obj.location[1], obj1.obj.location[2])
                    loc2 = (obj2.obj.location[0], obj2.obj.location[1], obj2.obj.location[2])
                    
                    dist = ((loc1[0] - loc2[0]) ** 2 + 
                           (loc1[1] - loc2[1]) ** 2 + 
                           (loc1[2] - loc2[2]) ** 2) ** 0.5
                    
                    if dist < (obj1.bounding_radius + obj2.bounding_radius):
                        return True
            except (AttributeError, ReferenceError, IndexError) as e:
                # Object may have been removed or invalidated
                print_warning(f"Cannot check collision - object reference may be invalid: {e}")
                continue
        
        return False

<<<<<<< HEAD
batch = Batch(objectsPerBatch=3, objectsPerSample=5, samples=10, startFrame=0)
=======
batch = Batch(objectsPerBatch=7, objectsPerSample=5, samples=100, startFrame=0)
batch.GenerateStationarySceneSamples()
# batch.GenerateBatch()
>>>>>>> 289faf6ac8c52500fb68092851b0d8ea8d0b7784

# @bpy.app.handlers.persistent
# def on_scene_loaded(dummy):
#     print("[INFO] Scene loaded, initializing batch generation...")

# bpy.app.handlers.load_post.append(on_scene_loaded)
# bpy.ops.wm.open_mainfile(filepath=backgroundPath)
