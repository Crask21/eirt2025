import bpy
import random
from light import Light
from camera import Camera
from object import Object
from backgroundLoader import BackgroundLoader
from objectLoader import ObjectLoader
from background import Background
import sys
import os
from os.path import dirname
sys.path.append(dirname(__file__))


# import objectLoader
# print(f"[INFO] ObjectLoader file path: {objectLoader.__file__}")

objectsPath = 'F:\\datasets\\eirt_objects'
savePath = 'F:\\datasets\\eirt_output'


class Batch:
    def __init__(self, objectsPerBatch: int = 1, objectsPerSample: int = 1, samples: int = 100):

        # ----------------------------- sample parameters ---------------------------- #
        self.objectsPerBatch = objectsPerBatch
        if objectsPerSample > objectsPerBatch:
            self.objectsPerSample = objectsPerBatch
        else:
            self.objectsPerSample = objectsPerSample

        # -------------------------- loaders and scene setup ------------------------- #
        self.objectLoader = ObjectLoader(DatabasePath=objectsPath, debug=False)
        self.backgroundLoader = BackgroundLoader(
            DatabasePath=objectsPath, debug=False)
        self.camera = Camera()
        self.light = Light(light_type='AREA', energy=15000, location=(
            0, 0, 20), rotation=(0, 0, 0), radius=50.0)

        # ------------------------------- spawn objects ------------------------------ #
        self.objects = []
        default_spawn_position = (0.0, 0.0, -10.0)
        total_dataset_objects = self.objectLoader.TotalObjects

        for _ in range(self.objectsPerBatch):
            rand_index = random.randint(0, total_dataset_objects - 1)
            obj, class_name, class_id = self.objectLoader.CreateObject(
                rand_index)
            self.objects.append(
                Object(obj, class_id, class_name, spawn_position=default_spawn_position))

        # ----------------------------- spawn background ----------------------------- #
        self.background = None

        # -------------------------- set scene and save path ------------------------- #
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.use_nodes = True
        bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True
        bpy.context.scene.cycles.samples = 512
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = samples - 1

        tree = bpy.context.scene.node_tree
        for n in tree.nodes:
            tree.nodes.remove(n)

        # Render Layers node
        rl = tree.nodes.new('CompositorNodeRLayers')

        # File Outputs
        composite_out = tree.nodes.new('CompositorNodeOutputFile')
        composite_out.base_path = os.path.join(savePath, "rgb")

        mask_out = tree.nodes.new('CompositorNodeOutputFile')
        mask_out.base_path = os.path.join(savePath, "mask")

        # Connect RGB
        tree.links.new(rl.outputs['Image'], composite_out.inputs['Image'])
        # Connect object index pass
        tree.links.new(rl.outputs['IndexOB'], mask_out.inputs['Image'])

        # ---------------------------------- example --------------------------------- #
        # obj, class_name, class_id = self.objectLoader.CreateObject(0, class_name="person")
        # self.objects.append(Object(obj, class_id, class_name, spawn_position=default_spawn_position))
        # self.objects[0].setKeyframe(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), frame=10)
        # self.objects[0].setKeyframe(position=(2.0, 2.0, 0.0), rotation=(0.0, 0.0, 1.57), scale=(1.0, 1.0, 2.0), frame=20)
        # self.objects[0].clearPosition()
        # self.objects[0].setOnlyKeyframe(frame=30)

    def addObject(self, obj):
        self.objects.append(obj)


batch = Batch(objectsPerBatch=5)
