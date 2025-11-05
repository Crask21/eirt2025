
import sys
from os.path import dirname
sys.path.append(dirname(__file__))


from background import Background
from objectLoader import ObjectLoader
from backgroundLoader import BackgroundLoader
from object import Object
import random
# import objectLoader
# print(f"[INFO] ObjectLoader file path: {objectLoader.__file__}")

class Batch:
    def __init__(self):

        self.objectLoader = ObjectLoader(DatabasePath='E:\\datasets\\eirt_objects', debug=True)
        self.backgroundLoader = BackgroundLoader(DatabasePath='E:\\datasets\\eirt_objects', debug=True)

        default_spawn_position = (0.0, 0.0, -10.0)
        
        total_dataset_objects =  self.objectLoader.TotalObjects

        # obj, class_name, class_id = self.objectLoader.CreateObject(random.randint(0, total_dataset_objects - 1))
        obj, class_name, class_id = self.objectLoader.CreateObject(2)
        self.objects = []
        self.background = None
        
        self.objects.append(Object(obj, class_id, class_name, spawn_position=default_spawn_position))


        self.objects[0].setKeyframe(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), frame=10)
        self.objects[0].setKeyframe(position=(2.0, 2.0, 0.0), rotation=(0.0, 0.0, 1.57), scale=(1.0, 1.0, 1.0), frame=20)
        self.objects[0].clearPosition()
        self.objects[0].setOnlyKeyframe(frame=30)

    def addObject(self, obj):
        self.objects.append(obj)


batch = Batch()
