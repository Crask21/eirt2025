from object import Object
from background import Background
from objectLoader import ObjectLoader
from backgroundLoader import BackgroundLoader


class Batch:
    def __init__(self, ):

        self.objectLoader = ObjectLoader(DatabasePath="path/to/database", debug=True)
        self.backgroundLoader = BackgroundLoader()


        self.objects = []
        self.background = None

    def addObject(self, obj):
        self.objects.append(obj)

