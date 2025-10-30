class Batch:
    def __init__(self):
        self.objects = []
        self.background = None

    def addObject(self, obj):
        self.objects.append(obj)

        