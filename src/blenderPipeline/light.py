import bpy

class Light:
    def __init__(self, light_type='AREA', energy=1000, location=(0, 0, 10), rotation=(0, 0, 0), radius=1.0):
        # Create a new light data block
        light_data = bpy.data.lights.new(name="lighting", type=light_type)
        light_data.energy = energy
        light_data.size = radius

        # Create a new light object
        self.light_object = bpy.data.objects.new(name="lighting", object_data=light_data)

        # Set the location and rotation of the light object
        self.light_object.location = location
        self.light_object.rotation_euler = rotation

        # Link the light object to the current collection
        bpy.context.collection.objects.link(self.light_object)


# light = Light(light_type='AREA', energy=15000, location=(0, 0, 20), rotation=(0, 0, 0), radius=50.0)  # Example usage