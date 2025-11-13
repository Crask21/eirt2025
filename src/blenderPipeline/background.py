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
        return (x, y, 0.0)
    
    def EnableCameraUpdates(self, obj=None):
        """
        Enable camera updates for a 3DGS object. This makes the object update 
        its view automatically when the camera moves.

        Args:
            obj: Blender object to enable camera updates for. If None, uses active object.
        """
        if obj is None:
            obj = bpy.context.view_layer.objects.active

        if obj and 'KIRI_3DGS_Render_GN' in obj.modifiers:
            # Set the update mode to 'Enable Camera Updates'
            obj.sna_kiri3dgs_active_object_update_mode = 'Enable Camera Updates'
            print(f"Enabled camera updates for {obj.name}")
        else:
            print("Error: Object must have KIRI_3DGS_Render_GN modifier")

    def DisableCameraUpdates(self, obj=None):
        """
        Disable camera updates for a 3DGS object.

        Args:
            obj: Blender object to disable camera updates for. If None, uses active object.
        """
        if obj is None:
            obj = bpy.context.view_layer.objects.active

        if obj and 'KIRI_3DGS_Render_GN' in obj.modifiers:
            # Set the update mode to 'Disable Camera Updates'
            obj.sna_kiri3dgs_active_object_update_mode = 'Disable Camera Updates'
            print(f"Disabled camera updates for {obj.name}")
        else:
            print("Error: Object must have KIRI_3DGS_Render_GN modifier")

    def UpdateActiveToView(self, obj=None):
        """
        Update the active 3DGS object to match the current viewport view.
        This is equivalent to clicking 'Update Active To View' in the addon UI.

        Args:
            obj: Blender object to update. If None, uses active object.
        """
        if obj is None:
            obj = bpy.context.view_layer.objects.active

        if obj and 'KIRI_3DGS_Render_GN' in obj.modifiers:
            # Make the object active
            bpy.context.view_layer.objects.active = obj
            # Call the addon's update function
            bpy.ops.sna.dgs_render_align_active_to_view_30b13('EXEC_DEFAULT')
            print(f"Updated {obj.name} to current view")
        else:
            print("Error: Object must have KIRI_3DGS_Render_GN modifier")

    def UpdateAllCameraEnabledObjects(self):
        """
        Update all objects that have camera updates enabled to the current view.
        This updates all 3DGS objects at once.
        """
        # Call the addon's function that updates all enabled objects
        bpy.ops.sna.dgs_render_update_enabled_3dgs_objects_6d7f4(
            'EXEC_DEFAULT')
        print("Updated all camera-enabled 3DGS objects")
    


# # spawn a plane and make it the active object
# bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, location=(0.0, 0.0, 0.0))
# bg_obj = bpy.context.active_object
# background = Background(bg_obj, limits=(-4.0, 4.0, -4.0, 4.0))

# # Example of getting a random position within the background limits
# random_position = background.getRandomPosition()
# print(f"Random position on background: {random_position}")

# background = Background("E:\\datasets\\eirt_background\\dreamlab_lowres.ply", limits=(-3,3,-5.2,5.2))


