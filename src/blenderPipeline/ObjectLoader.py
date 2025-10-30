import bpy
import os
from pathlib import Path
bpy.ops.preferences.addon_enable(module="kiri_3dgs")

# ------------------------------- ObjectLoader ------------------------------- #


class ObjectLoader:
    def __init__(self, DatabasePath: str, debug: bool = False):
        self.allowed_extensions = [".obj", ".stl", ".ply"]
        self.DatabasePath = Path(DatabasePath)
        if debug:
            print(f"[INFO] DatabasePath set to: {self.DatabasePath}")
        self.Classes = self.read_classes()
        if debug:
            print(f"[INFO] Classes found: {self.Classes}")
        self.ObjectsByClass = self.organize_objects_by_class()
        self.TotalObjects = len(self.FindAllObjects())
        if debug:
            print(f"[INFO] TotalObjects found: {self.TotalObjects}")
        self.AllObjects = self.FindAllObjects()

    def organize_objects_by_class(self) -> dict[str, list[Path]]:
        """Organize objects into a dictionary by their class (folder name).

        Returns:
            dict[str, list[Path]]: A dictionary mapping class names to lists of object file paths.
        """
        objects_by_class = {cls: [] for cls in self.Classes}
        for class_folder in self.DatabasePath.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                if class_name in objects_by_class:
                    # Add all allowed extensions files in this class folder to the list
                    for file in class_folder.iterdir():
                        # Ensure it's a file and has an allowed extension
                        if file.is_file() and file.suffix in self.allowed_extensions:
                            objects_by_class[class_name].append(file)
        return objects_by_class

    def read_classes(self) -> list[str]:
        """
        Read class names from the database path.
        Returns:
            list[str]: A list of class names.
        """
        # Each class is a folder in the database path
        classes = []
        for entry in os.listdir(self.DatabasePath):
            entry_path = self.DatabasePath / entry
            if entry_path.is_dir():
                class_name = entry_path.name[:]
                classes.append(class_name)
        return classes

    def FindAllObjects(self) -> list[Path]:
        """
        Scan the database path for all object files.
        Returns:
            list[Path]: A list of paths to the object files.
        """
        object_list = []
        # List all files in the directory and child directories
        for root, dirs, files in os.walk(self.DatabasePath):
            for file in files:
                if file.endswith(tuple(self.allowed_extensions)):
                    # Append full file path
                    object_list.append(Path(root) / file)
        return object_list

    def CreateObject(self, ObjIndex=None, class_name: str = None):
        """
        Create an object from the database.
        Args:
            ObjIndex (int, optional): Index of the object to load. Defaults to None.
            class_name (str, optional): Class name to load the object from. Defaults to None
        """
        if class_name:
            objects = self.ObjectsByClass.get(class_name, [])
            if ObjIndex is not None and 0 <= ObjIndex < len(objects):
                obj_path = str(objects[ObjIndex])  # Convert Path to string
                # Try different import methods based on file extension
                if obj_path.lower().endswith('.stl'):
                    bpy.ops.wm.stl_import(filepath=obj_path)
                elif obj_path.lower().endswith('.obj'):
                    bpy.ops.wm.obj_import(filepath=obj_path)
                elif obj_path.lower().endswith('.ply'):
                    # Check if this is a 3DGS PLY file by trying the kiri_3dgs importer first
                    try:
                        bpy.ops.sna.dgs_render_import_ply_bf139(
                            'EXEC_DEFAULT', filepath=obj_path)
                    except:
                        # Fall back to regular PLY import if not a 3DGS file
                        bpy.ops.wm.ply_import(filepath=obj_path)

                return bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        elif class_name is None and ObjIndex is not None:
            if 0 <= ObjIndex < len(self.AllObjects):
                # Convert Path to string
                obj_path = str(self.AllObjects[ObjIndex])
                # Try different import methods based on file extension
                if obj_path.lower().endswith('.stl'):
                    bpy.ops.wm.stl_import(filepath=obj_path)
                elif obj_path.lower().endswith('.obj'):
                    bpy.ops.wm.obj_import(filepath=obj_path)
                elif obj_path.lower().endswith('.ply'):
                    # Check if this is a 3DGS PLY file by trying the kiri_3dgs importer first
                    try:
                        bpy.ops.sna.dgs_render_import_ply_bf139(
                            'EXEC_DEFAULT', filepath=obj_path)
                    except:
                        # Fall back to regular PLY import if not a 3DGS file
                        bpy.ops.wm.ply_import(filepath=obj_path)

                return bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        raise ValueError("Invalid ObjIndex or class_name")

    def CreateGaussianSplat(self, ObjIndex=None, class_name: str = None):
        """
        Create a 3D Gaussian Splat object from the database using the kiri_3dgs addon.
        This method specifically uses the 3DGS importer for PLY files.

        Args:
            ObjIndex (int, optional): Index of the object to load. Defaults to None.
            class_name (str, optional): Class name to load the object from. Defaults to None

        Returns:
            Object: The imported 3DGS object or None if failed
        """
        obj_path = None

        if class_name:
            objects = self.ObjectsByClass.get(class_name, [])
            if ObjIndex is not None and 0 <= ObjIndex < len(objects):
                obj_path = str(objects[ObjIndex])
        elif class_name is None and ObjIndex is not None:
            if 0 <= ObjIndex < len(self.AllObjects):
                obj_path = str(self.AllObjects[ObjIndex])

        if obj_path and obj_path.lower().endswith('.ply'):
            try:
                # Use the kiri_3dgs importer for 3DGS PLY files
                bpy.ops.sna.dgs_render_import_ply_bf139(
                    'EXEC_DEFAULT', filepath=obj_path)
                return bpy.context.selected_objects[0] if bpy.context.selected_objects else None
            except Exception as e:
                print(f"Error loading 3DGS file {obj_path}: {e}")
                return None
        else:
            raise ValueError("File must be a PLY file for 3DGS import")

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

# ---------------------------------------------------------------------------- #
#                                    Blender                                   #
# ---------------------------------------------------------------------------- #


# Example usage
path = "test_dir"
object_loader = ObjectLoader(path, True)
object_loader.CreateObject(ObjIndex=2, class_name="chair")
