import bpy
import os
from pathlib import Path
# bpy.ops.preferences.addon_enable(module="kiri_3dgs")
import json

# ------------------------------- ObjectLoader ------------------------------- #
# print("objectLoader.py loaded")

class ObjectLoader:
    def __init__(self, DatabasePath: str, debug: bool = False, includeGaussianSplatts: bool = False):
        self.allowed_extensions = [".obj", ".stl", "gltf", ".glb"]
        if includeGaussianSplatts:
            self.allowed_extensions.append(".ply")
        self.DatabasePath = Path(DatabasePath)
        self.debug = debug
        if debug:
            print(f"[INFO] DatabasePath set to: {self.DatabasePath}")

        with open(self.DatabasePath / "class_id.json", 'r') as f:
            self.class_id_dict = json.load(f)
        if debug:
            print(f"[INFO] Loaded class_id.json: {self.class_id_dict}")

        self.Classes = self.read_classes()
        if debug:
            print(f"[INFO] Classes found: {self.Classes}")

        self.ObjectsByClass = self.organize_objects_by_class()
        self.TotalObjects = len(self.FindAllObjects())
        if debug:
            print(f"[INFO] TotalObjects found: {self.TotalObjects}")

        self.AllObjects = self.FindAllObjects()

# ------------------------------- Create Object ------------------------------ #
    def CreateObject(self, ObjIndex=None, spawn_position: tuple[float, float, float] = None, class_name: str = None):
        """
        Create an object from the database.
        Args:
            ObjIndex (int, optional): Index of the object to load. Defaults to None.
            class_name (str, optional): Class name to load the object from. Defaults to None
        """
        if class_name:
            if class_name not in self.class_id_dict:
                raise ValueError(f"Class name '{class_name}' not found in class_id.json")
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
                elif obj_path.lower().endswith('.gltf') or obj_path.lower().endswith('.glb'):
                    bpy.ops.import_scene.gltf(filepath=obj_path)
                if self.debug:
                    print(f"[INFO] Imported object: {obj_path} from class '{class_name}'")
                    print(f"[INFO] Assigned class_id: {self.class_id_dict[class_name]}")

                return bpy.context.selected_objects[0], class_name, self.class_id_dict[class_name] if bpy.context.selected_objects else None

        elif class_name is None and ObjIndex is not None:
            if 0 <= ObjIndex < len(self.AllObjects):
                # Convert Path to string
                obj_path = str(self.AllObjects[ObjIndex])
                class_name = self.AllObjects[ObjIndex].parent.name
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
                elif obj_path.lower().endswith('.gltf') or obj_path.lower().endswith('.glb'):
                    bpy.ops.import_scene.gltf(filepath=obj_path)
                if self.debug:
                    print(f"[INFO] Imported object: {obj_path} from class '{class_name}'")
                    print(f"[INFO] Assigned class_id: {self.class_id_dict[class_name]}")
                return bpy.context.selected_objects[0], class_name, self.class_id_dict[class_name] if bpy.context.selected_objects else None
        raise ValueError("Invalid ObjIndex or class_name")

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
                class_name = entry_path.name
                if class_name.lower() != "_unsorted":
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

# --------------------------- 3DGS Camera Updates ---------------------------- #
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

    def ListAllObjects(self):
        """
        List all objects currently in the Blender scene.

        Returns:
            list: of dictionaries with object information (if detailed=True)
        """
        objects_info = []
        for obj in bpy.context.scene.objects:
            obj_info = {
                'name': obj.name,
                'type': obj.type,
                'location': tuple(obj.location),
                'is_3dgs': 'KIRI_3DGS_Render_GN' in [mod.name for mod in obj.modifiers] if obj.type == 'MESH' else False
            }
            objects_info.append(obj_info)

        # Debug print:
        if self.debug:
            print(f"Found {len(objects_info)} objects in scene:")
            for obj_info in objects_info:
                status = "ACTIVE" if obj_info['name'] == bpy.context.view_layer.objects.active.name else ""
                print(
                    f"  - {obj_info['name']} [{obj_info['type']}]{' [3DGS]' if obj_info['is_3dgs'] else ''} {status}")
        return objects_info

    def GetObjectByName(self, name):
        """
        Get a Blender object by its name.

        Args:
            name (str): Name of the object to find

        Returns:
            bpy.types.Object or None: The object if found, None otherwise
        """
        obj = bpy.data.objects.get(name)
        if obj:
            print(f"Found object: {name}")
            return obj
        else:
            print(f"Object '{name}' not found")
            return None


# ---------------------------------------------------------------------------- #
#                                    Blender                                   #
# ---------------------------------------------------------------------------- #

# Example usage
# path = 'F:\\datasets\\eirt_objects'
# object_loader = ObjectLoader(path, True)
# objects = object_loader.FindAllObjects()
# print(f"Total objects found: {len(objects)}")
# print("objects by class:")
# for obj in objects:
#     print(f" - {obj}")
# object_loader.CreateObject(ObjIndex=0, class_name="person")

# object_loader.UpdateAllCameraEnabledObjects()
# object_loader.CreateObject(ObjIndex=2, class_name="chair")
# object_loader.EnableCameraUpdates()
# object_loader.UpdateActiveToView()
# object_loader.DisableCameraUpdates()
