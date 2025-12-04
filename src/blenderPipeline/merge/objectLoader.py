import bpy
import os
import time
import gc
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
        
        # Track 3DGS imports for memory management
        self._3dgs_import_count = 0
        self._total_gaussians_loaded = 0
        
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
                    try:
                        obj_ply = []
                        # --- Spawn two objects --- #
                        # Store object NAMES before import (not references!)
                        obj_names_before = set(obj.name for obj in bpy.context.scene.objects)
                        
                        bpy.ops.sna.dgs_render_import_ply_bf139(
                            'EXEC_DEFAULT', filepath=obj_path)
                        
                        # Track imports and calculate adaptive delay
                        self._3dgs_import_count += 1
                        
                        # CRITICAL: Adaptive delay based on accumulated load
                        # More objects loaded = longer delay needed for memory management
                        base_delay = 0.15  # Base 150ms
                        accumulated_delay = min(self._3dgs_import_count * 0.05, 0.5)  # Max 500ms
                        total_delay = base_delay + accumulated_delay
                        
                        if self.debug:
                            print(f"[DEBUG] 3DGS import #{self._3dgs_import_count}, waiting {total_delay:.2f}s for processing")
                        
                        time.sleep(total_delay)
                        
                        # Force garbage collection after delay to free memory
                        gc.collect()
                        
                        # CRITICAL: Use name-based lookup instead of direct references
                        # This avoids holding stale references to objects while Kiri addon processes them
                        new_obj_names = [obj.name for obj in bpy.context.scene.objects if obj.name not in obj_names_before]
                        if not new_obj_names:
                            raise ValueError("No objects created by 3DGS import")
                        
                        # Store the NAME, not the object reference yet
                        new_obj_name = new_obj_names[0]
                        
                        # Small additional delay to ensure Kiri addon finishes adding modifiers
                        time.sleep(0.05)
                        
                        # NOW get the object reference after Kiri is done
                        new_obj = bpy.data.objects.get(new_obj_name)
                        if not new_obj:
                            raise ValueError(f"3DGS object '{new_obj_name}' disappeared after import")
                        
                        obj_ply.append(new_obj)
                        
                        print(f"[INFO] ✓ Loaded 3DGS object '{new_obj.name}' (import #{self._3dgs_import_count})")
                        
                        bpy.ops.wm.ply_import(filepath=obj_path)
                        # Get the latest imported object safely
                        imported_objects = [obj for obj in bpy.context.selected_objects]
                        if not imported_objects:
                            raise ValueError("No objects selected after PLY import")
                        shadow = imported_objects[0]
                        shadow.name = obj_ply[0].name + "_shadow"
                        obj_ply.append(shadow)

                        return obj_ply, class_name, self.class_id_dict[class_name]
                    except Exception as e:
                        # Fall back to regular PLY import if not a 3DGS file
                        print(f"[ERROR] Failed to import PLY file with 3DGS: {obj_path}. Error: {e}")
                        print(f"[INFO] Attempting standard PLY import...")
                        try:
                            bpy.ops.wm.ply_import(filepath=obj_path)
                            if bpy.context.selected_objects:
                                return bpy.context.selected_objects[0], class_name, self.class_id_dict[class_name]
                            else:
                                raise ValueError(f"No object selected after PLY import: {obj_path}")
                        except Exception as e2:
                            raise ValueError(f"Failed to import PLY file: {obj_path}. Error: {e2}")
                elif obj_path.lower().endswith('.gltf') or obj_path.lower().endswith('.glb'):
                    bpy.ops.import_scene.gltf(filepath=obj_path)
                if self.debug:
                    print(f"[INFO] Imported object: {obj_path} from class '{class_name}'")
                    print(f"[INFO] Assigned class_id: {self.class_id_dict[class_name]}")
                
                # Verify object was imported successfully
                if not bpy.context.selected_objects:
                    raise ValueError(f"Failed to import object: {obj_path}. No objects selected after import.")

                return bpy.context.selected_objects[0], class_name, self.class_id_dict[class_name]

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
                        obj_ply = []
                        # --- Spawn two objects --- #
                        # Store object NAMES before import (not references!)
                        obj_names_before = set(obj.name for obj in bpy.context.scene.objects)
                        
                        bpy.ops.sna.dgs_render_import_ply_bf139(
                            'EXEC_DEFAULT', filepath=obj_path)
                        
                        # Track imports and calculate adaptive delay
                        self._3dgs_import_count += 1
                        
                        # CRITICAL: Adaptive delay based on accumulated load
                        # More objects loaded = longer delay needed for memory management
                        base_delay = 0.15  # Base 150ms
                        accumulated_delay = min(self._3dgs_import_count * 0.05, 0.5)  # Max 500ms
                        total_delay = base_delay + accumulated_delay
                        
                        if self.debug:
                            print(f"[DEBUG] 3DGS import #{self._3dgs_import_count}, waiting {total_delay:.2f}s for processing")
                        
                        time.sleep(total_delay)
                        
                        # Force garbage collection after delay to free memory
                        gc.collect()
                        
                        # CRITICAL: Use name-based lookup instead of direct references
                        # This avoids holding stale references to objects while Kiri addon processes them
                        new_obj_names = [obj.name for obj in bpy.context.scene.objects if obj.name not in obj_names_before]
                        if not new_obj_names:
                            raise ValueError("No objects created by 3DGS import")
                        
                        # Store the NAME, not the object reference yet
                        new_obj_name = new_obj_names[0]
                        
                        # Small additional delay to ensure Kiri addon finishes adding modifiers
                        time.sleep(0.05)
                        
                        # NOW get the object reference after Kiri is done
                        new_obj = bpy.data.objects.get(new_obj_name)
                        if not new_obj:
                            raise ValueError(f"3DGS object '{new_obj_name}' disappeared after import")
                        
                        obj_ply.append(new_obj)
                        
                        bpy.ops.wm.ply_import(filepath=obj_path)
                        # Get the latest imported object safely
                        imported_objects = [obj for obj in bpy.context.selected_objects]
                        if not imported_objects:
                            raise ValueError("No objects selected after PLY import")
                        shadow = imported_objects[0]
                        shadow.name = obj_ply[0].name + "_shadow"
                        obj_ply.append(shadow)



                        return obj_ply, class_name, self.class_id_dict[class_name]
                    except Exception as e:
                        # Fall back to regular PLY import if not a 3DGS file
                        print(f"[ERROR] Failed to import PLY file with 3DGS: {obj_path}. Error: {e}")
                        print(f"[INFO] Attempting standard PLY import...")
                        try:
                            bpy.ops.wm.ply_import(filepath=obj_path)
                            if bpy.context.selected_objects:
                                return bpy.context.selected_objects[0], class_name, self.class_id_dict[class_name]
                            else:
                                raise ValueError(f"No object selected after PLY import: {obj_path}")
                        except Exception as e2:
                            raise ValueError(f"Failed to import PLY file: {obj_path}. Error: {e2}")
                elif obj_path.lower().endswith('.gltf') or obj_path.lower().endswith('.glb'):
                    bpy.ops.import_scene.gltf(filepath=obj_path)
                if self.debug:
                    print(f"[INFO] Imported object: {obj_path} from class '{class_name}'")
                    print(f"[INFO] Assigned class_id: {self.class_id_dict[class_name]}")
                # Encode filepath in the bpy object
                if not bpy.context.selected_objects:
                    raise ValueError(f"Failed to import object: {obj_path}. No objects selected after import.")
                bpy.context.selected_objects[0]["obj_path"] = obj_path
                return bpy.context.selected_objects[0], class_name, self.class_id_dict[class_name]
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
#path = 'objects'
#object_loader = ObjectLoader(path, True)
# objects = object_loader.FindAllObjects()
# print(f"Total objects found: {len(objects)}")
# print("objects by class:")
# for obj in objects:
#     print(f" - {obj}")
# object_loader.CreateObject(ObjIndex=0, class_name="person")

# object_loader.UpdateAllCameraEnabledObjects()
#object_loader.CreateObject(ObjIndex=2, class_name="chair", )
# object_loader.EnableCameraUpdates()
# object_loader.UpdateActiveToView()
# object_loader.DisableCameraUpdates()
