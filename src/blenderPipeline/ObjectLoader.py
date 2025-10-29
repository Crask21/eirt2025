import bpy
import os
from pathlib import Path


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
                # Try different STL import methods based on file extension
                if obj_path.lower().endswith('.stl'):
                    bpy.ops.wm.stl_import(filepath=obj_path)
                elif obj_path.lower().endswith('.obj'):
                    bpy.ops.wm.obj_import(filepath=obj_path)
                elif obj_path.lower().endswith('.ply'):
                    bpy.ops.wm.ply_import(filepath=obj_path)

                return bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        elif class_name is None and ObjIndex is not None:
            if 0 <= ObjIndex < len(self.AllObjects):
                # Convert Path to string
                obj_path = str(self.AllObjects[ObjIndex])
                # Try different STL import methods based on file extension
                if obj_path.lower().endswith('.stl'):
                    bpy.ops.wm.stl_import(filepath=obj_path)
                elif obj_path.lower().endswith('.obj'):
                    bpy.ops.wm.obj_import(filepath=obj_path)
                elif obj_path.lower().endswith('.ply'):
                    bpy.ops.wm.ply_import(filepath=obj_path)

                return bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        raise ValueError("Invalid ObjIndex or class_name")

# ---------------------------------------------------------------------------- #
#                                    Blender                                   #
# ---------------------------------------------------------------------------- #


# Example usage
path = "test_dir"
object_loader = ObjectLoader(path, True)
object_loader.CreateObject(ObjIndex=0)
object_loader.CreateObject(ObjIndex=1)
object_loader.CreateObject(ObjIndex=2)
