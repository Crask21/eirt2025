import bpy

def toggle_object_visibility(show: bool):
    """
    Makes all objects in the current Blender scene visible in the viewport.
    This includes showing them in the viewport, enabling selection, and rendering.
    """
    scene = bpy.context.scene
    
    # Iterate through all objects in the scene
    for obj in scene.objects:
        # Skip objects that don't have "shadow" in their name
        if "shadow" not in obj.name.lower():
            continue
        # Make object visible in viewport
        obj.hide_viewport = not show
        
        # Make object visible in render
        obj.hide_render = not show
        
        # Make object selectable
        obj.hide_select = not show
        
        # If the object is in a collection, make sure the collection is visible too
        for collection in obj.users_collection:
            collection.hide_viewport = False
            collection.hide_render = False
    
    # Update the view layer to ensure changes are reflected
    bpy.context.view_layer.update()
    
    print(f"Made {len(scene.objects)} objects visible in the viewport")


toggle_object_visibility(show=False)
