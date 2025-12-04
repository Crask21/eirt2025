import bpy
from print_utils import print_green, print_warning, print_error

class Object:
    def __init__(self, obj: bpy.types.Object, class_id: int, class_name: str, spawn_position: tuple[float, float, float] = None):

        if type(obj) == list:
            self.obj = obj[0]
            self.shadow = obj[1]
            self.is_3dgs = True
        else: 
            self.obj = obj
            self.shadow = None
            self.is_3dgs = False


        self.class_id = class_id
        self.name = self.obj.name
        self.location = self.obj.location
        self.rotation = self.obj.rotation_euler
        self.scale = self.obj.scale
        self.obj.pass_index = class_id

        self.obj["class_name"] = class_name
        self.bounding_radius = 0.5  # Placeholder value

        # For resetting between batches:
        if spawn_position is not None:
            self.obj.location = spawn_position

        self.obj.rotation_mode = 'XYZ'

        self.default_location = self.obj.location.copy()
        self.default_rotation = self.obj.rotation_euler.copy()
        self.default_scale = self.obj.scale.copy()
        
        # DON'T configure 3DGS objects immediately - the Kiri modifier isn't added yet
        # It will be configured lazily when needed (e.g., before rendering)
        self._3dgs_configured = False

        # Set shadow settings
        if self.shadow is not None:
            self.shadow.pass_index = class_id
            self.shadow.location = spawn_position
            self.shadow.rotation_mode = 'XYZ'
            self.hide_shadow()
            self.configure_shadow()
            
        self.setKeyframe(self.default_location, self.default_rotation, self.default_scale, frame=0)
        
        
        # Link objects
        if self.shadow is not None:
            bpy.ops.object.select_all(action='DESELECT')  # Clear selection first
            self.obj.select_set(True)
            self.shadow.select_set(True)
            bpy.context.view_layer.objects.active = self.obj  # Set active object
            # Link objects to animation
            bpy.ops.object.make_links_data(type='ANIMATION')


    def setPosition(self, position: tuple[float, float, float]):
        try:
            self.obj.location = position
            if self.shadow is not None:
                self.shadow.location = position
        except (AttributeError, ReferenceError) as e:
            print_error(f"Cannot set position - object reference may be invalid: {e}")
            raise

    def setRotation(self, rotation: tuple[float, float, float]):
        try:
            self.obj.rotation_euler = rotation
            if self.shadow is not None:
                self.shadow.rotation_euler = rotation
        except (AttributeError, ReferenceError) as e:
            print_error(f"Cannot set rotation - object reference may be invalid: {e}")
            raise

    def setScale(self, scale: tuple[float, float, float]):
        try:
            self.obj.scale = scale
            if self.shadow is not None:
                self.shadow.scale = scale
        except (AttributeError, ReferenceError) as e:
            print_error(f"Cannot set scale - object reference may be invalid: {e}")
            raise
    
    def clearPosition(self):
        try:
            # CRITICAL: Copy values to avoid referencing memory that might be re-allocated
            self.obj.location = tuple(self.default_location)
            self.obj.rotation_euler = tuple(self.default_rotation)
            self.obj.scale = tuple(self.default_scale)
        except (AttributeError, ReferenceError) as e:
            print_error(f"Cannot clear position - object reference may be invalid: {e}")
            raise
            
        if self.shadow is not None:
            try:
                self.shadow.location = tuple(self.default_location)
                self.shadow.rotation_euler = tuple(self.default_rotation)
                self.shadow.scale = tuple(self.default_scale)
            except (AttributeError, ReferenceError) as e:
                print_warning(f"Cannot clear shadow position: {e}")
            
    def setKeyframe(self, position: tuple[float, float, float], rotation: tuple[float, float, float], scale: tuple[float, float, float], frame: int):
        self.setPosition(position)
        self.setRotation(rotation)
        self.setScale(scale)
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        self.obj.keyframe_insert(data_path="scale", frame=frame)
        if self.shadow is not None:
            try:
                self.shadow.keyframe_insert(data_path="location", frame=frame)
                self.shadow.keyframe_insert(data_path="rotation_euler", frame=frame)
                self.shadow.keyframe_insert(data_path="scale", frame=frame)
            except (AttributeError, ReferenceError) as e:
                print_warning(f"Cannot insert shadow keyframes: {e}")

        # CRITICAL: Re-fetch animation_data after keyframe insertion to avoid stale references
        # Animation data can be re-allocated when keyframes are added
        try:
            anim_data = self.obj.animation_data
            if anim_data:
                action = anim_data.action
                if action:
                    # Use index-based iteration to avoid holding fcurve references
                    fcurves = action.fcurves
                    for i in range(len(fcurves)):
                        fcurve = fcurves[i]
                        if fcurve and len(fcurve.keyframe_points) > 0:
                            fcurve.keyframe_points[-1].interpolation = 'CONSTANT'
        except (AttributeError, ReferenceError, IndexError) as e:
            # Non-critical - just log the warning
            print_warning(f"Could not set keyframe interpolation: {e}")

    def setOnlyKeyframe(self, frame: int):
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        self.obj.keyframe_insert(data_path="scale", frame=frame)
        if self.shadow is not None:
            try:
                self.shadow.keyframe_insert(data_path="location", frame=frame)
                self.shadow.keyframe_insert(data_path="rotation_euler", frame=frame)
                self.shadow.keyframe_insert(data_path="scale", frame=frame)
            except (AttributeError, ReferenceError) as e:
                print_warning(f"Cannot insert shadow keyframes: {e}")

        # CRITICAL: Re-fetch animation_data after keyframe insertion to avoid stale references
        try:
            anim_data = self.obj.animation_data
            if anim_data:
                action = anim_data.action
                if action:
                    # Use index-based iteration to avoid holding fcurve references
                    fcurves = action.fcurves
                    for i in range(len(fcurves)):
                        fcurve = fcurves[i]
                        if fcurve and len(fcurve.keyframe_points) > 0:
                            fcurve.keyframe_points[-1].interpolation = 'CONSTANT'
        except (AttributeError, ReferenceError, IndexError) as e:
            # Non-critical - just log the warning
            print_warning(f"Could not set keyframe interpolation: {e}")

    def show_shadow(self):
        if self.shadow is not None:
            self.shadow.hide_viewport = False
            self.shadow.hide_render = False

    def hide_shadow(self):
        if self.shadow is not None:
            self.shadow.hide_viewport = True
            self.shadow.hide_render = True
    
    def configure_shadow(self,point_radius: float = 0.02):
        """Create a new Geometry Nodes modifier.
        
        Note: This modifies object data and should be called carefully
        to avoid Blender crashes from array re-allocation.
        """
        if self.shadow is None:
            return
            
        try:
            # Create a new Geometry Nodes modifier
            mod = self.shadow.modifiers.new(name="Geo_MeshToPoints", type='NODES')

            # Create a fresh node group for this modifier
            ng = bpy.data.node_groups.new("MeshToPointsGroup", 'GeometryNodeTree')
            mod.node_group = ng

            nodes = ng.nodes
            links = ng.links

            # Clear any default nodes
            nodes.clear()

            # --- Create nodes --- #
            group_in  = nodes.new("NodeGroupInput")
            group_out = nodes.new("NodeGroupOutput")
            mesh2pts  = nodes.new("GeometryNodeMeshToPoints")

            # Optional: place nodes nicely
            group_in.location  = (-400, 0)
            mesh2pts.location  = (0, 0)
            group_out.location = (400, 0)

            # --- Set Mesh to Points settings ---
            mesh2pts.mode = 'VERTICES'              # same as dropdown "Vertices"
            mesh2pts.inputs["Radius"].default_value = point_radius  # radius input

            # --- Create links (wires) ---
            # Add a Geometry input and output to the group interface
            in_geo  = ng.interface.new_socket(
                name="Geometry",
                in_out='INPUT',
                socket_type='NodeSocketGeometry'
            )
            out_geo = ng.interface.new_socket(
                name="Geometry",
                in_out='OUTPUT',
                socket_type='NodeSocketGeometry'
            )

            links.new(group_in.outputs[in_geo.identifier], mesh2pts.inputs["Mesh"])
            links.new(mesh2pts.outputs["Points"], group_out.inputs[out_geo.identifier])
        except Exception as e:
            print_error(f"Failed to configure shadow geometry nodes: {e}")
            # Don't re-raise - shadow is optional

    def configure_3dgs_object(self, retry_if_missing=False):
        """
        Configure a 3DGS (Gaussian Splat) object for proper rendering.
        Enables camera updates so it composites correctly with native Blender renders.
        
        Args:
            retry_if_missing: If True, will return False if modifier not found yet (for retry logic)
        
        Returns:
            bool: True if successfully configured, False if needs retry
        """
        try:
            # Check if object has the KIRI_3DGS_Render_GN modifier
            has_kiri_modifier = False
            object_modifiers = self.obj.modifiers
            for mod in object_modifiers:
                if mod.name == 'KIRI_3DGS_Render_GN':
                    has_kiri_modifier = True
                    # Ensure modifier is enabled for rendering
                    mod.show_render = True
                    mod.show_viewport = True
                    break
            
            if not has_kiri_modifier:
                if retry_if_missing:
                    # Modifier not added yet by Kiri addon, will retry later
                    return False
                else:
                    print_warning(f"Object {self.obj.name} is marked as 3DGS but has no KIRI_3DGS_Render_GN modifier yet")
                    return False
            
            # Enable camera updates for proper compositing
            # BUT: Don't set it during init - let Kiri addon finish setting up first
            # The property setter triggers internal Kiri logic that expects the modifier to be fully initialized
            obj = self.obj
            if hasattr(obj, 'sna_kiri3dgs_active_object_update_mode'):
                # Check current value - only change if it's disabled
                current_mode = getattr(obj, 'sna_kiri3dgs_active_object_update_mode', None)
                if current_mode == 'Disable Camera Updates':
                    # Only change if disabled - avoid triggering Kiri's update logic during init
                    obj.sna_kiri3dgs_active_object_update_mode = 'Enable Camera Updates'
                    print_green(f"Enabled camera updates for 3DGS object: {obj.name}")
                elif current_mode == 'Enable Camera Updates':
                    # Already enabled, just confirm
                    print_green(f"Camera updates already enabled for 3DGS object: {obj.name}")
                return True
            else:
                print_warning(f"3DGS object {obj.name} missing camera update property")
                return False
                
        except KeyError as e:
            # This happens when Kiri's property update tries to access modifier before it's ready
            if retry_if_missing:
                return False
            print_warning(f"3DGS object {self.obj.name} modifier not ready yet: {e}")
            return False
        except Exception as e:
            print_error(f"Failed to configure 3DGS object {self.obj.name}: {e}")
            return False
    
    def ensure_3dgs_configured(self):
        """
        Ensure 3DGS object is configured. Called before rendering operations.
        Only configures once successfully.
        """
        if self.is_3dgs and not self._3dgs_configured:
            self._3dgs_configured = self.configure_3dgs_object(retry_if_missing=True)
        return self._3dgs_configured if self.is_3dgs else True

    @property
    def obj_class(self):
        return self.obj.get("class_name", "Undefined")