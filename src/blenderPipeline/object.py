import bpy


class Object:
    def __init__(self, obj: bpy.types.Object, class_id: int, class_name: str, spawn_position: tuple[float, float, float] = None):

        if type(obj) == list:
            self.obj = obj[0]
            self.shadow = obj[1]
        else: 
            self.obj = obj
            self.shadow = None


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

        # Set shadow settings
        if self.shadow is not None:
            self.shadow.pass_index = class_id
            self.shadow.location = spawn_position
            self.shadow.rotation_mode = 'XYZ'
            self.hide_shadow()
            self.configure_shadow()
            
        self.setKeyframe(self.default_location, self.default_rotation, self.default_scale, frame=0)

    def setPosition(self, position: tuple[float, float, float]):
        self.obj.location = position
        if self.shadow is not None:
            self.shadow.location = position

    def setRotation(self, rotation: tuple[float, float, float]):
        self.obj.rotation_euler = rotation
        if self.shadow is not None:
            self.shadow.rotation_euler = rotation

    def setScale(self, scale: tuple[float, float, float]):
        self.obj.scale = scale
        if self.shadow is not None:
            self.shadow.scale = scale
    
    def clearPosition(self):
        self.obj.location = self.default_location
        self.obj.rotation_euler = self.default_rotation
        self.obj.scale = self.default_scale
        if self.shadow is not None:
            self.shadow.location = self.default_location
            self.shadow.rotation_euler = self.default_rotation
            self.shadow.scale = self.default_scale
            
    def setKeyframe(self, position: tuple[float, float, float], rotation: tuple[float, float, float], scale: tuple[float, float, float], frame: int):
        self.setPosition(position)
        self.setRotation(rotation)
        self.setScale(scale)
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        self.obj.keyframe_insert(data_path="scale", frame=frame)
        if self.shadow is not None:
            self.shadow.keyframe_insert(data_path="location", frame=frame)
            self.shadow.keyframe_insert(data_path="rotation_euler", frame=frame)
            self.shadow.keyframe_insert(data_path="scale", frame=frame)

        for fcurve in self.obj.animation_data.action.fcurves:
            fcurve.keyframe_points[-1].interpolation = 'CONSTANT'

    def setOnlyKeyframe(self, frame: int):
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        self.obj.keyframe_insert(data_path="scale", frame=frame)
        if self.shadow is not None:
            self.shadow.keyframe_insert(data_path="location", frame=frame)
            self.shadow.keyframe_insert(data_path="rotation_euler", frame=frame)
            self.shadow.keyframe_insert(data_path="scale", frame=frame)

        for fcurve in self.obj.animation_data.action.fcurves:
            fcurve.keyframe_points[-1].interpolation = 'CONSTANT'

    def show_shadow(self):
        if self.shadow is not None:
            self.shadow.hide_viewport = False
            self.shadow.hide_render = False

    def hide_shadow(self):
        if self.shadow is not None:
            self.shadow.hide_viewport = True
            self.shadow.hide_render = True
    
    def configure_shadow(self,point_radius: float = 0.02):
        # Create a new Geometry Nodes modifier

        mod = self.shadow.modifiers.new(name="Geo_MeshToPoints", type='NODES')

        # Create a fresh node group for this modifier
        ng = bpy.data.node_groups.new("MeshToPointsGroup", 'GeometryNodeTree')
        mod.node_group = ng

        nodes = ng.nodes
        links = ng.links

        # Clear any default nodes
        nodes.clear()

        # --- Create nodes ---
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

    @property
    def obj_class(self):
        return self.obj.get("class_name", "Undefined")