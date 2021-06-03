import bpy
import math
from mathutils import Matrix, Euler
from os.path import join


def set_camera(cam_dist=7, cam_height=1, zrot_euler=0):
    # set camera properties and initial position
    bpy.ops.object.select_all(action="DESELECT")
    cam_ob = bpy.data.objects["Camera"]  # scn.camera
    # bpy.context.scene.objects.active = cam_ob  # blender < 2.8x
    bpy.context.view_layer.objects.active = cam_ob

    rot_z = math.radians(zrot_euler)
    rot_x = math.atan((cam_height - 1) / cam_dist)
    # Rotate -90 degrees around x axis to have the person face cam
    cam_rot = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0))).to_4x4()
    # Rotation by zrot_euler around z-axis
    cam_rot_z = Euler((0, rot_z, 0), "XYZ").to_matrix().to_4x4()
    cam_rot_x = Euler((rot_x, 0, 0), "XYZ").to_matrix().to_4x4()

    # Rotate around the object by rot_z with a fixed radius = cam_dist
    cam_trans = Matrix.Translation(
        [cam_dist * math.sin(rot_z), cam_dist * math.cos(rot_z), cam_height]
    )

    # cam_ob.matrix_world = cam_trans * cam_rot * cam_rot_z * cam_rot_x  # blender < 2.8x
    cam_ob.matrix_world = cam_trans @ cam_rot @ cam_rot_z @ cam_rot_x

    cam_ob.data.angle = math.radians(40)
    cam_ob.data.lens = 60
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32
    print("Camera location {}".format(cam_ob.location))
    print("Camera rotation {}".format(cam_ob.rotation_euler))
    print("Camera matrix {}".format(cam_ob.matrix_world))
    return cam_ob


def set_renderer(scene, resy, resx):
    scn = bpy.context.scene
    scn.cycles.film_transparent = True
    # blender < 2.8x
    # scn.render.layers['RenderLayer'].use_pass_vector = True
    # scn.render.layers['RenderLayer'].use_pass_normal = True
    # scene.render.layers['RenderLayer'].use_pass_emit = True
    # scene.render.layers['RenderLayer'].use_pass_material_index = True
    vl = bpy.context.view_layer
    vl.use_pass_vector = True
    vl.use_pass_normal = True
    vl.use_pass_emit = True
    vl.use_pass_material_index = True

    # set render size
    scn.render.resolution_x = resy
    scn.render.resolution_y = resx
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"


# create the different passes that we render
def create_composite_nodes(tree, output_types, tmp_path, bg_img_name=None, idx=0):
    res_paths = {
        k: join(tmp_path, "%05d_%s" % (idx, k)) for k in output_types if output_types[k]
    }

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create node for foreground image
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.location = -300, 400

    # create node for background image
    bg_im = tree.nodes.new("CompositorNodeImage")
    bg_im.location = -300, 30
    if bg_img_name is not None:
        bg_img = bpy.data.images.load(bg_img_name)
        bg_im.image = bg_img
        print("Set the background image!")

    # create node for mixing foreground and background images
    mix = tree.nodes.new("CompositorNodeMixRGB")
    mix.location = 40, 30
    mix.use_alpha = True

    # create node for the final output
    composite_out = tree.nodes.new("CompositorNodeComposite")
    composite_out.location = 240, 30

    # create node for saving depth
    if output_types["depth"]:
        depth_out = tree.nodes.new("CompositorNodeOutputFile")
        depth_out.location = 40, 700
        depth_out.format.file_format = "OPEN_EXR"
        depth_out.base_path = res_paths["depth"]

    # create node for saving normals
    if output_types["normal"]:
        normal_out = tree.nodes.new("CompositorNodeOutputFile")
        normal_out.location = 40, 600
        normal_out.format.file_format = "OPEN_EXR"
        normal_out.base_path = res_paths["normal"]

    # create node for saving foreground image
    if output_types["fg"]:
        fg_out = tree.nodes.new("CompositorNodeOutputFile")
        fg_out.location = 170, 600
        fg_out.format.file_format = "PNG"
        fg_out.base_path = res_paths["fg"]

    # create node for saving ground truth flow
    if output_types["gtflow"]:
        gtflow_out = tree.nodes.new("CompositorNodeOutputFile")
        gtflow_out.location = 40, 500
        gtflow_out.format.file_format = "OPEN_EXR"
        gtflow_out.base_path = res_paths["gtflow"]

    # create node for saving segmentation
    if output_types["segm"]:
        segm_out = tree.nodes.new("CompositorNodeOutputFile")
        segm_out.location = 40, 400
        segm_out.format.file_format = "OPEN_EXR"
        segm_out.base_path = res_paths["segm"]

    # merge fg and bg images
    tree.links.new(bg_im.outputs[0], mix.inputs[1])
    tree.links.new(layers.outputs["Image"], mix.inputs[2])

    tree.links.new(mix.outputs[0], composite_out.inputs[0])  # bg+fg image
    if output_types["fg"]:
        tree.links.new(layers.outputs["Image"], fg_out.inputs[0])  # save fg
    if output_types["depth"]:
        # 'Z' instead of 'Depth'  # blender < 2.8x
        tree.links.new(layers.outputs["Depth"], depth_out.inputs[0])  # save depth
    if output_types["normal"]:
        tree.links.new(layers.outputs["Normal"], normal_out.inputs[0])  # save normal
    if output_types["gtflow"]:
        # 'Speed' instead of 'Vector'  # blender < 2.8x
        tree.links.new(
            layers.outputs["Vector"], gtflow_out.inputs[0]
        )  # save ground truth flow
    if output_types["segm"]:
        tree.links.new(
            layers.outputs["IndexMA"], segm_out.inputs[0]
        )  # save segmentation

    return res_paths


# creation of the spherical harmonics material, using an OSL script
def create_sh_material(tree, sh_path, cloth_img_name):
    print("Building materials tree")
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    uv = tree.nodes.new("ShaderNodeTexCoord")
    uv.location = -800, 400

    uv_xform = tree.nodes.new("ShaderNodeVectorMath")
    uv_xform.location = -600, 400
    uv_xform.inputs[1].default_value = (0, 0, 1)
    # uv_xform.operation = 'AVERAGE'  # blender < 2.8x
    uv_xform.operation = "ADD"  # TODO

    cloth_img = bpy.data.images.load(cloth_img_name)
    uv_im = tree.nodes.new("ShaderNodeTexImage")
    uv_im.location = -400, 400
    uv_im.image = cloth_img

    rgb = tree.nodes.new("ShaderNodeRGB")
    rgb.location = -400, 200

    script = tree.nodes.new("ShaderNodeScript")
    script.location = -230, 400
    script.mode = "EXTERNAL"
    script.filepath = sh_path  # 'spher_harm/sh.osl' #using the same file from multiple jobs causes white texture
    script.update()

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.location = -60, 400

    mat_out = tree.nodes.new("ShaderNodeOutputMaterial")
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], script.inputs[0])
    tree.links.new(script.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])
