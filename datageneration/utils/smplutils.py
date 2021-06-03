import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion
import numpy as np
import pickle as pkl
import os

from utils.geometryutils import rodrigues2bshapes


class SMPL_Body:
    def __init__(self, smpl_data_folder, material, gender="female", person_no=0):
        # load fbx model
        bpy.ops.import_scene.fbx(
            filepath=os.path.join(
                smpl_data_folder,
                "basicModel_{}_lbs_10_207_0_v1.0.2.fbx".format(gender[0]),
            ),
            axis_forward="Y",
            axis_up="Z",
            global_scale=100,
        )
        J_regressors = pkl.load(
            open(os.path.join(smpl_data_folder, "joint_regressors.pkl"), "rb")
        )
        # 24 x 6890 regressor from vertices to joints
        self.joint_regressor = J_regressors["J_regressor_{}".format(gender)]
        armature_name = "Armature_{}".format(person_no)
        bpy.context.active_object.name = armature_name
        self.gender_name = "{}_avg".format(gender[0])

        self.obj_name = "body_{:d}".format(person_no)
        bpy.data.objects[armature_name].children[0].name = self.obj_name
        # not the default self.gender_name because each time fbx is loaded it adds some suffix
        self.ob = bpy.data.objects[self.obj_name]
        # Rename the armature
        self.ob.data.use_auto_smooth = False  # autosmooth creates artifacts
        # assign the existing spherical harmonics material
        self.ob.active_material = bpy.data.materials["Material_{}".format(person_no)]
        # clear existing animation data
        self.ob.data.shape_keys.animation_data_clear()
        self.arm_ob = bpy.data.objects[armature_name]
        self.arm_ob.animation_data_clear()

        self.setState0()
        # self.ob.select = True  # blender < 2.8x
        self.ob.select_set(True)
        # bpy.context.scene.objects.active = self.ob  # blender < 2.8x
        bpy.context.view_layer.objects.active = self.ob
        self.materials = self.create_segmentation(material)

        # unblocking both the pose and the blendshape limits
        for k in self.ob.data.shape_keys.key_blocks.keys():
            bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
            bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

        # bpy.context.scene.objects.active = self.arm_ob  # blender < 2.8x
        bpy.context.view_layer.objects.active = self.arm_ob

        # order
        self.part_match = {
            "root": "root",
            "bone_00": "Pelvis",
            "bone_01": "L_Hip",
            "bone_02": "R_Hip",
            "bone_03": "Spine1",
            "bone_04": "L_Knee",
            "bone_05": "R_Knee",
            "bone_06": "Spine2",
            "bone_07": "L_Ankle",
            "bone_08": "R_Ankle",
            "bone_09": "Spine3",
            "bone_10": "L_Foot",
            "bone_11": "R_Foot",
            "bone_12": "Neck",
            "bone_13": "L_Collar",
            "bone_14": "R_Collar",
            "bone_15": "Head",
            "bone_16": "L_Shoulder",
            "bone_17": "R_Shoulder",
            "bone_18": "L_Elbow",
            "bone_19": "R_Elbow",
            "bone_20": "L_Wrist",
            "bone_21": "R_Wrist",
            "bone_22": "L_Hand",
            "bone_23": "R_Hand",
        }

    def setState0(self):
        for ob in bpy.data.objects.values():
            # ob.select = False  # blender < 2.8x
            ob.select_set(False)
        # bpy.context.scene.objects.active = None  # blender < 2.8x
        bpy.context.view_layer.objects.active = None

    # create one material per part as defined in a pickle with the segmentation
    # this is useful to render the segmentation in a material pass
    def create_segmentation(self, material):
        print("Creating materials segmentation")
        sorted_parts = [
            "hips",
            "leftUpLeg",
            "rightUpLeg",
            "spine",
            "leftLeg",
            "rightLeg",
            "spine1",
            "leftFoot",
            "rightFoot",
            "spine2",
            "leftToeBase",
            "rightToeBase",
            "neck",
            "leftShoulder",
            "rightShoulder",
            "head",
            "leftArm",
            "rightArm",
            "leftForeArm",
            "rightForeArm",
            "leftHand",
            "rightHand",
            "leftHandIndex1",
            "rightHandIndex1",
        ]
        part2num = {part: (ipart + 1) for ipart, part in enumerate(sorted_parts)}
        materials = {}
        vgroups = {}
        with open("smpl_data/segm_per_v_overlap.pkl", "rb") as f:
            vsegm = pkl.load(f)
        bpy.ops.object.material_slot_remove()
        parts = sorted(vsegm.keys())
        for part in parts:
            vs = vsegm[part]
            # vgroups[part] = self.ob.vertex_groups.new(part)  # blender < 2.8x
            vgroups[part] = self.ob.vertex_groups.new(name=part)
            vgroups[part].add(vs, 1.0, "ADD")
            bpy.ops.object.vertex_group_set_active(group=part)
            materials[part] = material.copy()
            materials[part].pass_index = part2num[part]
            bpy.ops.object.material_slot_add()
            self.ob.material_slots[-1].material = materials[part]
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="DESELECT")
            bpy.ops.object.vertex_group_select()
            bpy.ops.object.material_slot_assign()
            bpy.ops.object.mode_set(mode="OBJECT")
        return materials

    def apply_trans_pose_shape(self, trans, pose, shape, scene, cam_ob, frame=None):
        """
        Apply trans pose and shape to character
        """
        # transform pose into rotation matrices (for pose) and pose blendshapes
        mrots, bsh = rodrigues2bshapes(pose)
        # set the location of the first bone to the translation parameter
        self.arm_ob.pose.bones[self.gender_name + "_Pelvis"].location = trans
        if frame is not None:
            self.arm_ob.pose.bones[self.gender_name + "_root"].keyframe_insert(
                "location", frame=frame
            )
            self.arm_ob.pose.bones[self.gender_name + "_root"].keyframe_insert(
                "rotation_quaternion", frame=frame
            )
        # set the pose of each bone to the quaternion specified by pose
        for ibone, mrot in enumerate(mrots):
            bone = self.arm_ob.pose.bones[
                self.gender_name + "_" + self.part_match["bone_{:02d}".format(ibone)]
            ]
            bone.rotation_quaternion = Matrix(mrot).to_quaternion()
            if frame is not None:
                bone.keyframe_insert("rotation_quaternion", frame=frame)
                bone.keyframe_insert("location", frame=frame)
        # apply pose blendshapes
        for ibshape, bshape in enumerate(bsh):
            self.ob.data.shape_keys.key_blocks[
                "Pose{:03d}".format(ibshape)
            ].value = bshape
            if frame is not None:
                self.ob.data.shape_keys.key_blocks[
                    "Pose{:03d}".format(ibshape)
                ].keyframe_insert("value", index=-1, frame=frame)
        # apply shape blendshapes
        for ibshape, shape_elem in enumerate(shape):
            self.ob.data.shape_keys.key_blocks[
                "Shape{:03d}".format(ibshape)
            ].value = shape_elem
            if frame is not None:
                self.ob.data.shape_keys.key_blocks[
                    "Shape{:03d}".format(ibshape)
                ].keyframe_insert("value", index=-1, frame=frame)

    def get_bone_locs(self, scene, cam_ob):
        n_bones = 24
        render_scale = scene.render.resolution_percentage / 100
        render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
        )
        bone_locations_2d = np.empty((n_bones, 2))
        bone_locations_3d = np.empty((n_bones, 3), dtype="float32")

        # obtain the coordinates of each bone head in image space
        for ibone in range(n_bones):
            bone = self.arm_ob.pose.bones[
                self.gender_name + "_" + self.part_match["bone_{:02d}".format(ibone)]
            ]
            # co_2d = world_to_camera_view(scene, cam_ob, self.arm_ob.matrix_world * bone.head)  # blender < 2.8x
            co_2d = world_to_camera_view(
                scene, cam_ob, self.arm_ob.matrix_world @ bone.head
            )
            # co_3d = self.arm_ob.matrix_world * bone.head  # blender < 2.8x
            co_3d = self.arm_ob.matrix_world @ bone.head
            bone_locations_3d[ibone] = (co_3d.x, co_3d.y, co_3d.z)
            bone_locations_2d[ibone] = (
                round(co_2d.x * render_size[0]),
                round(co_2d.y * render_size[1]),
            )
        return bone_locations_2d, bone_locations_3d

    def reset_pose(self):
        self.arm_ob.pose.bones[
            self.gender_name + "_root"
        ].rotation_quaternion = Quaternion((1, 0, 0, 0))

    def reset_joint_positions(self, shape, scene, cam_ob):
        """
        Reset the joint positions of the character according to its new shape
        """
        orig_trans = np.asarray(
            self.arm_ob.pose.bones[self.gender_name + "_Pelvis"].location
        ).copy()
        # zero the pose and trans to obtain joint positions in zero pose
        self.apply_trans_pose_shape(orig_trans, np.zeros(72), shape, scene, cam_ob)

        # obtain a mesh after applying modifiers
        bpy.ops.wm.memory_statistics()
        # me holds the vertices after applying the shape blendshapes
        # me = self.ob.to_mesh(scene, True, 'PREVIEW')  # blender < 2.8x
        depsgraph = bpy.context.evaluated_depsgraph_get()
        me = self.ob.evaluated_get(depsgraph).to_mesh()

        num_vertices = len(me.vertices)  # 6890
        reg_vs = np.empty((num_vertices, 3))
        for iiv in range(num_vertices):
            reg_vs[iiv] = me.vertices[iiv].co
        # bpy.data.meshes.remove(me)  # blender < 2.8x
        self.ob.evaluated_get(depsgraph).to_mesh_clear()

        # regress joint positions in rest pose
        joint_xyz = self.joint_regressor.dot(reg_vs)
        # adapt joint positions in rest pose
        # self.arm_ob.hide = False
        # Added this line
        # bpy.context.scene.objects.active = self.arm_ob  # blender < 2.8x
        bpy.context.view_layer.objects.active = self.arm_ob
        bpy.ops.object.mode_set(mode="EDIT")
        # self.arm_ob.hide = True
        for ibone in range(24):
            bb = self.arm_ob.data.edit_bones[
                self.gender_name + "_" + self.part_match["bone_{:02d}".format(ibone)]
            ]
            bboffset = bb.tail - bb.head
            bb.head = joint_xyz[ibone]
            bb.tail = bb.head + bboffset
        bpy.ops.object.mode_set(mode="OBJECT")
