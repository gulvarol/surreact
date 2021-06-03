import array
import os
import random
import sys
import time
from os.path import exists, join
from random import choice

import bpy
import Imath
import numpy as np
import OpenEXR
import scipy.io as sio
from mathutils import Vector

sys.path.insert(0, ".")
import utils.argutils as argutils
from utils.blenderutils import (
    create_composite_nodes,
    create_sh_material,
    set_camera,
    set_renderer,
)
from utils.osutils import disable_output_end, disable_output_start, mkdir_safe
from utils.randomutils import pick_background, pick_cam, pick_cloth, pick_shape
from utils.smplutils import SMPL_Body

start_time = time.time()


def log_message(message):
    elapsed_time = time.time() - start_time
    print("[{:.2f} s] {}".format(elapsed_time, message))


def main():
    # parse commandline arguments
    log_message(sys.argv)
    args = argutils.parse_opts()
    argutils.print_args(args)
    idx = args.idx
    repetition = args.repetition
    split_name = args.split_name
    cam_dist_range = args.cam_dist
    cam_height_range = args.cam_height
    zrot_euler = args.zrot_euler
    smpl_data_folder = args.smpl_data_folder
    smpl_data_filename = args.smpl_data_filename
    bg_path = args.bg_path
    vidlist_path = args.vidlist_path
    smpl_result_path = args.smpl_result_path
    smpl_estimation_method = args.smpl_estimation_method
    clothing_option = args.clothing_option
    tmp_path = args.tmp_path
    output_path = args.output_path
    use_pose_smooth = args.use_pose_smooth
    with_trans = args.with_trans
    track_id = args.track_id
    resy = args.resy
    resx = args.resx
    fbeg = args.fbeg
    fend = args.fend
    fskip = 1

    if smpl_estimation_method == "hmmr":
        from utils.hmmrutils import center_people, count_tracks, load_smpl_body_data
    elif smpl_estimation_method == "vibe":
        from utils.vibeutils import center_people, count_tracks, load_smpl_body_data
    else:
        raise ValueError("Unrecognized smpl_estimation_method")

    output_types = {
        "depth": True,
        "fg": True,
        "gtflow": True,
        "normal": True,
        "segm": True,
    }

    with open(vidlist_path, "r") as f:
        vid_paths = f.read().splitlines()

    name = vid_paths[idx]

    output_path = join(output_path, split_name, name)
    common_filename = "{}_v{:03d}_r{:02d}".format(
        name, int(zrot_euler), int(repetition)
    )
    tmp_path = join(tmp_path, split_name, common_filename)
    rgb_path = join(tmp_path, "rgb")
    mp4_path = join(output_path, "{}.mp4".format(common_filename))

    # Check if already computed (use segm for now)
    segm_path = join(output_path, "{}_segm.mat".format(common_filename))
    if os.path.isfile(mp4_path) and os.path.isfile(segm_path):
        print("Already rendered {}".format(segm_path))
        exit()

    # create tmp directory
    if not exists(tmp_path):
        mkdir_safe(tmp_path)

    # create output directory
    if not exists(output_path):
        mkdir_safe(output_path)

    num_tracks, all_track_list = count_tracks(
        name=name, smpl_result_path=smpl_result_path, datasetname=args.datasetname
    )
    log_message("Found {} people.".format(num_tracks))

    # action = int(name[-3:])
    # if action < 50:
    if num_tracks == 1:
        print("Disabling trans and multi-person if any.")
        with_trans = 0

    if track_id == -1:
        track_list = all_track_list
        # track_list = range(num_tracks)
    else:
        track_list = [track_id]
    log_message("Using tracks: {}.".format(track_list))
    num_people = len(track_list)

    hmmr_body_data = []
    for tid in track_list:
        hmmr_body_data.append(
            load_smpl_body_data(
                name=name,
                smpl_result_path=smpl_result_path,
                track_id=tid,
                with_trans=with_trans,
                use_pose_smooth=use_pose_smooth,
                datasetname=args.datasetname,
                noise_factor=args.noise_factor,
                noise_level=args.noise_level,
            )
        )
    log_message("Loaded body data for {}".format(name))
    pose_data = [data["poses"] for data in hmmr_body_data]
    trans_data = [data["trans"] for data in hmmr_body_data]
    trans_data = center_people(trans_data)

    # >> don't use random generator before this point <<

    # initialize RNG with seeds from sequence id
    import hashlib

    s = "synth_data:{:0.0f}:{:0.0f}:{:0.0f}".format(idx, zrot_euler, repetition)
    seed_number = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    log_message("GENERATED SEED {} from string {}".format(seed_number, s))
    random.seed(seed_number)
    np.random.seed(seed_number)

    # create copy-spher.harm. directory if not exists
    sh_dir = join(tmp_path, "spher_harm")
    if not exists(sh_dir):
        mkdir_safe(sh_dir)

    genders = {0: "female", 1: "male"}
    # pick random gender
    gender = choice(genders)
    # gender = 'male'

    scene = bpy.data.scenes["Scene"]
    scene.render.engine = "CYCLES"
    # bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True
    scene.render.film_transparent = True

    # Random background
    bg_img_name = pick_background(bg_path, split_name)

    log_message("Loading smpl data")
    smpl_data = np.load(join(smpl_data_folder, smpl_data_filename))

    log_message("Initializing scene (Clear default scene cube)")
    bpy.ops.object.delete()

    sh_dst = join(sh_dir, "sh_{:05d}.osl".format(idx))
    os.system("cp spher_harm/sh.osl {}".format(sh_dst))

    smpl_body_list = []
    cloth_img_names = []
    shape_data = []
    for person_no in range(num_people):
        # Random cloth
        cloth_img_name = pick_cloth(clothing_option, smpl_data_folder, split_name)
        cloth_img_names.append(cloth_img_name)
        material = bpy.data.materials.new(name="Material_{}".format(person_no))
        material.use_nodes = True
        create_sh_material(material.node_tree, sh_dst, cloth_img_name)
        # SMPL_Body object with ob, gender_name, arm_ob fields
        smpl_body_list.append(
            SMPL_Body(smpl_data_folder, material, gender, person_no=person_no)
        )
        # Random shape
        shape_data.append(pick_shape(smpl_data, gender, split_name))

    # Random light
    sh_coeffs = 0.7 * (2 * np.random.rand(9) - 1)
    # Ambient light (first coeff) needs a minimum  is ambient. Rest is uniformly distributed, higher means brighter.
    sh_coeffs[0] = 0.5 + 0.9 * np.random.rand()
    sh_coeffs[1] = -0.7 * np.random.rand()
    # spherical harmonics material needs a script to be loaded and compiled
    spherical_harmonics = []
    for mname, m in smpl_body_list[0].materials.items():
        spherical_harmonics.append(m.node_tree.nodes["Script"])
        spherical_harmonics[-1].filepath = sh_dst
        spherical_harmonics[-1].update()

    for ish, coeff in enumerate(sh_coeffs):
        for sc in spherical_harmonics:
            sc.inputs[ish + 1].default_value = coeff

    res_paths = create_composite_nodes(
        scene.node_tree, output_types, tmp_path, bg_img_name=bg_img_name, idx=idx
    )

    set_renderer(scene, resy, resx)
    cam_height, cam_dist = pick_cam(cam_height_range, cam_dist_range)
    print("Picked cam_height {}, cam_dist {}".format(cam_height, cam_dist))
    cam_ob = set_camera(cam_dist=cam_dist, cam_height=cam_height, zrot_euler=zrot_euler)

    # Different tracks have different num #data, take the min
    N = min([len(data) for data in pose_data])
    if fend == -1:
        log_message("Using all {} frames".format(N))
        fend = N
    elif fend > N:
        log_message(
            "Fend {} is out of boundary for {} frames setting it to {}.".format(
                fend, N, N
            )
        )
        fend = N
        # exit()
    log_message("Rendering frames {}:{}:{}:".format(fbeg, fskip, fend))

    matfile_info = join(output_path, "{}_info.mat".format(common_filename))
    log_message("Working on {}".format(matfile_info))

    # allocate
    dict_info = {}
    dict_info["bg"] = bg_img_name
    # dict_info['cloth'] = np.zeros((num_people, ), dtype=np.object)  # clothing texture image path
    dict_info["cloth"] = cloth_img_names
    # 0 for male, 1 for female
    dict_info["gender"] = list(genders)[list(genders.values()).index(gender)]
    # 2D joint positions in pixel space
    dict_info["joints2D"] = np.empty((num_people, 2, 24, N), dtype="float32")
    # 3D joint positions in world coordinates
    dict_info["joints3D"] = np.empty((num_people, 3, 24, N), dtype="float32")
    # joint angles from SMPL
    dict_info["pose"] = np.empty((num_people, 72, N), dtype="float32")
    dict_info["sequence"] = name
    # dict_info['shape'] = np.empty((num_people, 10), dtype='float32')
    dict_info["shape"] = shape_data
    dict_info["source"] = "ntu"
    dict_info["zrot_euler"] = zrot_euler
    dict_info["light"] = sh_coeffs
    dict_info["cam_height"] = cam_height
    dict_info["cam_dist"] = cam_dist

    for person_no in range(num_people):
        smpl_body_list[person_no].reset_joint_positions(
            shape_data[person_no], scene, cam_ob
        )
        # smpl_body_list[person_no].arm_ob.animation_data_clear()
    cam_ob.animation_data_clear()

    # LOOP TO CREATE 3D ANIMATION: create a keyframe animation with pose, trans, blendshapes
    for seq_frame, i in enumerate(range(fbeg, fend, fskip)):
        # For each person
        for person_no in range(num_people):
            pose = pose_data[person_no][i]
            trans = trans_data[person_no][i]
            shape = shape_data[person_no]
            scene.frame_set(seq_frame)
            # apply the translation, pose and shape to the character
            smpl_body_list[person_no].apply_trans_pose_shape(
                Vector(trans), pose, shape, scene, cam_ob, seq_frame
            )

            dict_info["pose"][person_no, :, seq_frame] = pose
            # scene.update()  # blender < 2.8x
            bpy.context.view_layer.update()

    for part, material in smpl_body_list[0].materials.items():
        material.node_tree.nodes["Vector Math"].inputs[1].default_value[:2] = (0, 0)

    # LOOP TO RENDER: iterate over the keyframes and render
    for seq_frame, i in enumerate(range(fbeg, fend, fskip)):
        scene.frame_set(seq_frame)

        # scene.render.use_antialiasing = False  # blender < 2.8x
        scene.render.filepath = join(rgb_path, "Image{:04d}.png".format(seq_frame))

        log_message("Rendering frame {}".format(seq_frame))

        # disable render output
        old = disable_output_start()
        # Render
        bpy.ops.render.render(write_still=True)
        # disable output redirection
        disable_output_end(old)

        for person_no in range(num_people):
            # bone locations should be saved after rendering so that the bones are updated
            bone_locs_2D, bone_locs_3D = smpl_body_list[person_no].get_bone_locs(
                scene, cam_ob
            )
            dict_info["joints2D"][person_no, :, :, seq_frame] = np.transpose(
                bone_locs_2D
            )
            dict_info["joints3D"][person_no, :, :, seq_frame] = np.transpose(
                bone_locs_3D
            )
            smpl_body_list[person_no].reset_pose()

    # save a .blend file for debugging:
    # bpy.ops.wm.save_as_mainfile(filepath=join(tmp_path, 'pre.blend'))

    # save RGB data with ffmpeg
    # (if you don't have h264 codec, you can replace with another one and control the quality with something like -q:v 3)
    cmd_ffmpeg = (
        "ffmpeg -loglevel panic -y -r 30 -i "
        "{}"
        " -c:v h264 -pix_fmt yuv420p -crf 23 "
        "{}"
        "".format(join(rgb_path, "Image%04d.png"), mp4_path)
    )
    log_message("Generating RGB video ({})".format(cmd_ffmpeg))
    os.system(cmd_ffmpeg)

    if output_types["fg"]:
        fg_mp4_path = join(output_path, "{}_fg.mp4".format(common_filename))
        cmd_ffmpeg_fg = (
            "ffmpeg  -loglevel panic -y -r 30 -i "
            "{}"
            " -c:v h264 -pix_fmt yuv420p -crf 23 "
            "{}"
            "".format(join(res_paths["fg"], "Image%04d.png"), fg_mp4_path)
        )
        log_message("Generating fg video ({})".format(cmd_ffmpeg_fg))
        os.system(cmd_ffmpeg_fg)

    # save annotation excluding png/exr data to _info.mat file
    sio.savemat(matfile_info, dict_info, do_compression=True)

    # .mat files
    matfile_normal = join(output_path, "{}_normal.mat".format(common_filename))
    matfile_gtflow = join(output_path, "{}_gtflow.mat".format(common_filename))
    matfile_depth = join(output_path, "{}_depth.mat".format(common_filename))
    matfile_segm = join(output_path, "{}_segm.mat".format(common_filename))
    dict_normal = {}
    dict_gtflow = {}
    dict_depth = {}
    dict_segm = {}
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # LOOP OVER FRAMES
    for seq_frame, i in enumerate(range(fbeg, fend, fskip)):
        log_message("Processing frame {}".format(seq_frame))
        for k, folder in res_paths.items():
            if not k == "vblur" and not k == "fg":
                path = join(folder, "Image{:04d}.exr".format(seq_frame))
                exr_file = OpenEXR.InputFile(path)
                if k == "normal":
                    mat = np.transpose(
                        np.reshape(
                            [
                                array.array("f", exr_file.channel(Chan, FLOAT)).tolist()
                                for Chan in ("R", "G", "B")
                            ],
                            (3, resx, resy),
                        ),
                        (1, 2, 0),
                    )
                    dict_normal["normal_{:d}".format(seq_frame + 1)] = mat.astype(
                        np.float32, copy=False
                    )  # +1 for the 1-indexing
                elif k == "gtflow":
                    mat = np.transpose(
                        np.reshape(
                            [
                                array.array("f", exr_file.channel(Chan, FLOAT)).tolist()
                                for Chan in ("R", "G")
                            ],
                            (2, resx, resy),
                        ),
                        (1, 2, 0),
                    )
                    dict_gtflow["gtflow_{:d}".format(seq_frame + 1)] = mat.astype(
                        np.float32, copy=False
                    )
                elif k == "depth":
                    mat = np.reshape(
                        [
                            array.array("f", exr_file.channel(Chan, FLOAT)).tolist()
                            for Chan in ("R")
                        ],
                        (resx, resy),
                    )
                    dict_depth["depth_{:d}".format(seq_frame + 1)] = mat.astype(
                        np.float32, copy=False
                    )
                elif k == "segm":
                    mat = np.reshape(
                        [
                            array.array("f", exr_file.channel(Chan, FLOAT)).tolist()
                            for Chan in ("R")
                        ],
                        (resx, resy),
                    )
                    dict_segm["segm_{:d}".format(seq_frame + 1)] = mat.astype(
                        np.uint8, copy=False
                    )
                # remove(path)

    if output_types["normal"]:
        sio.savemat(matfile_normal, dict_normal, do_compression=True)
    if output_types["gtflow"]:
        sio.savemat(matfile_gtflow, dict_gtflow, do_compression=True)
    if output_types["depth"]:
        sio.savemat(matfile_depth, dict_depth, do_compression=True)
    if output_types["segm"]:
        sio.savemat(matfile_segm, dict_segm, do_compression=True)

    # cleaning up tmp
    if tmp_path != "" and tmp_path != "/":
        log_message("Cleaning up tmp")
        os.system("rm -rf {}".format(tmp_path))

    log_message("Completed batch")


if __name__ == "__main__":
    main()
