import os
import numpy as np
import cv2
import time
import math
import random
import shutil
import argparse
import pickle
from tqdm import tqdm
from scipy.io import loadmat

from lib.renderer.camera import Camera
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
from lib.prt.sh_util import rotateSH
from lib.camera_utils import write_intri, write_extri

def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

def generate_cameras(dist=5.6, yaw_angle_step=15, roll_angle_step=30):
    cams = []
    target = [0, 0, 0]
    up = [0, 1, 0]

    for yaw_angle in range(0, 60+1, yaw_angle_step):
        for roll_angle in range(0, 360+1, roll_angle_step):
            eye = np.asarray([0, 0, dist])
            eye = make_rotate(np.radians(-yaw_angle), 0, 0) @ eye
            eye = make_rotate(0, np.radians(roll_angle), 0) @ eye

            fwd = np.asarray(target, np.float64) - eye
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)

            cams.append(
                {
                    'center': eye, 
                    'direction': fwd, 
                    'right': right, 
                    'up': -down, 
                }
            )

    return cams

def render_prt(input_dir, output_dir, subject_name, shs, rndr, im_size, scale=None, num_frames=10):

    cam = Camera(width=im_size, height=im_size, focal=3000, near=0.1, far=40)
    cam.sanity_check()

    mesh_file = os.path.join(input_dir, subject_name+'.obj')
    prt_file = os.path.join(input_dir, 'bounce', 'prt_data.mat')
    tex_file = os.path.join(input_dir, 'material0.jpeg')

    texture_image = cv2.imread(tex_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)
    prt_data = loadmat(prt_file)
    prt, face_prt = prt_data['bounce0'], prt_data['face']

    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    
    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 1.75/(vmax[up_axis] - vmin[up_axis])
    if scale is not None:
        y_scale = y_scale * scale

    rndr.set_norm_mat(y_scale, vmed)
    render_mesh_params = {'y_scale': y_scale, 'vmed': vmed}
    if up_axis == 2:
        R = make_rotate(math.radians(90),0,0)
        rndr.rot_matrix = R
        render_mesh_params['R'] = R
    write_pickle_file(os.path.join(output_dir, 'render_mesh_params.pkl'), render_mesh_params)

    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)    
    rndr.set_albedo(texture_image)

    cam_params = generate_cameras(dist=5.6)
    sh_IDs = np.random.choice(shs.shape[0], num_frames, replace=False)
    intri_cameras = {}
    extri_cameras = {}

    for cam_id, cam_param in enumerate(tqdm(cam_params, ascii=True, desc=subject_name)):
        cam.center = cam_param['center']
        cam.right = cam_param['right']
        cam.up = cam_param['up']
        cam.direction = cam_param['direction']
        cam.sanity_check()
        rndr.set_camera(cam)

        images_dir = os.path.join(output_dir, 'images', '{:02d}'.format(cam_id+1))
        os.makedirs(images_dir, exist_ok=True)
        masks_dir = os.path.join(output_dir, 'masks', '{:02d}'.format(cam_id+1))
        os.makedirs(masks_dir, exist_ok=True)

        intri_cameras['{:02d}'.format(cam_id+1)] = {
            'K': cam.get_intrinsic_matrix(),
            'dist': np.zeros((1, 5))  # dist: (1, 5)
        }
        extri_cameras['{:02d}'.format(cam_id+1)] = {
            'Rvec': cv2.Rodrigues(cam.get_rotation_matrix())[0],
            'R': cam.get_rotation_matrix(),
            'T': cam.get_translation_vector()
        }
        
        for frame_id in range(num_frames):
            sh_id = sh_IDs[frame_id]
            sh = shs[sh_id]
            sh_angle = 0.2*np.pi*(random.random()-0.5)
            sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

            rndr.set_sh(sh)        
            rndr.analytic = False
            rndr.use_inverse_depth = False
            rndr.display()

            out_all_f = rndr.get_color(0)
            out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGRA)

            img = np.uint8(out_all_f[..., :3] * 255)
            mask = np.uint8(out_all_f[..., 3] * 255)

            cv2.imwrite(os.path.join(images_dir, '{:0>6d}.jpg'.format(frame_id)), img)
            cv2.imwrite(os.path.join(masks_dir, '{:0>6d}.png'.format(frame_id)), mask)

    write_intri(os.path.join(output_dir, 'intri.yml'), intri_cameras)
    write_extri(os.path.join(output_dir, 'extri.yml'), extri_cameras)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/THuman2.0_Release')
    parser.add_argument('--output_dir', type=str, default='data/THuman2.0_easymocap')
    parser.add_argument('--ms_rate', type=int, default=1, help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('--egl',  action='store_true', help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('--size',  type=int, default=1024, help='rendering image size')
    parser.add_argument('--subject_name', type=str, default=None, help='the subject name')
    parser.add_argument('--scale',  type=float, default=None, help='the mesh to scale')
    args = parser.parse_args()

    shs = np.load('./env_sh.npy')

    # NOTE: GL context has to be created before any other OpenGL function loads.
    from lib.renderer.gl.init_gl import initialize_GL_context
    initialize_GL_context(width=args.size, height=args.size, egl=args.egl)

    from lib.renderer.gl.prt_render import PRTRender
    rndr = PRTRender(width=args.size, height=args.size, ms_rate=args.ms_rate, egl=args.egl)
    # rndr_uv = PRTRender(width=args.size, height=args.size, uv_mode=True, egl=args.egl)
    if args.subject_name is None:
        subject_names = os.listdir(args.data_root)
        subject_names.sort()
    else:
        subject_names = [args.subject_name]
    for subject_name in subject_names:
        input_dir = os.path.join(args.data_root, subject_name)
        output_dir = os.path.join(args.output_dir, subject_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        render_prt(input_dir, output_dir, subject_name, shs, rndr, im_size=args.size, scale=args.scale)
