import numpy as np
from . import d405_helpers_without_pyrealsense as dh
from .fingertip_geometry import fingertip_transforms
import cv2

suctioncup_height = {
    'cup_top': 0.014,
    'cup_bottom': 0.007,
    'cylinder_top': 0.004,
    'cylinder_bottom': 0.0
}

class ArucoToFingertips:
    def __init__(self, default_height_above_mounting_surface=None):
        self.default_height_above_mounting_surface = default_height_above_mounting_surface
        self.marker_left_name ='finger_left'
        self.marker_right_name = 'finger_right'
        self.marker_names = [self.marker_left_name, self.marker_right_name]
        
        self.fingertip_basename = 'link_gripper_fingertip_'
        self.aruco_basename = 'link_aruco_fingertip_'

        self.sides = ['left', 'right']

        self.transforms = fingertip_transforms()
        self.translations = {
            side: transform[:3, 3].copy() for side, transform in self.transforms.items()
        }
        self.rotations = {
            side: transform[:3, :3].copy() for side, transform in self.transforms.items()
        }

            
    def get_transforms(self):
        return self.transforms

    def get_rotations(self):
        return self.rotations

    def get_translations(self):
        return self.translations

    def get_fingertips(self, markers, height_above_mounting_surface=None):
        # Find the fingertip poses using finger ArUco markers observed from a gripper camera.

        fingertips = {}
        
        for k in markers:
            m = markers[k]
            name = m['info']['name']
            if name in self.marker_names:
                marker_pos = m['pos']
                marker_x_axis = m['x_axis']
                marker_y_axis = m['y_axis']
                marker_z_axis = m['z_axis']

                if 'left' in name:
                    side = 'left'
                else:
                    side = 'right'

                t = self.translations[side]

                A = np.zeros((3,3))
                A[:,0] = marker_x_axis.flatten()
                A[:,1] = marker_y_axis.flatten()
                A[:,2] = marker_z_axis.flatten()

                T = self.rotations[side]

                F = np.matmul(A, T)
                
                fingertip_x_axis = F[:,0].flatten()
                fingertip_y_axis = F[:,1].flatten()
                fingertip_z_axis = F[:,2].flatten()

                if (height_above_mounting_surface is None) and (self.default_height_above_mounting_surface is None):
                    # Use the bottom of the rubber cylinder at the
                    # base of the suction cup fingertip, which is also
                    # the top surface of the mounting surface for
                    # other fingertips.
                    fingertip_pos = marker_pos + np.matmul(A, t)
                elif height_above_mounting_surface is not None:
                    fingertip_pos = (marker_pos + np.matmul(A, t))  + (height_above_mounting_surface * fingertip_z_axis)
                elif self.default_height_above_mounting_surface is not None:
                    fingertip_pos = (marker_pos + np.matmul(A, t))  + (self.default_height_above_mounting_surface * fingertip_z_axis)
                    
                fingertips[side] = {'pos': fingertip_pos,
                                    'x_axis': fingertip_x_axis,
                                    'y_axis': fingertip_y_axis,
                                    'z_axis': fingertip_z_axis}


                
        return fingertips

    
    def draw_fingertip_origins(self, fingertips, image, camera_info):

        origins_3d = []
        sides = ['left', 'right']
        for side in sides: 
            f = fingertips.get(side, None)
            if f is not None: 
                origins_3d.append(f['pos'])
        
        origin_pixels = [dh.pixel_from_3d(p, camera_info) for p in origins_3d]

        origins_image = image
        radius = 6
        color = (255, 255, 255)
        thickness = 2
        for p in origin_pixels:
            center = np.round(p).astype(np.int32)
            cv2.circle(origins_image, center, radius, color, thickness) 

            
    def draw_fingertip_frames(self, fingertips, image, camera_info, axis_length_in_m=0.02, draw_origins=True, write_coordinates=False):

        # colors are in BGR format
        sides = ['left', 'right']
        axes = [('x_axis', (0, 0, 255)),
                ('y_axis', (0, 255, 0)),
                ('z_axis', (255, 0, 0))]
        thickness = 3
        origin_radius = 6
                
        for side in sides: 
            f = fingertips.get(side, None)
            if f is not None:
                to_draw = []
                origin = f['pos']
                origin_camera = dh.pixel_from_3d(origin, camera_info)
                origin_image = np.round(origin_camera).astype(np.int32)
                to_draw.append({'type': 'origin',
                                'z': origin[2],
                                'pix': origin_image})

                for axis, color in axes:
                    axis_tip = (axis_length_in_m * (f[axis] - origin)) + origin
                    axis_tip_camera = dh.pixel_from_3d(axis_tip, camera_info)
                    axis_tip_image = np.round(axis_tip_camera).astype(np.int32)
                    to_draw.append({'type': 'axis',
                                    'z': axis_tip[2],
                                    'base_pix': origin_image,
                                    'tip_pix': axis_tip_image,
                                    'color': color})

                to_draw_by_z = sorted(to_draw, key=lambda element: element['z'], reverse=True)

                for d in to_draw_by_z:
                    t = d['type']
                    if (t == 'origin') and draw_origins:
                        color = (255, 255, 255)
                        cv2.circle(image, d['pix'], origin_radius, color, -1, lineType=cv2.LINE_AA)
                    if (t == 'axis'): 
                        cv2.line(image, d['base_pix'], d['tip_pix'], d['color'], thickness, lineType=cv2.LINE_AA)
                    
                if write_coordinates:
                    x,y,z = origin * 100.0
                    text = "{:.1f}, {:.1f}, {:.1f} cm".format(x,y,z)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_size = 0.4

                    text_size = cv2.getTextSize(text, font, font_size, 2)
                    (text_width, text_height), text_baseline = text_size

                    shift = int(2.5 * origin_radius)
                    
                    if side == 'right': 
                        location = origin_image + np.array([shift, int(text_height/2)])
                    else:
                        location = origin_image + np.array([-(text_width + shift), int(text_height/2)])
                    cv2.putText(image, text, location, font, font_size, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, text, location, font, font_size, (255, 255, 255), 1, cv2.LINE_AA)
            
                    
