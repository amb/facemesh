# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "Facemesh",
    "category": "Paint",
    "author": "ambi",
    "description": "Face-off",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "Image Editor > Side Panel > Image",
    "warning": "EXPERIMENTAL",
}

import bpy
import bmesh
import math
import mathutils

# requires:
# opencv-python
# mediapipe
# protobuf==3.19.0
# matplotlib


# from .blender_pip import Pip
# Pip.install("")

# import sys
# from os.path import dirname

# file_dirname = dirname(__file__)
# if file_dirname not in sys.path:
#     sys.path.append(file_dirname)


import mediapipe as mp

# from . import mediapipe as mp
# from .mediapipe.python import *
# from .mediapipe.python import solutions as solutions
import numpy as np
from . import utils


class AMB_OT_FaceOff(bpy.types.Operator):
    bl_idname = "faceoff.execute"
    bl_label = "Copy face from image into 3D"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # cc = 0
        # for p in bpy.context.selected_objects[0].data.polygons:
        #     cc += 1
        #     print([i for i in p.vertices], end=", ")
        #     if cc == 4:
        #         cc = 0
        #         print()

        image = utils.get_area_image(context)
        if image is None:
            self.report({"WARNING"}, "Unable to load image")
            return {"CANCELLED"}
        ndimage = np.swapaxes(np.uint8(utils.image_to_ndarray(image) * 255.0), 0, 1)[..., :3]

        # Run MediaPipe Face Mesh.
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True,
        ) as face_mesh:
            results = face_mesh.process(ndimage)
            if results.multi_face_landmarks is None:
                self.report({"WARNING"}, "Unable to find facial landmarks")
                return {"CANCELLED"}

        # Draw annotations of landmarks
        # ret_image = np.empty((*annotated_image.shape[:2], 4), dtype=np.float32)
        # ret_image[..., :3] = np.float32(np.swapaxes(annotated_image, 0, 1)[..., :3]) / 255.0
        # ret_image[..., 3] = 1.0
        # utils.ndarray_to_image(image, ret_image)
        # bpy.ops.image.invert(invert_r=False, invert_g=False, invert_b=False)

        # Create mesh
        bm = bmesh.new()
        rs = [b for b in results.multi_face_landmarks][0].landmark
        for k in range(468):
            c_point = (rs[k].y - 0.5, rs[k].z - 0.5, rs[k].x - 0.5)
            bm.verts.new(c_point)

        bm.verts.ensure_lookup_table()
        bm.verts.index_update()

        fm_t = utils.QUAD_TESSELLATION
        for i in range(0, len(fm_t)):
            nf = bm.faces.new((bm.verts[j] for j in fm_t[i]))
            nf.smooth = True

        # Baked UV
        uv_layer = bm.loops.layers.uv.new()
        for face in bm.faces:
            for loop in face.loops:
                v = bm.verts[loop.vert.index]
                loop[uv_layer].uv = (v.co.x + 0.5, v.co.z + 0.5)

        me = bpy.data.meshes.new("Faceoff mesh")
        bm.to_mesh(me)
        bm.free()
        obj = bpy.data.objects.new("Faceoff object", me)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        return {"FINISHED"}

    # def draw(self, context):
    #     col = self.layout.column()
    #     box = col.box()
    #     row = box.row()
    #     row.label(text="Test label")

    # bl_category = "Image"
    # bl_regiontype = "UI"
    # bl_spacetype = "IMAGE_EDITOR"


classes = [AMB_OT_FaceOff]


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in classes[::-1]:
        bpy.utils.unregister_class(c)
