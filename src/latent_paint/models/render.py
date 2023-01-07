import kaolin as kal
import torch
import numpy as np

class Renderer:
    # from https://github.com/threedle/text2mesh

    def __init__(self, device, dim=(224, 224), interpolation_mode='nearest'):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()

    @staticmethod
    def get_camera_from_view(elev, azim, r=3.0, look_at_height=0.0):
        x = r * torch.sin(elev) * torch.sin(azim)
        y = r * torch.cos(elev)
        z = r * torch.sin(elev) * torch.cos(azim)

        pos = torch.tensor([x, y, z]).unsqueeze(0)
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height
        direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
        return camera_proj


    def render_single_view(self, mesh, face_attributes, elev=0, azim=0, radius=2, look_at_height=0.0):
        dims = self.dim

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height).to(self.device)
        
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        mask = (face_idx > -1).float()[..., None]

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)


    def render_single_view_texture(self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2,
                                   look_at_height=0.0, dims=None, white_background=False, disp=None):
        dims = self.dim if dims is None else dims
        ## pdb; pdb.set_trace() find grad !!
        # if white_background == False:
        #     import pdb;pdb.set_trace()
            
        if disp != None:
            verts = verts + disp
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height).to(self.device)
        
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)
        
        ### Latent Paint Rasterization
        # uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
        #     face_vertices_image, uv_face_attr)
        # # uv_features = uv_features.detach()

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        face_attributes = [
            uv_face_attr,
            torch.ones((1, faces.shape[0], 3, 1), device='cuda')
        ]
    
        # If you have nvdiffrast installed you can change rast_backend to
        # nvdiffrast or nvdiffrast_fwd 
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            dims[1], dims[0], face_vertices_camera[:, :, :, -1],
            face_vertices_image, face_attributes, face_normals[:, :, -1],
            rast_backend='cuda')
        
        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask = image_features
        image = kal.render.mesh.texture_mapping(texture_coords, texture_map.repeat(1, 1, 1, 1), mode='bilinear')
        image = torch.clamp(image * mask, 0., 1.)

        # mask = (face_idx > -1).float()[..., None]
        # image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
        # image_features = image_features * mask
        # image = image * mask
        
        # import pdb;pdb.set_trace()
        
        if white_background:
            # image_features += 1 * (1 - mask)
            image += 1 * (1 - mask)

        # return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)
        return image.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)
