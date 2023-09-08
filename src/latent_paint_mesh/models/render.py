import kaolin as kal
import torch
import numpy as np

class Renderer:
    # from https://github.com/threedle/text2mesh

    def __init__(self, 
                 device, 
                 dim=(224, 224), 
                 interpolation_mode='nearest',
                 lights=torch.tensor([1.0, 1.0, 1.0, 
                                      1.0, 0.0, 0.0, 
                                      0.0, 0.0, 0.0]),
                 ):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera_head = kal.render.camera.generate_perspective_projection(np.pi / 12).to(device)
        camera_body = kal.render.camera.generate_perspective_projection(np.pi / 4).to(device)
        
        # self.FloatTensor = torch.cuda.FloatTensor if self.device != 'cpu' else torch.FloatTensor
        
        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = [ 
            camera_head, # head
            camera_body, # body
        ]
        self.look_at_height = torch.tensor([
            [0.4],  # head
            [-0.3], # body
        ]).to(self.device)
        
        self.dim = dim
        self.background = torch.ones(dim).to(self.device).float()
        
        ### Light
        ### ref: https://github.com/threedle/text2mesh/blob/37d1c8491104b78ee55cd54cd09ab24cb1427714/render.py#L20
        self.lights = lights.unsqueeze(0).to(self.device)
        

    def get_camera_from_view(self, elev, azim, radius=3.0, look_at_height=0.0):
        x = radius * torch.sin(elev) * torch.sin(azim)
        y = radius * torch.cos(elev)
        z = radius * torch.sin(elev) * torch.cos(azim)
        # import pdb;pdb.set_trace()
        # pos = torch.tensor([x, y, z]).unsqueeze(0)
        pos = torch.vstack([x, y, z]).T.to(self.device)
        
        look_at = torch.zeros_like(pos).to(self.device)
        look_at[:, 1] = look_at_height.to(self.device)
        direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).to(self.device)

        camera_transform = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
        return camera_transform

    def compute_vertex_normals(self, faces, face_normals, num_vertices=None):
        r"""Computes normals for every vertex by averaging face normals
        assigned to that vertex for every face that has this vertex.

        Args:
           faces (torch.LongTensor): vertex indices of faces of a fixed-topology mesh batch with
                shape :math:`(\text{num_faces}, \text{face_size})`.
           face_normals (torch.FloatTensor): pre-normalized xyz normal values
                for every vertex of every face with shape
                :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, 3)`.
           num_vertices (int, optional): number of vertices V (set to max index in faces, if not set)

        Return:
            (torch.FloatTensor): of shape (B, V, 3)
        """

        if num_vertices is None:
            num_vertices = int(faces.max()) + 1

        B = face_normals.shape[0]
        V = num_vertices
        F = faces.shape[0]
        FSz = faces.shape[1]
        # print(B, V, F, FSz)

        vertex_normals = torch.zeros((B, V, 3), dtype=face_normals.dtype, device=face_normals.device)
        counts = torch.zeros((B, V), dtype=face_normals.dtype, device=face_normals.device)

        faces = faces.unsqueeze(0)
        fake_counts = torch.ones((B, F), dtype=face_normals.dtype, device=face_normals.device)
        #              B x F          B x F x 3
        # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
        # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
        
        # if B > 1:
        #     import pdb;pdb.set_trace()
            
#             vertex_normals.index_add(1, faces[0, :, 0], face_normals)
#             vertex_normals.index_add(1, faces[0, :, 1], face_normals)
#             vertex_normals.index_add(1, faces[0, :, 2], face_normals)
#             vertex_normals = torch.nn.functional.normalize(vertex_normals, eps=1e-6, dim=1, p=2)

        for i in range(FSz):
            vertex_normals.scatter_add_(1, faces[..., i:i+1].repeat(B, 1, FSz), face_normals)
            counts.scatter_add_(1, faces[..., i].repeat(B, 1), fake_counts)

        counts = counts.clip(min=1).unsqueeze(-1)
        vertex_normals = vertex_normals / counts
        return vertex_normals

    def render_single_view(self, 
                           mesh,
                           face_attributes,
                           elev=0,
                           azim=0,
                           radius=2,
                           look_at_height=0.0,
                           dims=None,
                           is_body=True,
                          ):
        # dims = self.dim
        dims = self.dim if dims is None else dims

        P = 1 if is_body is True else 0
        
        camera_transform = self.get_camera_from_view(
            torch.tensor(elev), 
            torch.tensor(azim), 
            radius,
            self.look_at_height[P]
        ).to(self.device)
        
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), 
            mesh.faces.to(self.device), 
            self.camera_projection[P], 
            camera_transform=camera_transform
        )
        
        vert_normals_world = self.compute_vertex_normals(mesh.faces, face_normals)
        vertex_face_normals = kal.ops.mesh.index_vertices_by_faces(vert_normals_world, mesh.faces)
        
        face_features = [
            face_attributes, 
            torch.ones((1, mesh.faces.shape[0], 3, 1), device='cuda:0'),
            vertex_face_normals,
        ]
        
        image_features, face_idx = kal.render.mesh.rasterize(
            dims[1], 
            dims[0],
            face_vertices_camera[:, :, :, -1],
            face_vertices_image, 
            face_attributes
        )
        
        texture_coords, mask, world_normals = image_features
        
        # mask = (face_idx > -1).float()[..., None]

        return texture_coords.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), world_normals.permute(0, 3, 1, 2)


    def render_single_view_texture(self, 
                                   verts, 
                                   faces, 
                                   uv_face_attr, 
                                   texture_map, 
                                   elev=0,
                                   azim=0,
                                   radius=2,
                                   look_at_height=0.0, 
                                   dims=None, 
                                   white_background=False,
                                   disp=None,
                                   is_body=True,
                                  ):
        dims = self.dim if dims is None else dims
        ## pdb; pdb.set_trace() find grad !!
        # if white_background == False:
        #     import pdb;pdb.set_trace()
        ### Add displacement
        if disp != None:
            verts = verts + disp
            
        P = 1 if is_body is True else 0
                
        # import pdb;pdb.set_trace()
        camera_transform = self.get_camera_from_view(
            elev, 
            azim, 
            radius,
            self.look_at_height[P]
        ).to(self.device)
        
        B = camera_transform.shape[0]
        
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), 
            faces.to(self.device), 
            self.camera_projection[P], 
            camera_transform=camera_transform
        )
        
        vert_normals_world = self.compute_vertex_normals(faces.to(self.device), face_normals)
        vertex_face_normals = kal.ops.mesh.index_vertices_by_faces(vert_normals_world, faces.to(self.device))
        
        ### Latent Paint Rasterization
        # uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
        #     face_vertices_image, uv_face_attr)
        # uv_features = uv_features.detach()
        
        # mask = (face_idx > -1).float()[..., None]
        # image_features = kal.render.mesh.texture_mapping(uv_features, texture_map, mode=self.interpolation_mode)
        # image_features = image_features * mask
        
        # if white_background:
        #     image_features += 1 * (1 - mask)
        # return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)
        

        ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        # import pdb;pdb.set_trace()
        
        face_features = [
            uv_face_attr.repeat(B, 1, 1, 1),
            torch.ones((B, faces.shape[0], 3, 1), device='cuda'),
            vertex_face_normals,
        ]
    
        ### If you have nvdiffrast installed you can change rast_backend to nvdiffrast or nvdiffrast_fwd 
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            dims[1], dims[0],
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_features,
            face_normals[:, :, -1],
            rast_backend='cuda'
        )
            
        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask, world_normals = image_features
        image = kal.render.mesh.texture_mapping(texture_coords, texture_map.repeat(B, 1, 1, 1), mode='bilinear')
        
        # if white_background == False:
        #     import pdb; pdb.set_trace()
        #     from torchvision.transforms import ToPILImage
        #     ToPILImage()(image.permute(0, 3, 1, 2)[0]).save('tm.png')
        #     ToPILImage()(mask.permute(0, 3, 1, 2)[0]).save('tm.png')
        #     ToPILImage()((image * mask).permute(0, 3, 1, 2)[0]).save('tm.png')
        image = image * mask
        
        ## Lighting
        # https://github.com/threedle/text2mesh/blob/37d1c8491104b78ee55cd54cd09ab24cb1427714/render.py#L278
        # if lighting:
        # image_normals = face_normals[:, face_idx].squeeze(0)
        
        # if B > 1:
        #     import pdb;pdb.set_trace()
        
        # curr_light = self.lights.clone()
        # curr_light.requires_grad = True
        
        image_lighting = kal.render.mesh.spherical_harmonic_lighting(world_normals, self.lights).unsqueeze(0)
        image_lighting = image_lighting.clamp(0, 1).permute(1, 0, 2, 3)
        
            # C = world_normals.shape[-1]
            # image_lighting = image_lighting.repeat(1, C, 1, 1).permute(0, 2, 3, 1).to(self.device)
            # image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
            # import pdb;pdb.set_trace()
            # C = image.shape[-1]
            
        # from torchvision import transforms
        # transforms.ToPILImage()(image_lighting[:, 0]).save('t.png')
        # transforms.ToPILImage()(world_normals.permute(0,3,1,2)[1]*0.5+0.5).save('t.png')
        
            # image_lighting = image_lighting.repeat(1, C, 1, 1).permute(0, 2, 3, 1).to(self.device)
            # print(image.shape, image_lighting.shape)
            # image = image * image_lighting
            # image = torch.clamp(image, 0.0, 1.0) ## this is latent feature do not need cliping

        if white_background:
            image += 1 * (1 - mask)
            
        return image.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), world_normals.permute(0, 3, 1, 2), image_lighting

    def render_single_view_texture_lighting(self, 
                                            verts, 
                                            faces, 
                                            uv_face_attr, 
                                            texture_map, 
                                            elev=0, 
                                            azim=0, 
                                            radius=2,
                                            look_at_height=0.0, 
                                            dims=None, 
                                            white_background=False, 
                                            disp=None,
                                            is_body=True,
                                           ):
        dims = self.dim if dims is None else dims
        
        ### Add displacement
        if disp != None:
            verts = verts + disp
        
        P = 1 if is_body is True else 0
        
        camera_transform = self.get_camera_from_view(
                torch.tensor(elev), 
                torch.tensor(azim), 
                radius,
                self.look_at_height[P]
            ).to(self.device)
        
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), 
            faces.to(self.device), 
            self.camera_projection[P], 
            camera_transform=camera_transform
        )
        
        vert_normals_world = self.compute_vertex_normals(faces.to(self.device), face_normals)
        vertex_face_normals = kal.ops.mesh.index_vertices_by_faces(vert_normals_world, faces.to(self.device))
        # ### Perform Rasterization ###
        # Construct attributes that DIB-R rasterizer will interpolate.
        # the first is the UVS associated to each face
        # the second will make a hard segmentation mask
        face_features = [
            uv_face_attr,
            torch.ones((1, faces.shape[0], 3, 1), device='cuda'),
            vertex_face_normals,
        ]
    
        # If you have nvdiffrast installed you can change rast_backend to
        # nvdiffrast or nvdiffrast_fwd 
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            dims[1],
            dims[0],
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_features,
            face_normals[:, :, -1],
            rast_backend='cuda'
        )
            
        # image_features is a tuple in composed of the interpolated attributes of face_attributes
        texture_coords, mask, world_normals = image_features
        # image = kal.render.mesh.texture_mapping(texture_coords, texture_map.repeat(1, 1, 1, 1), mode='bilinear')
        # image = kal.render.mesh.texture_mapping(texture_coords, texture_map.repeat(1, 1, 1, 1), mode='bilinear')
        # image = image * mask
        
        ## Lighting
        # https://github.com/threedle/text2mesh/blob/37d1c8491104b78ee55cd54cd09ab24cb1427714/render.py#L278
        # image_normals = face_normals[:, face_idx].squeeze(0)
        # image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
        image_lighting = kal.render.mesh.spherical_harmonic_lighting(world_normals, self.lights).unsqueeze(0)
        
        C = image_normals.shape[-1] # [1, 512, 512, 3]
        image_lighting = image_lighting.repeat(1, C, 1, 1).permute(0, 2, 3, 1).to(self.device)
        image_lighting = image_lighting.clamp(0, 1)
        
        return image_lighting.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), world_normals.permute(0, 3, 1, 2)
