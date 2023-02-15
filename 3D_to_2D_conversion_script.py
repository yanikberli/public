import os
import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm

#getting data
file_dir= "/home/user/Downloads/Yanik_3D_only/nii/"
fileglob= '/home/user/Downloads/Yanik_3D_only/nii/*/*/brain.nii.gz'
filelist = glob.glob(fileglob, recursive= True)

#compression function
def compress(thickness, plane):
    ''' description:
    The function takes nii image files from a directory, loads them and compresses them in a given axis to a given thickness bevore saving it as a nii.gz image to a directory.
    Parameters:
    thickness: Type: int; defines the thickness of compressing slices
    plane: Type: string; defines the plane in wich the image is to be compressed possible parameters:
    for x-Axis: 'x', 'sag' or 'sagital'
    for y-Axis: 'y', 'cor' or 'coronal'
    for z-Axis: 'z', 'trans' or 'transversal'
    '''
    
    new_dir=os.path.join(file_dir+ str(thickness)+"px_"+plane)
    if not os.path.exists(os.path.join(file_dir+"2D_Files")):
        os.mkdir(new_dir)
    pbar=tqdm(total=len(filelist))
    for file in filelist:
        #folder and filename management
        pbar.update(1)
        file_path=os.path.split(file)
        file_name= ""
        for x in file.split('/')[-2:-1]:
            file_name+=x
        file_folder= file.split('/')[-3:-2]
        folderdir = os.path.join(new_dir, *file_folder)
        if not os.path.exists(folderdir):
            os.mkdir(folderdir)
        orig_image=nib.load(file)
        image_data=orig_image.get_fdata()
        if plane in ['z', 'trans', 'transversal', 'axial']:
        #z axis/transversal plane
            a=orig_image.header.get_zooms()[0]
            b=orig_image.header.get_zooms()[1]
            c=thickness*orig_image.header.get_zooms()[2]
            zoom=np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, thickness, 0],
                [0, 0, 0, 1]])
            zaxis=image_data.shape[2]
            adj_range=zaxis//thickness
            arr=np.zeros([image_data.shape[0], image_data.shape[1], adj_range])
            for i in range(adj_range):
                selected_slices=image_data[:, :, i*thickness:(i+1)*thickness]
                compressed_slices= np.max(selected_slices, axis=2)
                arr[:, :, i]=compressed_slices
        elif plane in ['y', 'cor', 'coronal']:
        #y axis/coronal plane
            a=orig_image.header.get_zooms()[0]
            b=thickness*orig_image.header.get_zooms()[1]
            c=orig_image.header.get_zooms()[2]
            zoom=np.array([[1, 0, 0, 0],
                [0, thickness, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
            yaxis=image_data.shape[1]
            adj_range=yaxis//thickness
            arr=np.zeros([image_data.shape[0],adj_range, image_data.shape[2]])
            for i in range(adj_range):
                selected_slices=image_data[:, i*thickness:(i+1)*thickness, :]
                compressed_slices= np.max(selected_slices, axis=1)
                arr[:, i, :]=compressed_slices
        elif plane in ['x', 'sag', 'sagital']:
        #x axis/sagital plane
            a=thickness*orig_image.header.get_zooms()[0]
            b=orig_image.header.get_zooms()[1]
            c=orig_image.header.get_zooms()[2]
            zoom=np.array([[thickness, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
            xaxis=image_data.shape[0]
            adj_range=xaxis//thickness
            arr=np.zeros([adj_range, image_data.shape[1], image_data.shape[2]])           
            for i in range(adj_range):
                selected_slices=image_data[i*thickness:(i+1)*thickness, :, :]
                compressed_slices= np.max(selected_slices, axis=0)
                arr[i, :, :]=compressed_slices
        else: print('invalid')

        #create new array and safe
        #print(np.array(arr).shape)
        final_arr= np.stack(arr)
        
        ni_img = nib.Nifti1Image(final_arr, orig_image.affine)
        adj_aff=np.matmul(zoom,ni_img.affine)
        adj_spacing = ni_img.header
        adj_spacing['pixdim'][1:4]  = [a, b, c]
        final_img = nib.Nifti1Image(final_arr, adj_aff, adj_spacing)
        nib.save(final_img, os.path.join(folderdir,file_name+'.nii'))
    pbar.close()     
compress(3, 'cor')
