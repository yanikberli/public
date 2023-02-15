import numpy as np
import scipy
import nibabel as nib
import glob
import os
import pandas as pd
from tqdm import tqdm
file_directory3D= '/home/user/Downloads/Yanik_3D_only/nii/'
file_directory2D= '/home/user/Downloads/Yanik_3D_only/5px_axial'
filelist3D = glob.glob(file_directory3D+"*/*/lesions2.nii.gz")
#lists for parameters to save
F1_list=[]
col6="F1 score"
tpa_list=[]
col5="Nr. of true positives for positive predicitive value"
tpg_list=[]
col4="Nr. of true positives in sensitivity"
labels2D=[]
col3="Number of 2D Labels"
labels3D=[]
col2="Number of 3D Labels"
pat_nr=[]
col1="Pat. number"
pbar=tqdm(total=len(filelist3D))
for file in filelist3D:
    pbar.update(1)
    #getting Data
    orig_3Dimage=nib.load(file)
    Arr3D=orig_3Dimage.get_fdata()
    file2= os.path.join(file_directory2D,*file.split("/")[-3:])
    if os.path.exists(file2):
            orig_2Dimage=nib.load(file2)
            Arr2D=orig_2Dimage.get_fdata()
    else:
                num_features2D="error"
                continue
    #check if same pat
    pat_name= file.split("/")[-2:-1]
    pat_name2=file2.split("/")[-2:-1]
    if not pat_name==pat_name2:
            pat_name= "error"
    #downsampling 3D image
    thickness=Arr3D.shape[2]//Arr2D.shape[2]
    downsampled3D=np.zeros([Arr2D.shape[0], Arr2D.shape[1], Arr2D.shape[2]])
    for i in range(Arr2D.shape[2]):
        selected_slices=Arr3D[:, :, i*thickness:(i+1)*thickness]
        combined_slices=selected_slices[:, :, 0]+selected_slices[:, :, 1]+selected_slices[:, :, 2]
        downsampled3D[:, :, i]=combined_slices
    #apply label function
    labeled_array3D, num_features3D = scipy.ndimage.label(downsampled3D)
    labeled_array2D, num_features2D = scipy.ndimage.label(Arr2D)
    #calculate positive predictive vlaue
    TPa = 0
    for i in range (num_features3D):
        lesions3 = (labeled_array3D == i).astype(int)
        overlap = np.logical_and(lesions3, labeled_array2D).astype(int)
        num_tp3D = (overlap == 1).sum()
        num_ground_truth= (lesions3==1).sum()
        relative_to_groundtruth = num_tp3D/num_ground_truth
        if relative_to_groundtruth >= 0.3:
                TPa += 1
    #calculate sensitivity   
    TPg = 0
    for f in range (num_features2D):
        lesions2 = (labeled_array2D == f).astype(int)
        overlap = np.logical_and(lesions2, labeled_array3D).astype(int)
        num_tp2D = (overlap == 1).sum()
        num_segmented= (lesions2==1).sum()
        relative_to_segmented = num_tp2D/num_segmented
        if relative_to_segmented >= 0.3:
                TPg += 1
    #claculate F1 score
    try:
        SeL = TPg/num_features3D
        PL = TPa/num_features2D
        F1=2*((SeL*PL)/(SeL+PL))
    except ZeroDivisionError:
        print("Zero division error")
        F1='error'
        pass
    #add result to the lists
    pat_nr.append(pat_name)
    labels3D.append(num_features3D)
    labels2D.append(num_features2D)
    tpg_list.append(TPg)
    tpa_list.append(TPa)
    F1_list.append(F1)  
pbar.close()          
a={col1:pat_nr,col2:labels3D,col3:labels2D, col4:tpg_list, col5:tpa_list, col6:F1_list}
df=pd.DataFrame.from_dict(a, orient='index')
df = df.transpose()
df.to_excel('label_comparison.xlsx', sheet_name='sheet1', index=False)
