
import SimpleITK as sitk

def storeVolume_mhd(model_filepath,data,data_filepath,data_type):
    sitk_img = sitk.GetImageFromArray(data, isVector=False)
    sitk_img.dtype=data_type
    
    
    img_orig_model =sitk.ReadImage(model_filepath)

    sitk_img.SetSpacing(img_orig_model.GetSpacing())
    sitk_img.SetOrigin(img_orig_model.GetOrigin())
    sitk.WriteImage(sitk_img, data_filepath)
    print('write file '+data_filepath+ ' successful!')
            
def loadVolume_mhd(filepath):
    image = sitk.ReadImage(filepath)
    data = sitk.GetArrayFromImage(image) # z, y, x  
    # data = image.GetOrigin() # x, y, z
    print("read "+filepath+" successful!")
    return data


