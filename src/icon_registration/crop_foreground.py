import torch                                                                                                            
import torch.nn.functional as F                                                                                         
import itk     
import numpy as np
                                                                                                                        
import torch                                                                                                            
                                                                                                                        
def torch_crop_foreground(tensor: torch.Tensor, additional_crop_pixels: int = 0) -> torch.Tensor:                       
    """                                                                                                                 
    Crops a PyTorch tensor to its foreground by removing uniform boundary regions.                                      
                                                                                                                        
    This function finds the first non-uniform slice from each direction and crops the tensor                            
    accordingly. It works with both 2D and 3D tensors.                                                                  
                                                                                                                        
    Args:                                                                                                               
        tensor (torch.Tensor): Input tensor of shape (H, W) for 2D or (D, H, W) for 3D                                  
        additional_crop_pixels (int, optional): Additional pixels to crop from each boundary.                           
            Defaults to 0.                                                                                              
                                                                                                                        
    Returns:                                                                                                            
        torch.Tensor: Cropped tensor containing only the foreground region                                              
                                                                                                                        
    Raises:                                                                                                             
        ValueError: If input tensor is not 2D or 3D                                                                     
                                                                                                                        
    Example:                                                                                                            
        >>> x = torch.zeros((100, 100))                                                                                 
        >>> x[25:75, 25:75] = 1                                                                                         
        >>> cropped = torch_crop_foreground(x)                                                                          
        >>> print(cropped.shape)                                                                                        
        torch.Size([50, 50])                                                                                            
    """                                                                                                                 
    if not (2 <= tensor.dim() - 2 <= 3):                                                                                
        raise ValueError("Input tensor must be 2D or 3D")                                                               
                                                                                                                        
    def first_nonequal(fn):                                                                                             
        i = 0                                                                                                           
        while True:                                                                                                     
            slice_tensor = fn(i)                                                                                        
            if not torch.all(slice_tensor == slice_tensor.flatten()[0]):                                                
                return i + additional_crop_pixels                                                                       
            i += 1                                                                                                      
                                                                                                                        
    if tensor.dim() - 2 == 2:                                                                                           
        # Find boundaries for 2D tensor                                                                                 
        upper_1 = first_nonequal(lambda i: tensor[:, :, :, tensor.shape[3] - 1 - i])                                    
        upper_2 = first_nonequal(lambda i: tensor[:, :, tensor.shape[2] - 1 - i])                                       
                                                                                                                        
        lower_1 = first_nonequal(lambda i: tensor[:, :, :, i])                                                          
        lower_2 = first_nonequal(lambda i: tensor[:, :, i])                                                             
                                                                                                                        
        # Crop the tensor                                                                                               
        return tensor[:, :, lower_2:tensor.shape[0] - upper_2,                                                          
                     lower_1:tensor.shape[1] - upper_1]                                                                 
                                                                                                                        
    else:  # 3D case                                                                                                    
        # Find boundaries for 3D tensor                                                                                 
        upper_1 = first_nonequal(lambda i: tensor[:, :, :, :, tensor.shape[4] - 1 - i])                                 
        upper_2 = first_nonequal(lambda i: tensor[:, :, :, tensor.shape[3] - 1 - i])                                    
        upper_3 = first_nonequal(lambda i: tensor[:, :, tensor.shape[2] - 1 - i])                                       
                                                                                                                        
        lower_1 = first_nonequal(lambda i: tensor[:, :, :, :, i])                                                       
        lower_2 = first_nonequal(lambda i: tensor[:, :, :, i])                                                          
        lower_3 = first_nonequal(lambda i: tensor[:, :, i])                                                             
                                                                                                                        
        # Crop the tensor                                                                                               
        return tensor[:, :, lower_3:tensor.shape[2] - upper_3,                                                          
                     lower_2:tensor.shape[3] - upper_2,                                                                 
                     lower_1:tensor.shape[4] - upper_1]                                                                                                      
                                                                                                                                                                                                                                          
def itk_crop_foreground(image, additional_crop_pixels=0):                                                               
                                                                                                                        
    def first_nonequal(fn):                                                                                             
        i = 0                                                                                                           
        while True:                                                                                                     
            slice = fn(i)                                                                                               
            if not np.all(slice == slice.flat[0]):                                                                      
                return i + additional_crop_pixels                                                                       
            i += 1                                                                                                      
                                                                                                                        
    arr = itk.GetArrayFromImage(image)                                                                                  
                                                                                                                        
                                                                                                                        
    if len(arr.shape) == 2:                                                                                             
                                                                                                                        
        upper_1 = first_nonequal(lambda i: arr[:, arr.shape[1] - 1 - i])                                                
        upper_2 = first_nonequal(lambda i: arr[arr.shape[0] - 1 - i])                                                   
                                                                                                                        
        upper = (upper_1, upper_2)                                                                                      
                                                                                                                        
        lower_1 = first_nonequal(lambda i: arr[:, i])                                                                   
        lower_2 = first_nonequal(lambda i: arr[i])                                                                      
                                                                                                                        
        lower = (lower_1, lower_2)                                                                                      
    if len(arr.shape) == 3:                                                                                             
        upper_1 = first_nonequal(lambda i: arr[:, :, arr.shape[2] - 1 - i])                                             
        upper_2 = first_nonequal(lambda i: arr[:, arr.shape[1] - 1 - i])                                                
        upper_3 = first_nonequal(lambda i: arr[arr.shape[0] - 1 - i])                                                   
                                                                                                                        
        upper = (upper_1, upper_2, upper_3)                                                                             
                                                                                                                        
        lower_1 = first_nonequal(lambda i: arr[:, :, i])                                                                
        lower_2 = first_nonequal(lambda i: arr[:, i])                                                                   
        lower_3 = first_nonequal(lambda i: arr[i])                                                                      
                                                                                                                        
        lower = (lower_1, lower_2, lower_3)                                                                             
                                                                                                                
    crop_filter = itk.CropImageFilter[type(image), type(image)].New()                                                   
    crop_filter.SetUpperBoundaryCropSize(upper)                                                                         
    crop_filter.SetLowerBoundaryCropSize(lower)                                                                         
    crop_filter.SetInput(image)                                                                                         
    crop_filter.Update()                                                                                                
                                                                                                                        
    output = crop_filter.GetOutput()                                                                                  
                                                                                                                            
    itk.imwrite(output, "/playpen/tgreer/tmp_img.nrrd")                                                                 
    output = itk.imread("/playpen/tgreer/tmp_img.nrrd")                                                                 
                                                                                                                        
    return output                                                                                                       
                     
