import matplotlib.pyplot as plt

def image_save(ii,save_path,inputs, one_hot_labels,outputs):    
        a1=inputs.cpu().numpy()                
        c1=one_hot_labels.cpu().numpy()               
        d1 =outputs.cpu().detach().numpy()
                 
        f,ax = plt.subplots(1,5)  
        
        ax[0].imshow(a1[0,0,20,:,:],cmap='gray')
        ax[0].set_title('orig  ')          
        
        ax[1].imshow(c1[0,0,20,:,:],cmap='gray')
        ax[1].set_title('back')
        
        ax[2].imshow(c1[0,1,20,:,:],cmap='gray')
        ax[2].set_title('obj')
        
        ax[3].imshow(d1[0,0,20,:,:],cmap='gray')
        ax[3].set_title('out_b')
        
        ax[4].imshow(d1[0,1,20,:,:],cmap='gray')
        ax[4].set_title('out') 
        plt.savefig(save_path+'{0}.jpg'.format(ii))
    