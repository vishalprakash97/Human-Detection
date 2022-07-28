import numpy as np
from PIL import Image as im
import os
from scipy.stats import mode

# Convolution Function
def convolve(image,filter):
    #Takes sub_image and filter,returns weighted sum
    return (image*filter).sum()

# Save Image Function
def save_image(array,output_path,name):
    #Takes array, and filename as input, saves array as image with name as filename
    
    #convert numpy array to image
    image=im.fromarray(array.astype(np.uint8)) 
    
    #save image at output location
    image.save(os.path.join(output_path,name+".bmp")) 

#Convert to Greyscale
def greyscale(image):
    #Extract Image shape
    height, width, channel = image.shape
    imageGray = np.zeros([height, width])
    #convert each pixel to greyscale
    for i in range(height):
        for j in range(width):
            imageGray[i][j] = np.round(0.299*image[i][j][0] + 0.587*image[i][j][1] + 0.114*image[i][j][2]).astype(int)
    
    return imageGray  
  
# Gradient Calculation
def gradient_operation(image):
    '''Takes image as input, Returns Horizonal and Vertical Gradient, Gradient magnitude and angles''' 
    #Define Prewitt's Gradient operators
    Px=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    Py=np.array([[[1,1,1],[0,0,0],[-1,-1,-1]]])
    #Extract size of image and gradient operator
    n,m=image.shape
    n_g,m_g=Px.shape
    #initialize output arrays to zero
    angles=np.zeros([n,m])
    Gx=np.zeros([n,m])
    Gy=np.zeros([n,m])
    
    mid_ng=int(n_g/2)#find mid-point of operator
    mid_mg=int(m_g/2)
    #Apply convolution for each 3x3 sub-image
    for i in range(mid_ng,n-mid_ng):
        for j in range(mid_mg,m-mid_mg):
            #Extract 3x3 sub-image
            sub_image=image[i-mid_ng:i+mid_ng+1,j-mid_mg:j+mid_mg+1]
            #Calculate horizontal,vertical gradient
            Gx[i,j]=convolve(sub_image,Px)
            Gy[i,j]=convolve(sub_image,Py)
            #Calculate inverse tangent of Gy/Gx
            if(Gx[i,j]!=0):
                angles[i,j]=np.degrees(np.arctan2(Gy[i,j],Gx[i,j]))
                #Converting negative angles so that angles is in range [0,360]
                if angles[i,j]<0:
                    angles[i,j]+=360

    #Gradient Calculation and Normalisation
    G=np.sqrt(Gx**2 + Gy**2)
    G=np.round(G/(3*np.sqrt(2)))
    
    return G,angles

def find_bins(num):
    #Calulates bin_centers between which magnitude is divided
    bin_start=num-num%20
    if(num-bin_start)>10:
        b1=bin_start+10
    else:
        b1=(bin_start-10)%180
    b2=(b1+20)%180
    return b1,b2

def find_weights(a):
    #Calculate weights based on bin-centers
    b1,b2=find_bins(a)
    w1=1-(abs(a+180-b1)%180)/20
    w2=1-w1
    return b1,b2,w1,w2

#HOG Feature Calculation
def HOG_feature(magnitude,angles):
    
    h,w=magnitude.shape
    cell_histograms=[]
    #First compute histogram for each cell
    for i in range(0,h,8):
        row_hist=[]
        for j in range(0,w,8):
            #Extract gradients and angles for each cell
            cell_gradients = magnitude[i:i+8,j:j+8].flatten()
            cell_angles = angles[i:i+8,j:j+8].flatten()
            cell_angles=cell_angles%180
            
            hist=np.zeros(9)
            bin_centers=np.array([a+10 for a in range(0,180,20)])
            #Calclulate histogram for bin
            for k in range(64):
                #Find bin centers and weightage
                b1,b2,w1,w2=find_weights(cell_angles[k])
                #increment histogram
                hist[bin_centers==b1]+=w1*cell_gradients[k]
                hist[bin_centers==b2]+=w2*cell_gradients[k]
            
            row_hist.append(hist)
        cell_histograms.append(row_hist)
    
    cell_histograms=np.array(cell_histograms)    
    m_cells,n_cells, _ = cell_histograms.shape

    block_vector = []

    for i in range(m_cells-1):
        for j in range(n_cells-1):
            #Derive HOG for block
            block = cell_histograms[i:i+2, j:j+2].flatten()
            if (np.linalg.norm(block,2)!= 0):
                #normalize the HOG
                block = block/np.linalg.norm(block,2)
            block_vector.append(block)
    block_vector = np.array(block_vector)
    HOG = block_vector.flatten()
    
    return HOG

#Histogram Similarity
def histogram_intersection(test, train):
    return (np.minimum(test, train).sum()) / train.sum()

#K Nearest Neighbours Classifier
def KNN_classifier(training_HOG, test_HOG, k, train_names,test_names):
    
    #Initialize Nearest Neighbors dictionary
    test_NN = {}

    #Compute and fill the histogram intersection of each test image with each training image
    for i in range(len(test_HOG)):
        test_vector = test_HOG[i]
        test_fname = test_names[i][0]
        hist_similarity=[]
        for j in range(len(training_HOG)):
            train_vector = training_HOG[j]
            #compute histogram similarity
            hist_similarity.append(histogram_intersection(test_vector,train_vector))       
            
        
        
        test_NN[test_fname]={'neighbours':[]}
        #Extract indices of k-best neighbours
        indices=np.argsort(hist_similarity)[::-1][:k]
        labels=[]
        for index in indices:
            value=hist_similarity[index]
            fname = train_names[index][0]
            label=train_names[index][1]
            #append filename,similarity,and label
            test_NN[test_fname]['neighbours'].append([fname,value,label])
            labels.append(label)
            
        #find mode of the labels of neigbhours
        classification=mode(labels)[0][0]
        test_NN[test_fname]['classification']=classification

    return test_NN        

#Print HOG to file
def print_HOG(file_names,ref_names,ref_HOG, outputPath):
    for fname in file_names:
        names=[i[0] for i in ref_names ]
        #Find index of file
        index = names.index(fname)
        #Get HoG feature
        hog = ref_HOG[index]
        #Write to File
        with open(os.path.join(outputPath, 'HOG_' + fname + '.txt'), 'w') as f:
            for feature in hog:
                f.write(str(feature)+'\n')





