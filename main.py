#import libraries
import os
from helper import greyscale, gradient_operation, save_image, HOG_feature,KNN_classifier, print_HOG
from imageio import imread
from pathlib import Path


#create list of directories
training_directories=["Training images (Neg)","Training images (Pos)"]
test_directories=["Test images (Neg)","Test images (Pos)"]

#set output directory
outputPath = "Output"
if os.path.exists("Output")==False:
    os.mkdir("Output")
    print("Output Folder Created")


#to file names and original classification
training_fname=[]
test_fname=[]

#to store image
training_set = []
test_set = []

#Populate training and Test Sets
for i,folder in enumerate(training_directories):
    for file in os.listdir(folder):
        file_path=os.path.join(folder,file)
        
        #add filename and classification
        training_fname.append([Path(file).stem,bool(i)])
        
        #read image and  convert to greyscale
        image=imread(file_path)
        image=greyscale(image)
        #add to training set
        training_set.append(image)
        
for i,folder in enumerate(test_directories):
    for file in os.listdir(folder):
        file_path=os.path.join(folder,file)
        test_fname.append([Path(file).stem,bool(i)])
        
        #read file and convert to greyscale
        image=imread(file_path)
        image=greyscale(image)
        #add to test set
        test_set.append(image)


#Initialize gradient lists
training_gradients = []
test_gradients = []


#Calculate Gradient angles and magnitude for each image
for image in training_set:
    G,angles = gradient_operation(image)
    training_gradients.append([G,angles])

for i,image in enumerate(test_set):
    G,angles = gradient_operation(image)
    #save gradients of test_files
    save_image(G,outputPath,test_fname[i][0])
    
    test_gradients.append([G,angles])

#Compute HOG feature for each image

#initialize HoG Feature arrays for training and test images
training_HOG = []
test_HOG = []

for i in range(len(training_gradients)):
    #fetch Gradient Magnitudes and Angles for the image
    G, angles = training_gradients[i]
    #compute and append the HoG feature vector
    training_HOG.append(HOG_feature(G,angles))
    
for i in range(len(test_gradients)):
    #Get Gradient magnitude and angles for the image
    G , angles = test_gradients[i]
    
    #compute and append HOG vector
    test_HOG.append(HOG_feature(G,angles))    


#List of file names for which HOG has to be printed
hog_training_fname = ['crop001028a', 'crop001030c', '00000091a_cut']
hog_test_fname = ['crop001278a', 'crop001500b', '00000090a_cut']


#Print HoG vector
print_HOG(hog_training_fname, training_fname, training_HOG, outputPath)
print_HOG(hog_test_fname, test_fname, test_HOG, outputPath)


#Classify the Test Images using the 3-Nearest Neighbor Classifier
test_NN = KNN_classifier(training_HOG, test_HOG, 3,training_fname,test_fname)

#%%
count=0
for index,file in enumerate(test_NN):
    
    print(file)
    print(test_NN[file]['neighbours'])
    print("Prediction: ",test_NN[file]['classification']," Label: ",test_fname[index][1])
    
    if test_NN[file]['classification']== test_fname[index][1]:
        count+=1

print("\nAcurracy:",count*10,"%")
    