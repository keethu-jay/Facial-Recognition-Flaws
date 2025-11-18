# Facial-Recognition-Flaws
Group Project for Responsible AI. Testing the resiliency of facial recognition tools given their rising importance in law enforcement.


### Database Used:
Fairface from Hugging Face

### Datasets:
For each of the 7 races there will be a baseline set and 3 attack sets. So 4 total versions.

### Perturbation Attacks Used:
C&W
PGD
FGSM

### Model we are testing/attacking:
FaceNet

### Process:
Fairface has thousands of images sorted by age,gender and race. There are 7 races, 2 genders, and 9 age ranges. There will be 7 datasets created, each with 90 images, 5 images for each gender in each age range.

### Baseline:
FaceNet compares two images of faces and based on their similarity assumes if they are the same person or not. When creating a baseline for our project we want to prove that FaceNet works and can accurately identify each face in every dataset as unique. To do this for each face in each dataset we will compare it with another face of the same gender in the same age range. If the distance calculated is greater than 1.0 FaceNet sees them as different people. 

Ex) compare an asian women aged (20-29) to another asian women aged (20-29) in Facenet. 

### Attack: 
Each of the 7 racial datasets will have 3 attack versions for (C&W, PGD, and FGSM). These attacks minimally perturb the image. The point of the attack is to make FaceNet think that two images it correctly identified as different earlier are now identified as the same image.
