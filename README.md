# flagid 🏳️
AI country flag identification neural network for detecting country flags in photos, based on efficientnet08.
Fast 92% accurate AI flag identifier, trained on world flag dataset. 

## Features:
- VERY easy to use
- IDs blurry, rotated flags and images of flags in the wild
- Fast and runs on dogshit PCs

Looking for contributions to read me with image examples

To train:
https://www.kaggle.com/code/mreowie/country-flag-identifier/edit

## Usage instructions
1. 
```
git clone https://github.com/MintOcha/flagid
cd flagid
```

2. Install requirements

```
pip install torch torchvision pillow numpy requests opencv-python selenium webdriver-manager scikit-learn tqdm matplotlib
```

3. Run the script! Set test_image_path = "image.png" to be your image name and run use_model.py

I hope i made this easy to use :)

<img width="1832" height="915" alt="image" src="https://github.com/user-attachments/assets/231633de-d662-4d54-bb4b-ee74bd339bbf" />

Examples:
<img width="1560" height="1393" alt="image" src="https://github.com/user-attachments/assets/2c0c9f1c-5b0b-47b1-83dc-007cd76f679d" />

results!! 
```
    accuracy                           0.92      5154
   macro avg       0.92      0.92      0.92      5154
weighted avg       0.92      0.92      0.92      5154
```
