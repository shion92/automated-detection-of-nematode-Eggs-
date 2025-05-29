# Labeling Instructions for Training

Follow these steps to label images for training for nematode egg detection:

## Step-by-Step Instructions

1. **Launch the Labeling Tool**  
    Open a terminal and run the following command to launch the labeling tool:  
    ```bash
    labelImg
    ```
    or 
    ```bash
    labelme
    ```

2. **Select the Image Folder**  
    In the user interface, choose the folder containing the images you want to label.

3. **Start Labeling**  
    - Click the **'Create Polygons'** or **'Create RectBox'** button to begin labeling.  
    - Carefully draw polygons or bouding boxes around the nematode eggs in the images.  
    - Use zoom (Ctrl + scroll up) for better precision.

4. **Maintain Consistent Labels**  
    - Assign the label **`nematode egg`** to all identified nematode eggs.  
    - Ensure the label name is consistent across all images, as inconsistent labels will be treated as separate classes by the model.

5. **Save the Annotations**  
    - After labeling, click the **'Save'** button.  
    - Annotation files will be saved automatically in `.json` format if using labelme to the same folder as the images. 
    - If using LabelImg, annotations will be saved in `.xml` format to the folder you specify.

## Notes
- Consistency in labeling is crucial for model training.  
- Ensure all polygons are tightly drawn around the nematode eggs to improve model performance.
- Keep filenames consistent with the image name (e.g., Image_01.xml for Image_01.tif).
