# -image-generative-AI-model
Open Anaconda Prompt : 
1. conda create -n myenv
2. conda activate myenv
3. conda install -c conda-forge opencv
4. conda install numpy
5. conda install -c conda-forge pillow
6. pip install tensorflow
7. pip install tensorflow-gpu
8. conda install jupyter
9. jupyter-notebook

        import os
        import cv2
        import numpy as np
        from PIL import Image
        from tensorflow.keras.applications import VGG19
        from tensorflow.keras.applications.vgg19 import preprocess_input
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.models import Model
        base_model = VGG19(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv2').output)
        
        image_path = r'C:\Users\Anjali Gidwani\Downloads\Marble_AI_DS_Intern_Assignment -20240427T135245Z-001\Marble_AI_DS_Intern_Assignment\input_product'
        image_path
        def preprocess_image(img_path):
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            return img_array
        product_image_path = r'C:\Users\Anjali Gidwani\Downloads\Marble_AI_DS_Intern_Assignment -20240427T135245Z-001\Marble_AI_DS_Intern_Assignment\input_product'
        product_image_path
        background_image_path = r'C:\Users\Anjali Gidwani\Downloads\Marble_AI_DS_Intern_Assignment -20240427T135245Z-001\Marble_AI_DS_Intern_Assignment\Background'
        background_image_path
        def generate_output_images(product_image_path, background_image_path):
            # Load and preprocess product and background images
            product_img = cv2.imread(product_image_path)
            product_img_rgb = cv2.cvtColor(product_img, cv2.COLOR_BGR2RGB)
        
        background_img = cv2.imread(background_image_path)
        background_img_rgb = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        
        # Resize background image to match product image size
        background_img_resized = cv2.resize(background_img_rgb, (product_img_rgb.shape[1], product_img_rgb.shape[0]))
        
        # Create a mask for blending
        mask = 255 * np.ones(product_img_rgb.shape, product_img_rgb.dtype)
        
        # Compute center position for blending
        center = (product_img_rgb.shape[1] // 2, product_img_rgb.shape[0] // 2)
        
        # Perform seamless cloning for blending
        try:
            blended_output = cv2.seamlessClone(product_img_rgb, background_img_resized, mask, center, cv2.NORMAL_CLONE)
        except cv2.error as e:
            print("Error during blending:", e)
            return
    
        # Save the generated output image
        output_path = r'C:\Users\Anjali Gidwani\Downloads\Marble_AI_DS_Intern_Assignment -20240427T135245Z-001\Marble_AI_DS_Intern_Assignment\output_image' + os.path.basename(product_image_path) + '_' + os.path.basename(background_image_path)
        cv2.imwrite(output_path, cv2.cvtColor(blended_output, cv2.COLOR_RGB2BGR))
    
        product_images_folder = r'C:\Users\Anjali Gidwani\Downloads\Marble_AI_DS_Intern_Assignment -20240427T135245Z-001\Marble_AI_DS_Intern_Assignment\input_product'
        product_images_folder
        background_images_folder = r'C:\Users\Anjali Gidwani\Downloads\Marble_AI_DS_Intern_Assignment -20240427T135245Z-001\Marble_AI_DS_Intern_Assignment\Background'
        background_images_folder
        output_folder = r'C:\Users\Anjali Gidwani\Downloads\Marble_AI_DS_Intern_Assignment -20240427T135245Z-001\Marble_AI_DS_Intern_Assignment\output_image'
        if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        for product_image in os.listdir(product_images_folder):
        for background_image in os.listdir(background_images_folder):
            generate_output_images(os.path.join(product_images_folder, product_image), os.path.join(background_images_folder, background_image))
