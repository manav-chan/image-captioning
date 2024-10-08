# Image Captioning
   - Defined and trained an Image Caption Generator model on the Flickr 8K image dataset and achieved a BLEU-1 score of 0.546282 and BLEU-2 score of 0.320683.
   - Used pretrained VGG16 CNNd model for image feature extraction after removing the last classification layer.
   - Implemented Long Short-Term Memory Recurrent Neural Network layer in the model.
   - Created a web server for hosting the web application using Python Flask.
   - Created GUI using HTML, Javascript and Bootstrap.

   ## BLEU Score
   - Bilingual Evaluation Understudy scores are a metric used to evaluate the quality of text which has been machine-translated from one language to another. They are also commonly used for evaluating the quality of generated text in tasks like image captioning, where the generated text is compared against one or more reference texts.
   - BLEU-1: Measures the precision of unigrams (single words). It checks how many single words in the generated text match the reference text.
   - BLEU-2: Measures the precision of bigrams (pairs of consecutive words). It checks how many word pairs in the generated text match the reference text.
   - BLEU-1 score of 0.546282 and BLEU-2 score of 0.320683 are considered reasonably good scores. For state-of-the-art models, BLEU-1 scores can often be in the range of 0.6 to 0.7, and BLEU-2 scores can be around 0.4 to 0.5 or higher.
   
   ## Model Architecture
 
![model](https://github.com/user-attachments/assets/5df370bb-1fcd-4b8d-8b21-af4ce51930f6)

   ## How to use?
   1. Clone this repository, navigate into the repository.
   2. Download the machine learning model from this [link](https://drive.google.com/file/d/1EQ1gj9u3hHrDGsxhL1C-g1-PA_lAwUnV/view?usp=sharing) into the root directory of the application.
   3. Create a python virtual environment.
      ```terminal
      python3 -m venv venv
      ```
   4. Activate the virtual environment.
      ```terminal
      source venv/bin/activate
      ```
   5. Build the project.
      ```terminal
      pip install -r requirements.txt
      ```
   6. Run the appliction.
      ```terminal
      python app.py
      ```

   ## GUI
![Screenshot from 2024-07-10 19-40-34](https://github.com/user-attachments/assets/066f0f94-baba-4117-bf14-0c4d1815d388)
![Screenshot from 2024-07-10 19-41-42](https://github.com/user-attachments/assets/36b01bcb-3a16-43b4-b379-23c62ef29385)


