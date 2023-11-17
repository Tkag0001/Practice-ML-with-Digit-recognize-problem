# Practice-ML-with-Digit-recognize-problem
Solving digits recognize problem 
## Introdution:
  As the name, I think that joining into the problem, analyzing and coding to solve it is the easiest way to improve knowledge of Maching learning - Deep learning and solving problem skills.
  
  In this problem we just recognize the handwritten digit, not detect. I use Colab to train the models.

![demo](https://github.com/Tkag0001/Practice-ML-with-Digit-recognize-problem/assets/107709392/d7917711-4b0f-4710-98bc-009fba15d2c9)


## Resource:

I use two datasets:
- Kaggle: [mnist-dataset](https://www.kaggle.com/competitions/digit-recognizer) -- You have to download, put into a drive folder and change some url in train_model file to run.
- Keras: [mnist-dataset](https://keras.io/api/datasets/mnist/) -- You just run cell to load the dataset.

## Explanation of files:
1) In code folder:
   - Train_model1.ipynb: train_file with dataset from Kaggle
   - Train_model2.ipynb: train_file with dataset from Keras.
   - applicate_model.py: test_file identifies a digit of an image.
     
2) In data folder:
   - saved_model1.h5 and saved_model1.pkl: weight of model1.
   - saved_model2.h2 and saved_model2.pkl: weight of model2.
   - test_image.png: image to test in applicate_model.py
