# Body-Type-Predictor
This is a creation of a descriptive Machine Model that uses SVM algorithm to classify the BRI into four categories: Lean, Normal, Overweight, and Obese. With this, body type is determined. This then classifies and give you appropriate output. 

Steps to process the model

1. Google Colab
   a. If used in Google Colab, use the SVM Algorithm for colab.py
   b. Save the .csv file Determine_BRI.csv into your google drive
   c. Copy private key in url and paste it in the file_id (Click share and get the link. The ID will be after the text /d/)
      Example: https://drive.google.com/file/d/14Oqmrcf2miZQjFHQd0r_n9Xva7fI-er1/view?usp=drive_link
      ID: 14Oqmrcf2miZQjFHQd0r_n9Xva7fI-er1
   d. In the download_data function, there will be file_id. Replace the private key to the value defaultly present in it
   e. Do others as said in comments of the program
   f. After mounting drive function is the chdir function. Give your directory's link and change the directory to it
   g. Run the program cell by cell

2. Local training
   a. Create a local repository by cloning the repository locally.
   b. Download git to do that.
   c. In terminal change directory to desired directory and use the cli code in powershell git clone https://github.com/Vintage-Geek/Body-Type-Predictor
   d. After cloning, open the SVM Algorithm code along with the .csv file.
   e. You can open the .csv file using excel too so that clarifications of data can be done.
   f. Now, if you want, change the name of the dataset and then change the name in read command to make practice.
   g. Then tune the classifier in the svm_classifier function to get the desired accuracy
   h. Tune as per required, and then use the joblib part of the program and change the place the code is saved as.pkl file
   i. Now, you can use the model file to then make a UI for it and procede
