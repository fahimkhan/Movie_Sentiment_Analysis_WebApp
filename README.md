## This is web application for Movie Snetiment Analysis from the book Python Machine Learning by Sebastian Raschka




- Prerequisite 

	1. Python
	2. Flask
	3. sqlite3
	4. wtforms



- First get the data from `http://ai.stanford.edu/~amaas/data/sentiment/` .Extract it and keep it in the folder where Generating_Data_From_Archive.py resides. 

- To convert it from archive to proper csv file run the `python Generating_Data_From_Archive.py` it will create movie_data.csv

- Once the data is gathered in the csv file run the `python Generating_Model.py` which will create serialized object of model and store into pkl_objects directory.


- To run the web application just type `python app.py`
