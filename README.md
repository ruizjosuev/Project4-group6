In this project, we explore a Spotify dataset to predict the danceability of songs using various audio features and machine learning. We then use the trained model to build an interactive web application that classifies song danceability and recommends similar tracks.

Tools & Technologies

Data Cleaning & ML Modeling: Python, Jupyter Notebook, Pandas, Scikit-learn, LightGBM
Web Framework: Flask
Frontend: HTML, CSS, JavaScript
Visualization: Tableau

Research Questions

We focused on three main questions:

Can machine learning models accurately classify songs into danceability categories (Low, Medium, High)?
What mood categories correlate with the highest streams and the most popular songs?
How do audio features like energy, tempo, loudness, and danceability vary across music genres?

Machine Learning
Target Variable: danceability, binned into Low, Medium, and High.
We experimented with various classification algorithms and selected LightGBM as the best-performing model.
After training, the model was saved and integrated into a Flask web application.

Web Application 
Danceability Classifier: Users can input a track and artist name. The app returns the song’s predicted danceability category.
Recommender System: Built using K-Nearest Neighbors (KNN). It recommends songs with similar features like danceability, energy, tempo, and instrumentalness.
Data Visualization: Explored the correlations and trends in Tableau dashboards to visually answer the second and third research questions.


