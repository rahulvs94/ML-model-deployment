# ML model deployment

Basic salary prediction model based on work experience:

1. model.py :
	- Implemented linear regression, multi-layer perceptron model on data
	- Stored model in .pkl format using Pickle library 
	
2. server.py :
	- Created a server using flask 
	- Load the .pkl model
	- Handle the POST request coming from request from request.py
	- Storing the prediction of input from request
	
3. request.py :
	- Requesting the server for the prediction
	
	
	
Reference: https://hackernoon.com/deploy-a-machine-learning-model-using-flask-da580f84e60c	