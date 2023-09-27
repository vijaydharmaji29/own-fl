### Vijay Dharmaji

To run the image classification example

Enter the main OwnFL directory: 

As you already know, the first thing you can do is run the database and aggregator: # FL server side

python -m fl_main.pseudodb.pseudo_db

python -m fl_main.aggregator.server_th 

Then, start the first and second agents to run the image classification example: 

### First agent


python -m examples.image_classification.classification_engine 1 50001 a1

### Second agent

python -m examples.image_classification.classification_engine 1 50002 a2

### Additional Info: 

This is the process for running multiple clients on the same computer / device

For running on multiple computers just run without any additional commmand line arguments
however, ensure to change the ip addresses of the server and database.
