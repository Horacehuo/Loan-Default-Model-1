# Loan-Default-Model-1
This repo contains a modeling for Credit Default using python with a large semi-claer dataset.The method used here is logistics regression but for the easy of use in the work place, the final outcome from regression model has been transformed into score card.

The original data_set is called " cs-training" from Kaggle.com, which i have been uploaded in this repo and the final score card result has also been uploaded.

Please check the html format file for an overlook( with graphs), this repo will only explain the steps of analysis: 

1.import the data and lean up: 
    After we check the Describedata.csv we created, the variables" MonthlyIncome" and "NumberOfDependents" have lost some data.
    we can not remove "MonthlyIncome" instead we use Random Forest to make up. but for NumberOfDependents, we can just remove it.
    
   
