# Loan-Default-Model-1
This repo contains a modeling for Credit Default using python with a large semi-claer dataset.The method used here is logistics regression but for the easy of use in the work place, the final outcome from regression model has been transformed into score card.

The original data_set is called " cs-training" from Kaggle.com, which i have been uploaded in this repo and the final score card result has also been uploaded.

# 1.import the data and clean up: 
After we check the Describedata.csv we created, the variables" MonthlyIncome" and "NumberOfDependents" have lost some data.
we can not remove "MonthlyIncome" instead we use Random Forest to make up. but for NumberOfDependents, we can just remove it.
    
  ![output_9_0](https://user-images.githubusercontent.com/39636026/42730327-3989470e-8825-11e8-90fe-dbe8baf2071c.png)

    
   
