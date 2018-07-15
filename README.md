# Loan-Default-Model-1
This repo contains a modeling for Credit Default using python with a large semi-claer dataset.The method used here is logistics regression but for the easy of use in the work place, the final outcome from regression model has been transformed into score card.

The original data_set is called " cs-training" from Kaggle.com, which i have been uploaded in this repo and the final score card result has also been uploaded.

# 1.import the data and clean them up: 
After we check the Describedata.csv we created:
![discrip_of_data](https://user-images.githubusercontent.com/39636026/42730452-f043b5e6-8826-11e8-8608-df4d1287b4d5.png)the variables" MonthlyIncome" and "NumberOfDependents" have lost some data.we can not remove "MonthlyIncome" instead we use Random Forest to make up. but for NumberOfDependents, we can just remove it.

Also, for those unreasonable values(age=0,NumberOfTime30-59DaysPastDueNotWorse>90 ) and outliers, we use code to deal with: 
   data = data[data['age'] > 0],
   data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90],
   data['SeriousDlqin2yrs']=1-data['SeriousDlqin2yrs'],
 

    
   
