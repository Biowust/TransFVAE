from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('train.csv')
subsample_size = 500  
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()

label = 'state'
print("Summary of class variable: \n", train_data[label].describe())

save_path = 'predict'  
predictor = TabularPredictor(label=label, path=save_path).fit(train_data)
