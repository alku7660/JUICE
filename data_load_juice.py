"""
Dataset loader
"""

"""
Imports
"""

import os
import copy
from model_params import clf_model, best_model_params
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'

def euclidean(x1,x2):
    """
    Calculation of the euclidean distance between two different instances
    Input x1: Instance 1
    Input x2: Instance 2
    Output euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1-x2)**2))

def sort_data_distance(x,data,data_label):
    """
    Function to organize dataset with respect to distance to instance x
    Input x: Instance (can be the instane of interest or a synthetic instance)
    Input data: Training dataset
    Input data_label: Training dataset label
    Output data_sorted_distance: Training dataset sorted by distance to the instance of interest x
    """
    sort_data_distance = []
    for i in range(len(data)):
        dist = euclidean(data[i],x)
        sort_data_distance.append((data[i],dist,data_label[i]))      
    sort_data_distance.sort(key=lambda x: x[1])
    return sort_data_distance

class Dataset:
    """
    **Parts of this class are adapted from the MACE algorithm methodology (please, see: https://github.com/amirhk/mace)**
    Dataset Class: The class contains the following attributes:
        (1) seed_int:               Seed integer
        (2) train_fraction:         Train fraction
        (3) name:                   Dataset name
        (4) label_str:              Name of the column label,
        (5) raw_df:                 Raw or processed dataframe,
        (6) juice_binary:           List of binary features
        (7) juice_categorical:      List of categorical features 
        (8) juice_numerical:        List of numerical features (includes ordinal and continuous)
        (9) step:                   Step size of the continuous features changes
       (10) balanced_df:            Balanced dataset,
       (11) balanced_df_label:      Balanced dataset label,
       (12) train_pd:               Training dataset dataframe,
       (13) juice_train_pd:         Training dataset dataframe with preprocessing,
       (14) juice_train_np:         Training dataset array with preprocessing,
       (15) test_pd:                Testing dataset dataframe, 
       (16) juice_test_pd:          Testing dataset dataframe with preprocessing,
       (17) juice_test_np:          Testing dataset array with preprocessing,
       (18) train_target:           Training dataset targets,
       (19) test_target:            Testing dataset targets,
       (20) test_undesired_pd:      Undesired class testing dataset,
       (21) test_undesired_target:  Undesired class testing dataset targets,
       (22) oh_juice_bin_enc:       One-Hot Encoder used for binary feature preprocessing,
       (23) oh_juice_bin_enc_cols:  Output columns of One-Hot Encoded binary features,
       (24) oh_juice_cat_enc:       One-Hot Encoder used for categorical feature preprocessing,
       (25) oh_juice_cat_enc_cols:  Output columns of One-Hot Encoded categorical features,
       (26) juice_scaler:           MinMaxScaler for data preprocessing,
       (27) feat_type:              Feature type vector, 
       (28) feat_mutable:           Feature mutability vector,
       (29) feat_dir:               Feature directionality vector,
       (30) feat_cost:              Feature unit cost vector,
       (31) feat_step:              Feature step size vector,
       (32) feat_cat:               Feature categorical group indicator vector 
       (33) train_sorted:           Sorted training dataset with respect to a given instance (initialized as needed when an instance is required)
       (34) undesired_class:        Undesired class of the dataset
    """

    def __init__(self,seed_int,train_fraction,data_str,label_str,
                 raw_df,binary,categorical,numerical,step):

        self.seed = seed_int
        self.train_fraction = train_fraction
        self.name = data_str
        self.label_str = label_str
        self.raw_df = raw_df
        self.juice_binary = binary
        self.juice_categorical = categorical
        self.juice_numerical = numerical
        self.step = step
        self.balanced_df, self.balanced_df_label = self.data_balancing() 
        self.train_pd, self.test_pd, self.train_target, self.test_target = train_test_split(self.balanced_df,self.balanced_df_label,train_size=self.train_fraction,random_state=self.seed)
        self.juice_encoder_scaler_fit_transform_train()
        self.juice_test_pd, self.juice_test_np = self.juice_encoder_scaler_transform_test(self.test_pd)
        self.undesired_class = self.undesired_class_data()
        self.feat_type = self.define_feat_type()
        self.feat_mutable = self.define_mutability()
        self.feat_dir = self.define_directionality()
        self.feat_cost = self.define_feat_cost()
        self.feat_step = self.define_feat_step()
        self.feat_cat = self.define_category_groups()
        self.train_sorted = None

    def data_balancing(self):
        """
        Method that balances the dataset (Adapted from MACE algorithm methodology (please, see: https://github.com/amirhk/mace))
        """
        data_label = self.raw_df[self.label_str]
        unique_values_and_count = data_label.value_counts()
        if self.name in ['heart','ionosphere']:
            number_of_subsamples_per_class = unique_values_and_count.min() // 50 * 50
        else:
            number_of_subsamples_per_class = unique_values_and_count.min() // 250 * 250
        balanced_df = pd.concat([self.raw_df[(data_label == 0).to_numpy()].sample(number_of_subsamples_per_class, random_state = self.seed),
        self.raw_df[(data_label == 1).to_numpy()].sample(number_of_subsamples_per_class, random_state = self.seed),]).sample(frac = 1, random_state = self.seed)
        balanced_df_label = balanced_df[self.label_str]
        del balanced_df[self.label_str[0]]
        return balanced_df, balanced_df_label

    def juice_encoder_scaler_fit_transform_train(self):
        """
        Method that fits the encoder and scaler for the dataset and transforms the training dataset according to the JUICE framework
        """
        oh_juice_bin_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        oh_juice_cat_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        oh_juice_bin_cat_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore',sparse=False)
        juice_scaler = MinMaxScaler(clip=True)
        juice_train_data_bin, juice_train_data_cat, juice_train_data_num, juice_train_data_bin_cat = self.train_pd[self.juice_binary], self.train_pd[self.juice_categorical], self.train_pd[self.juice_numerical], self.train_pd[self.juice_binary+self.juice_categorical]
        enc_juice_train_data_bin = oh_juice_bin_enc.fit_transform(juice_train_data_bin).toarray()
        enc_juice_train_data_cat = oh_juice_cat_enc.fit_transform(juice_train_data_cat).toarray()
        oh_juice_bin_cat_enc.fit(juice_train_data_bin_cat)
        scaled_juice_train_data_num = juice_scaler.fit_transform(juice_train_data_num)
        self.oh_juice_bin_enc, self.oh_juice_bin_enc_cols = oh_juice_bin_enc, oh_juice_bin_enc.get_feature_names_out(self.juice_binary)
        self.oh_juice_cat_enc, self.oh_juice_cat_enc_cols = oh_juice_cat_enc, oh_juice_cat_enc.get_feature_names_out(self.juice_categorical)
        self.oh_juice_bin_cat_enc = oh_juice_bin_cat_enc
        self.juice_scaler = juice_scaler
        scaled_juice_train_data_num_pd = pd.DataFrame(scaled_juice_train_data_num,index=juice_train_data_num.index,columns=self.juice_numerical)
        enc_juice_train_data_bin_pd = pd.DataFrame(enc_juice_train_data_bin,index=juice_train_data_bin.index,columns=self.oh_juice_bin_enc_cols)
        enc_juice_train_data_cat_pd = pd.DataFrame(enc_juice_train_data_cat,index=juice_train_data_cat.index,columns=self.oh_juice_cat_enc_cols)
        self.juice_train_pd = self.transform_to_juice_format(scaled_juice_train_data_num_pd,enc_juice_train_data_bin_pd,enc_juice_train_data_cat_pd)
        self.juice_all_cols = self.juice_train_pd.columns.to_list()
        self.juice_train_np = self.juice_train_pd.to_numpy()
        
    def juice_encoder_scaler_transform_test(self,test_df):
        """
        Method that uses the encoder and scaler for the dataset and transforms the testing dataset according to the JUICE framework
        Input test_df: Testing dataframe to transform (encode and scale)
        Output juice_test_pd:
        """
        juice_test_data_bin, juice_test_data_cat, juice_test_data_num = test_df[self.juice_binary], test_df[self.juice_categorical], test_df[self.juice_numerical]
        enc_juice_test_data_bin, enc_juice_test_data_cat = self.oh_juice_bin_enc.transform(juice_test_data_bin).toarray(), self.oh_juice_cat_enc.transform(juice_test_data_cat).toarray()
        scaled_juice_test_data_num = self.juice_scaler.transform(juice_test_data_num)
        enc_juice_test_data_bin_pd = pd.DataFrame(enc_juice_test_data_bin,index=juice_test_data_bin.index,columns=self.oh_juice_bin_enc_cols)
        enc_juice_test_data_cat_pd = pd.DataFrame(enc_juice_test_data_cat,index=juice_test_data_cat.index,columns=self.oh_juice_cat_enc_cols)
        scaled_juice_test_data_num_pd = pd.DataFrame(scaled_juice_test_data_num,index=juice_test_data_num.index,columns=self.juice_numerical)
        juice_test_pd = self.transform_to_juice_format(scaled_juice_test_data_num_pd,enc_juice_test_data_bin_pd,enc_juice_test_data_cat_pd)
        juice_test_np = juice_test_pd.to_numpy()
        return juice_test_pd, juice_test_np

    def transform_to_juice_format(self,num_data,enc_bin_data,enc_cat_data):
        """
        Method that transforms an instance of interest to the JUICE format to be comparable
        Input num_data: The numerical (continuous) variables in DataFrame transformed into the JUICE format
        Input enc_bin_data: The binary variables transformed in DataFrame into the JUICE format
        Input enc_cat_cata: The categorical variables transformed in DataFrame into the JUICE format
        Output enc_juice_data_pd: The DataFrame instance in the JUICE format
        """
        if self.name in ['compass']:
            enc_juice_data_pd = pd.concat((enc_bin_data[self.oh_juice_bin_enc_cols[:2]],num_data[self.juice_numerical[0]],
                                enc_bin_data[self.oh_juice_bin_enc_cols[2:]],num_data[self.juice_numerical[1]]),axis=1)
        elif self.name in ['credit']:
            enc_juice_data_pd = pd.concat((enc_bin_data[self.oh_juice_bin_enc_cols[:2]],num_data[self.juice_numerical[0:9]],
                                enc_bin_data[self.oh_juice_bin_enc_cols[2:]],num_data[self.juice_numerical[9:]]),axis=1)
        elif self.name in ['adult']:
            enc_juice_data_pd = pd.concat((enc_bin_data[self.oh_juice_bin_enc_cols[0]],num_data[self.juice_numerical[0]],
                                enc_bin_data[self.oh_juice_bin_enc_cols[1]],num_data[self.juice_numerical[1:5]],
                                enc_cat_data[self.oh_juice_cat_enc_cols[:7]],num_data[self.juice_numerical[-1]],
                                enc_cat_data[self.oh_juice_cat_enc_cols[7:]]),axis=1)
        elif self.name == 'german':
            enc_juice_data_pd = pd.concat((enc_bin_data,num_data),axis=1)
        elif self.name in ['heart','synthetic_disease','synthetic_athlete']:
            enc_juice_data_pd = pd.concat((enc_bin_data,num_data,enc_cat_data),axis=1)
        elif self.name in ['ionosphere']:
            enc_juice_data_pd = num_data
        return enc_juice_data_pd

    def filter_undesired_class(self,model):
        """
        Method that obtains the undesired class instances according to the JUICE selected model
        Input model: Model object containing the trained model for JUICE framework
        """
        self.juice_test_pd['pred'] = list(model.juice_sel.predict(self.juice_test_pd))
        undesired_indices = self.juice_test_pd.loc[self.juice_test_pd['pred'] == self.undesired_class].index.to_list()
        self.juice_test_undesired_pd, self.test_undesired_target = self.juice_test_pd.loc[undesired_indices,:], self.test_target.loc[undesired_indices,:]
        del self.juice_test_pd['pred']
        del self.juice_test_undesired_pd['pred']
        self.test_undesired_pd = self.test_pd.loc[undesired_indices,:]
        self.juice_test_undesired_np = self.juice_test_undesired_pd.to_numpy()
        self.test_undesired_np = self.test_undesired_pd.to_numpy()

    def change_targets_to_numpy(self):
        """
        Method that changes the targets to numpy if they are dataframes
        """
        if isinstance(self.train_target, pd.Series) or isinstance(self.train_target, pd.DataFrame):
            self.train_target = self.train_target.to_numpy().reshape((len(self.train_target.to_numpy()),))
        if isinstance(self.test_target, pd.Series) or isinstance(self.test_target, pd.DataFrame):
            self.test_target = self.test_target.to_numpy().reshape((len(self.test_target.to_numpy()),))
        if isinstance(self.test_undesired_target, pd.Series) or isinstance(self.test_undesired_target, pd.DataFrame):
            self.test_undesired_target = self.test_undesired_target.to_numpy().reshape((len(self.test_undesired_target.to_numpy()),))

    def add_test_predictions(self,predictions):
        """
        Method to add the test data predictions from a model
        Input predictions: Predictions for the test dataset
        """
        self.test_pred = predictions
    
    def add_sorted_train_data(self,instance):
        """
        Method to add/change a sorted array of the training dataset according to distance from an instance
        Input instance: Instance of interest from which to calculate all the distances
        """
        self.train_sorted = sort_data_distance(instance,self.juice_train_np,self.train_target) 

    def undesired_class_data(self):
        """
        Method to obtain the undesired class
        """
        if self.name in ['compass','credit','german','heart','synthetic_disease']:
            undesired_class = 1
        elif self.name in ['ionosphere','adult','synthetic_athlete']:
            undesired_class = 0
        return undesired_class

    def define_feat_type(self):
        """
        Method that obtains a feature type vector corresponding to each of the featurs
        Output feat_type: Dataset feature type series in usable format
        """
        feat_type = self.juice_train_pd.dtypes
        feat_type_2 = copy.deepcopy(feat_type)
        feat_list = feat_type.index.tolist()
        if self.name in ['ionosphere']:
            for i in feat_list:
                feat_type_2.loc[i] = 'num-con'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if i in ['Age','ExerciseMinutes','SleepHours']:
                    feat_type_2.loc[i] = 'num-con'
                elif 'Weight' in i:
                    feat_type_2.loc[i] = 'num-ord'
                elif 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_type_2.loc[i] = 'bin'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Sex' in i or 'TrainingTime' in i or 'Diet' in i or 'Sport' in i:
                    feat_type_2.loc[i] = 'bin'
                elif i in ['Age','SleepHours']:
                    feat_type_2.loc[i] = 'num-con'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Charge' in i:
                    feat_type_2.loc[i] = 'bin'
                elif 'Priors' in i or 'Age' in i:
                    feat_type_2.loc[i] = 'num-ord'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Married' in i or 'History' in i:
                    feat_type_2.loc[i] = 'bin'
                elif 'Amount' in i or 'Balance' in i or 'Spending' in i:
                    feat_type_2.loc[i] = 'num-con'
                elif 'Total' in i or 'Age' in i or 'Education' in i:
                    feat_type_2.loc[i] = 'num-ord'
        elif self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i:
                    feat_type_2.loc[i] = 'bin'
                elif 'EducationLevel' in i:
                    feat_type_2.loc[i] = 'num-ord'
                elif 'EducationNumber' in i or 'Capital' in i or 'Hours' in i or 'Age' in i:
                    feat_type_2.loc[i] = 'num-con'
        elif self.name == 'german':
            for i in feat_list:
                if 'Sex' in i:
                    feat_type_2.loc[i] = 'bin'
                elif 'Age' in i or 'Credit' in i or 'Loan' in i:
                    feat_type_2.loc[i] = 'num-con'
        elif self.name == 'heart':
            for i in feat_list:
                if 'Sex' in i or 'ChestPain' in i or 'ECG' in i:
                    feat_type_2.loc[i] = 'bin'
                elif i in ['Age','RestBloodPressure','Chol','BloodSugar']:
                    feat_type_2.loc[i] = 'num-con'
        return feat_type_2

    def define_mutability(self):
        """
        Method that outputs mutable features per dataset
        Output feat_mutable: Mutability of each feature
        """
        feat_list = self.feat_type.index.tolist()
        feat_mutable  = dict()
        if self.name == 'synthetic_disease':
            for i in feat_list:
                if i == 'Age':
                    feat_mutable[i] = 0
                elif 'Weight' in i or 'ExerciseMinutes' in i or 'SleepHours' in i or 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_mutable[i] = 1
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if i in ['Age','Sex']:
                    feat_mutable[i] = 0
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_mutable[i] = 1
        elif self.name == 'ionosphere':
            for i in feat_list:
                if i == '0':
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'compass':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Age' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1   
        elif self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Native' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'german':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_mutable[i] = 0
                else:
                    feat_mutable[i] = 1
        elif self.name == 'heart':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_mutable[i] = 0
                elif 'ChestPain' in i or 'ECG' in i or i in ['RestBloodPressure','Chol','BloodSugar']:
                    feat_mutable[i] = 1
        feat_mutable = pd.Series(feat_mutable)
        return feat_mutable

    def define_directionality(self):
        """
        Method that outputs change directionality of features per dataset
        Output feat_dir: Plausible direction of change of each feature
        """
        feat_list = self.feat_type.index.tolist()
        feat_dir  = dict()
        if self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Age' in i:
                    feat_dir[i] = 0
                elif 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i:
                    feat_dir[i] = 'any'
                elif 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_dir[i] = 'any'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_dir[i] = 0
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_dir[i] = 'any'
        elif self.name == 'ionosphere':
            for i in feat_list:
                feat_dir[i] = 'any'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i:
                    feat_dir[i] = 0
                elif 'Charge' in i or 'Priors' in i:
                    feat_dir[i] = 'any'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Age' in i or 'Male' in i:
                    feat_dir[i] = 0
                elif 'OverLast6Months' in i or 'MostRecent' in i or 'Total' in i or 'History' in i:
                    feat_dir[i] = 'any'   
                elif 'Education' in i:
                    feat_dir[i] = 'pos'
                elif 'Married' in i:
                    feat_dir[i] = 'any'
        elif self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Native' in i:
                    feat_dir[i] = 0
                elif 'Education' in i:
                    feat_dir[i] = 'pos'
                else:
                    feat_dir[i] = 'any'
        elif self.name == 'german':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_dir[i] = 0
                else:
                    feat_dir[i] = 'any'
        elif self.name == 'heart':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_dir[i] = 0
                elif 'ChestPain' in i or 'ECG' in i:
                    feat_dir[i] = 'any'
                elif i in ['RestBloodPressure','Chol','BloodSugar']:
                    feat_dir[i] = 'any'
        feat_dir = pd.Series(feat_dir)
        return feat_dir

    def define_feat_cost(self):
        """
        Method that allocates a unit cost of change to the features of the datasets
        Output feat_cost: Theoretical unit cost of changing each feature
        """
        feat_cost  = dict()
        feat_list = self.feat_type.index.tolist()
        if self.name == 'ionosphere':
            for i in feat_list:
                if i == '0':
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Age' in i:
                    feat_cost[i] = 0
                elif 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i:
                    feat_cost[i] = 1
                elif 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_cost[i] = 1
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_cost[i] = 0
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_cost[i] = 1
        elif self.name == 'compass':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i:
                    feat_cost[i] = 0
                elif 'Charge' in i:
                    feat_cost[i] = 1#10
                elif 'Priors' in i:
                    feat_cost[i] = 1#20
        elif self.name == 'credit':
            for i in feat_list:
                if 'Age' in i or 'Male' in i:
                    feat_cost[i] = 0
                elif 'OverLast6Months' in i or 'MostRecent' in i or 'TotalOverdueCounts' in i or 'History' in i:
                    feat_cost[i] = 1#20
                elif 'TotalMonthsOverdue' in i:
                    feat_cost[i] = 1#10   
                elif 'Education' in i:
                    feat_cost[i] = 1#50
                elif 'Married' in i:
                    feat_cost[i] = 1#50
        elif self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Native' in i:
                    feat_cost[i] = 0
                elif 'EducationLevel' in i:
                    feat_cost[i] = 1#50
                elif 'EducationNumber' in i:
                    feat_cost[i] = 1#20
                elif 'WorkClass' in i:
                    feat_cost[i] = 1#10
                elif 'Capital' in i:
                    feat_cost[i] = 1#5
                elif 'Hours' in i:
                    feat_cost[i] = 1#2
                elif 'Marital' in i:
                    feat_cost[i] = 1#50
                elif 'Occupation' in i:
                    feat_cost[i] = 1#10
                elif 'Relationship' in i:
                    feat_cost[i] = 1#50
        elif self.name == 'german':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        elif self.name == 'heart':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_cost[i] = 0
                elif 'ChestPain' in i or 'ECG' in i:
                    feat_cost[i] = 1
                elif i in ['RestBloodPressure','Chol','BloodSugar']:
                    feat_cost[i] = 1
        feat_cost = pd.Series(feat_cost)
        return feat_cost

    def define_feat_step(self):
        """
        Method that estimates the step size of all features (used for ordinal features)
        Output feat_step: Plausible step size for each feature 
        """
        feat_step = pd.Series(data=1/(self.juice_scaler.data_max_ - self.juice_scaler.data_min_),index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['num-ord','num-con']])
        for i in self.feat_type.keys().tolist():
            if self.feat_type.loc[i] == 'num-con':
                feat_step.loc[i] = self.step
            elif self.feat_type.loc[i] == 'num-ord':
                continue
            else:
                feat_step.loc[i] = 0
        feat_step = feat_step.reindex(index = self.feat_type.keys().to_list())
        return feat_step

    def define_category_groups(self):
        """
        Method that assigns categorical groups to different one-hot encoded categorical features
        Output feat_cat: Theoretical unit cost of changing each feature
        """
        feat_cat = copy.deepcopy(self.feat_type)
        feat_list = self.feat_type.index.tolist()
        if self.name in ['ionosphere']:
            for i in feat_list:
                feat_cat[i] = 'non'
        elif self.name == 'Ionosphere':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'synthetic_sever_disease':
            for i in feat_list:
                if 'Age' in i or 'Smokes' in i or 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i:
                    feat_cat.loc[i] = 'non'
                elif 'Diet' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Stress' in i:
                    feat_cat.loc[i] = 'cat_2'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'SleepHours' in i:
                    feat_cat.loc[i] = 'non'
                elif 'TrainingTime' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Diet' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Sport' in i:
                    feat_cat.loc[i] = 'cat_3'
        elif self.name == 'compass':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'credit':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Native' in i or 'EducationLevel' or i in 'EducationNumber' in i or 'Capital' in i or 'Hours' in i:
                    feat_cat.loc[i] = 'non'
                elif 'WorkClass' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Marital' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Occupation' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'Relation' in i:
                    feat_cat.loc[i] = 'cat_4'
        elif self.name == 'german':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'heart':
            for i in feat_list:
                if i in ['Age','Sex','RestBloodPressure','Chol','BloodSugar']:
                    feat_cat.loc[i] = 'non'
                elif 'ChestPain' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'ECG' in i:
                    feat_cat.loc[i] = 'cat_2'
        return feat_cat

class Model:
    """
    Class that contains the trained models for JUICE framework
    """
    def __init__(self,data_obj,grid_search_path):
        self.model_params_path = grid_search_path
        self.train_clf_model(data_obj)
    
    def train_clf_model(self,data_obj):
        """
        Method that trains the classifier model according to the data object received
        """
        grid_search_results = pd.read_csv(str(self.model_params_path)+'/grid_search_final.csv',index_col = ['dataset','model'])
        sel_model_str, params_best, params_rf = best_model_params(grid_search_results,data_obj.name)
        self.juice_sel, self.juice_rf = clf_model(sel_model_str,params_best,params_rf,data_obj.juice_train_pd,data_obj.train_target)

def load_model_dataset(data_str,train_fraction,seed,step,path_here = None):
    """
    Function to load all datasets according to data_str and train_fraction, and the corresponding selected and RF models for counterfactual search
    Input data_str: Name of the dataset to load
    Input train_fraction: Percentage of dataset instances to use as training dataset
    Input seed: Random seed to be used
    Input step: Size of the step to be used for continuous variable changes
    Input path: Path to the grid search results for model parameter selection
    Output data_obj: Dataset object
    Output model_obj: Model object
    """
    if data_str == 'synthetic_disease':
        binary = ['Smokes']
        categorical = ['Diet','Stress']
        numerical = ['Age','ExerciseMinutes','SleepHours','Weight']
        label = ['Label']
        processed_df = pd.read_csv(dataset_dir+'synthetic_disease/processed_synthetic_disease.csv',index_col=0)
    elif data_str == 'synthetic_athlete':
        binary = ['Sex']
        categorical = ['Diet','Sport','TrainingTime']
        numerical = ['Age','SleepHours']
        label = ['Label']
        processed_df = pd.read_csv(dataset_dir+'synthetic_athlete/processed_synthetic_athlete.csv',index_col=0)
    elif data_str == 'ionosphere':
        binary = []
        categorical = []
        numerical = ['0','2','4','5','6','7','26','30'] # Chosen based on MDI
        label = ['label']
        processed_df = pd.read_csv(dataset_dir+'/ionosphere/processed_ionosphere.csv',index_col=0)
    elif data_str == 'compass':
        processed_df = pd.DataFrame()
        binary = ['Race','Sex','ChargeDegree']
        categorical = []
        numerical = ['PriorsCount','AgeGroup']
        label = ['TwoYearRecid (label)']
        processed_df = pd.read_csv(dataset_dir+'compass/processed_compass.csv',index_col=0)
    elif data_str == 'credit':
        binary = ['isMale','isMarried','HasHistoryOfOverduePayments']
        categorical = []
        numerical = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount','TotalOverdueCounts','TotalMonthsOverdue','AgeGroup','EducationLevel']
        label = ['NoDefaultNextMonth (label)']
        processed_df = pd.read_csv(dataset_dir + '/credit/credit_processed.csv') # File obtained from MACE algorithm Datasets (please, see: https://github.com/amirhk/mace)
    elif data_str == 'adult':
        binary = ['Sex','NativeCountry']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        numerical = ['Age','EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek','EducationLevel']
        label = ['label']
        processed_df = pd.read_csv(dataset_dir+'adult/processed_adult.csv',index_col=0)
    elif data_str == 'german':
        binary = ['Sex']
        categorical = []
        numerical = ['Age','Credit','LoanDuration']
        label = ['GoodCustomer (label)']  
        processed_df = pd.read_csv(dataset_dir+'german/processed_german.csv',index_col=0)
    elif data_str == 'heart':
        binary = ['Sex']
        categorical = ['ChestPain','ECG']
        numerical = ['Age','RestBloodPressure','Chol','BloodSugar']
        label = ['class']
        processed_df = pd.read_csv(dataset_dir+'heart/processed_heart.csv',index_col=0)
    
    data_obj = Dataset(seed,train_fraction,data_str,label,
                 processed_df,binary,categorical,numerical,step)
    if path_here is not None:
        model_obj = Model(data_obj,path_here)
        data_obj.filter_undesired_class(model_obj)
        data_obj.change_targets_to_numpy()
        return data_obj, model_obj
    else:
        return data_obj