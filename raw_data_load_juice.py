"""
Dataset loader
"""

"""
Imports
"""
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'

"""
The following functions may be added and replace the load_dataset_model function in the data_load_juice.py to load a raw version of the datasets. This
raw version does not have any preprocessing as carried out by the MACE algorithm. Please, see: https://github.com/amirhk/mace.
Note that you must import LabelEncoder from sklearn.preprocessing and setup the raw files from their corresponding sources (UCI, Propublica) 
"""

def erase_missing(data,data_str):
    """
    Function that eliminates instances with missing values
    Input data: The dataset of interest
    Input data_str: Name of the dataset
    Output data: Filtered dataset without points with missing values
    """
    data = data.replace({'?':np.nan})
    data = data.replace({' ?':np.nan})
    if data_str == 'compass':
        for i in data.columns:
            if data[i].dtype == 'O' or data[i].dtype == 'str':
                if len(data[i].apply(type).unique()) > 1:
                    data[i] = data[i].apply(float)
                    data.fillna(0,inplace=True)    
                data.fillna('0',inplace=True)
            else:
                data.fillna(0,inplace=True)
    data.dropna(axis=0,how='any',inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def nom_to_num(data):
    """
    Function to transform categorical features into encoded numerical values.
    Input data: The dataset to encode the categorical features.
    Output data: The dataset with categorical features encoded into numerical features.
    """
    encoder = LabelEncoder()
    if data['label'].dtypes == object or data['label'].dtypes == str:
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
    return data, encoder

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
        raw_df = pd.read_csv(dataset_dir+'synthetic_disease/synthetic_disease.csv',index_col=0)
    elif data_str == 'synthetic_athlete':
        binary = ['Sex']
        categorical = ['Diet','Sport','TrainingTime']
        numerical = ['Age','SleepHours']
        label = ['Label']
        raw_df = pd.read_csv(dataset_dir+'synthetic_athlete/synthetic_athlete.csv',index_col=0)
    elif data_str == 'ionosphere':
        binary = []
        categorical = []
        numerical = ['0','2','4','5','6','7','26','30'] #Chosen based on MDI
        label = ['label']
        columns = [str(i) for i in range(34)]
        columns = columns + label
        data = pd.read_csv(dataset_dir+'/ionosphere/ionosphere.data',names=columns)
        data = data[numerical + label]
        raw_df, lbl_encoder = nom_to_num(data)
    elif data_str == 'compass':
        raw_df = pd.DataFrame()
        binary = ['Race','Sex','ChargeDegree']
        categorical = []
        numerical = ['PriorsCount','AgeGroup']
        label = ['TwoYearRecid (label)']
        # Data filtering and preparation: As seen in MACE algorithm and based on Propublica methodology. Please, see: https://github.com/amirhk/mace)
        FEATURES_CLASSIFICATION = ['age_cat','race','sex','priors_count','c_charge_degree']
        CONT_VARIABLES = ['priors_count']
        CLASS_FEATURE = 'two_year_recid'
        SENSITIVE_ATTRS = ['race']
        df = pd.read_csv(dataset_dir+'/compass/compas-scores-two-years.csv')
        df = df.dropna(subset=["days_b_screening_arrest"])
        tmp = \
            ((df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)) & \
            (df["is_recid"] != -1) & \
            (df["c_charge_degree"] != "O") & \
            (df["score_text"] != "NA") & \
            ((df["race"] == "African-American") | (df["race"] == "Caucasian"))
        df = df[tmp == True]
        df = pd.concat([df[FEATURES_CLASSIFICATION],df[CLASS_FEATURE],], axis=1)
        raw_df['TwoYearRecid (label)'] = df['two_year_recid']
        raw_df.loc[df['age_cat'] == 'Less than 25', 'AgeGroup'] = 1
        raw_df.loc[df['age_cat'] == '25 - 45', 'AgeGroup'] = 2
        raw_df.loc[df['age_cat'] == 'Greater than 45', 'AgeGroup'] = 3
        raw_df.loc[df['race'] == 'African-American', 'Race'] = 1
        raw_df.loc[df['race'] == 'Caucasian', 'Race'] = 2
        raw_df.loc[df['sex'] == 'Male', 'Sex'] = 1
        raw_df.loc[df['sex'] == 'Female', 'Sex'] = 2
        raw_df['PriorsCount'] = df['priors_count']
        raw_df.loc[df['c_charge_degree'] == 'M', 'ChargeDegree'] = 1
        raw_df.loc[df['c_charge_degree'] == 'F', 'ChargeDegree'] = 2
        raw_df = raw_df.reset_index(drop=True)
    elif data_str == 'credit':
        binary = ['isMale','isMarried','HasHistoryOfOverduePayments']
        categorical = []
        numerical = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount','TotalOverdueCounts','TotalMonthsOverdue','AgeGroup','EducationLevel']
        label = ['NoDefaultNextMonth (label)']
        raw_df = pd.read_csv(dataset_dir + '/credit/credit_processed.csv') # File obtained from MACE algorithm Datasets (please, see: https://github.com/amirhk/mace)
    elif data_str == 'adult':
        binary = ['Sex','NativeCountry']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        numerical = ['Age','EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek','EducationLevel']
        label = ['label']
        attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
        int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        sensitive_attrs = ['sex']
        attrs_to_ignore = ['sex', 'race','fnlwgt']
        attrs_for_classification = set(attrs) - set(attrs_to_ignore)
        this_files_directory = dataset_dir+data_str+'/'
        data_files = ["adult.data", "adult.test"]
        X = []
        y = []
        x_control = {}
        attrs_to_vals = {}
        for k in attrs:
            if k in sensitive_attrs:
                x_control[k] = []
            elif k in attrs_to_ignore:
                pass
            else:
                attrs_to_vals[k] = []
        for file_name in data_files:
            full_file_name = os.path.join(this_files_directory, file_name)
            print(full_file_name)
            for line in open(full_file_name):
                line = line.strip()
                if line == "":
                    continue
                line = line.split(", ")
                if len(line) != 15 or "?" in line:
                    continue
                class_label = line[-1]
                if class_label in ["<=50K.", "<=50K"]:
                    class_label = 0
                elif class_label in [">50K.", ">50K"]:
                    class_label = +1
                else:
                    raise Exception("Invalid class label value")
                y.append(class_label)
                for i in range(0, len(line) - 1):
                    attr_name = attrs[i]
                    attr_val = line[i]
                    if attr_name == "native_country":
                        if attr_val != "United-States":
                            attr_val = "Non-United-Stated"
                    elif attr_name == "education":
                        if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                            attr_val = "prim-middle-school"
                        elif attr_val in ["9th", "10th", "11th", "12th"]:
                            attr_val = "high-school"
                    if attr_name in sensitive_attrs:
                        x_control[attr_name].append(attr_val)
                    elif attr_name in attrs_to_ignore:
                        pass
                    else:
                        attrs_to_vals[attr_name].append(attr_val)
        all_attrs_to_vals = attrs_to_vals
        all_attrs_to_vals['sex'] = x_control['sex']
        all_attrs_to_vals['label'] = y
        first_key = list(all_attrs_to_vals.keys())[0]
        for key in all_attrs_to_vals.keys():
            assert (len(all_attrs_to_vals[key]) == len(all_attrs_to_vals[first_key]))
        df = pd.DataFrame.from_dict(all_attrs_to_vals)
        raw_df = pd.DataFrame()
        raw_df['label'] = df['label']
        raw_df.loc[df['sex'] == 'Male', 'Sex'] = 1
        raw_df.loc[df['sex'] == 'Female', 'Sex'] = 2
        raw_df['Age'] = df['age'].astype(int)
        raw_df.loc[df['native_country'] == 'United-States', 'NativeCountry'] = 1
        raw_df.loc[df['native_country'] == 'Non-United-Stated', 'NativeCountry'] = 2
        raw_df.loc[df['workclass'] == 'Federal-gov', 'WorkClass'] = 1
        raw_df.loc[df['workclass'] == 'Local-gov', 'WorkClass'] = 2
        raw_df.loc[df['workclass'] == 'Private', 'WorkClass'] = 3
        raw_df.loc[df['workclass'] == 'Self-emp-inc', 'WorkClass'] = 4
        raw_df.loc[df['workclass'] == 'Self-emp-not-inc', 'WorkClass'] = 5
        raw_df.loc[df['workclass'] == 'State-gov', 'WorkClass'] = 6
        raw_df.loc[df['workclass'] == 'Without-pay', 'WorkClass'] = 7
        raw_df['EducationNumber'] = df['education_num'].astype(int)
        raw_df.loc[df['education'] == 'prim-middle-school', 'EducationLevel'] = int(1)
        raw_df.loc[df['education'] == 'high-school', 'EducationLevel'] = int(2)
        raw_df.loc[df['education'] == 'HS-grad', 'EducationLevel'] = int(3)
        raw_df.loc[df['education'] == 'Some-college', 'EducationLevel'] = int(4)
        raw_df.loc[df['education'] == 'Bachelors', 'EducationLevel'] = int(5)
        raw_df.loc[df['education'] == 'Masters', 'EducationLevel'] = int(6)
        raw_df.loc[df['education'] == 'Doctorate', 'EducationLevel'] = int(7)
        raw_df.loc[df['education'] == 'Assoc-voc', 'EducationLevel'] = int(8)
        raw_df.loc[df['education'] == 'Assoc-acdm', 'EducationLevel'] = int(9)
        raw_df.loc[df['education'] == 'Prof-school', 'EducationLevel'] = int(10)
        raw_df.loc[df['marital_status'] == 'Divorced', 'MaritalStatus'] = 1
        raw_df.loc[df['marital_status'] == 'Married-AF-spouse', 'MaritalStatus'] = 2
        raw_df.loc[df['marital_status'] == 'Married-civ-spouse', 'MaritalStatus'] = 3
        raw_df.loc[df['marital_status'] == 'Married-spouse-absent', 'MaritalStatus'] = 4
        raw_df.loc[df['marital_status'] == 'Never-married', 'MaritalStatus'] = 5
        raw_df.loc[df['marital_status'] == 'Separated', 'MaritalStatus'] = 6
        raw_df.loc[df['marital_status'] == 'Widowed', 'MaritalStatus'] = 7
        raw_df.loc[df['occupation'] == 'Adm-clerical', 'Occupation'] = 1
        raw_df.loc[df['occupation'] == 'Armed-Forces', 'Occupation'] = 2
        raw_df.loc[df['occupation'] == 'Craft-repair', 'Occupation'] = 3
        raw_df.loc[df['occupation'] == 'Exec-managerial', 'Occupation'] = 4
        raw_df.loc[df['occupation'] == 'Farming-fishing', 'Occupation'] = 5
        raw_df.loc[df['occupation'] == 'Handlers-cleaners', 'Occupation'] = 6
        raw_df.loc[df['occupation'] == 'Machine-op-inspct', 'Occupation'] = 7
        raw_df.loc[df['occupation'] == 'Other-service', 'Occupation'] = 8
        raw_df.loc[df['occupation'] == 'Priv-house-serv', 'Occupation'] = 9
        raw_df.loc[df['occupation'] == 'Prof-specialty', 'Occupation'] = 10
        raw_df.loc[df['occupation'] == 'Protective-serv', 'Occupation'] = 11
        raw_df.loc[df['occupation'] == 'Sales', 'Occupation'] = 12
        raw_df.loc[df['occupation'] == 'Tech-support', 'Occupation'] = 13
        raw_df.loc[df['occupation'] == 'Transport-moving', 'Occupation'] = 14
        raw_df.loc[df['relationship'] == 'Husband', 'Relationship'] = 1
        raw_df.loc[df['relationship'] == 'Not-in-family', 'Relationship'] = 2
        raw_df.loc[df['relationship'] == 'Other-relative', 'Relationship'] = 3
        raw_df.loc[df['relationship'] == 'Own-child', 'Relationship'] = 4
        raw_df.loc[df['relationship'] == 'Unmarried', 'Relationship'] = 5
        raw_df.loc[df['relationship'] == 'Wife', 'Relationship'] = 6
        raw_df['CapitalGain'] = df['capital_gain'].astype(int)
        raw_df['CapitalLoss'] = df['capital_loss'].astype(int)
        raw_df['HoursPerWeek'] = df['hours_per_week'].astype(int)
    elif data_str == 'german':
        binary = ['Sex']
        categorical = []
        numerical = ['Age','Credit','LoanDuration']
        label = ['GoodCustomer (label)']
        raw_df = pd.DataFrame()
        raw_df = pd.read_csv(dataset_dir+'/german/german_raw.csv')
        raw_df['GoodCustomer (label)'] = raw_df['GoodCustomer']
        raw_df['GoodCustomer (label)'] = (raw_df['GoodCustomer (label)'] + 1) / 2
        raw_df.loc[raw_df['Gender'] == 'Male', 'Sex'] = 1
        raw_df.loc[raw_df['Gender'] == 'Female', 'Sex'] = 0
        raw_df['Age'] = raw_df['Age']
        raw_df['Credit'] = raw_df['Credit']
        raw_df['LoanDuration'] = raw_df['LoanDuration']   
    elif data_str == 'heart':
        binary = ['Sex']
        categorical = ['ChestPain','ECG']
        numerical = ['Age','RestBloodPressure','Chol','BloodSugar']
        label = ['class']
        columns = ['Age','Sex','ChestPain','RestBloodPressure','Chol','BloodSugar','ECG','thalach','exang','oldpeak','slope','ca','thal','class']
        data = pd.read_csv(dataset_dir+'/heart/processed.cleveland.data',names=columns)
        raw_df = data[['Sex','Age','ChestPain','RestBloodPressure','Chol','BloodSugar','ECG','class']]
        raw_df = erase_missing(raw_df,data_str)
        raw_df['class'].replace(2,1,inplace=True)
        raw_df['class'].replace(3,1,inplace=True)
        raw_df['class'].replace(4,1,inplace=True)
        
    data_obj = Dataset(seed,train_fraction,data_str,label,
                 raw_df,binary,categorical,numerical,step)
    if path_here is not None:
        model_obj = Model(data_obj,path_here)
        data_obj.filter_undesired_class(model_obj)
        data_obj.change_targets_to_numpy()
        return data_obj, model_obj
    else:
        return data_obj