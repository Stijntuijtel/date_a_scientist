import time
import pandas                                   as pd
import numpy                                    as np
from matplotlib                 import pyplot   as plt
from multiprocessing            import Condition
from re                         import I
from argon2                     import hash_password_raw
from datetime                   import datetime
from sklearn                    import preprocessing
from sklearn.preprocessing      import MinMaxScaler
from sklearn.model_selection    import train_test_split
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.metrics            import accuracy_score, recall_score, precision_score
from sklearn                    import svm
from sklearn.neighbors          import KNeighborsRegressor
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import log_loss


def read_csv():
    return pd.read_csv("C:\\Users\\M65D426\\Onedrive - NN\\Documents\\Codecademy\\Projecten\\capstone_starter\\profiles.csv")


def explore_data(df):
    print('Size original df: ' + str(df.shape))

    for column in df.columns:
        column_head     = "df." + column + ".value_counts().head()"
        summ            = eval(column_head)
        column_keys     = [summ.keys()[i]       for i in range(len(summ))]
        column_values   = [summ[summ.keys()[i]] for i in range(len(summ))]

        # Uncomment to show exploring plots
        # make_plot(column_keys, column_values, 'Distribution of ' + column.capitalize())

    # Examining offspring variable
    offspring_data      = df.offspring.value_counts()
    offspring_keys_raw  = [offspring_data.keys()[i]         for i in range(len(offspring_data))]
    offspring_keys      = [o.replace('sn&rsquo;t', 's not') for o in offspring_keys_raw]
    offspring_values    = [offspring_data[i]                for i in range(len(offspring_data))]
    print('Empty rows column offspring: ' + str(df.offspring.isna().sum()))              # 35.561 empty fields

    make_plot(offspring_keys, offspring_values, 'Distribution of Offspring')


def make_plot(x_val, y_val, title):
    x_num       = [n + 1 for n in range(len(x_val))]
    fig, ax     = plt.subplots()
    ax.bar(x_num, y_val)
    x_legend    = '\n'.join(f'{n} - {name}' for n,name in zip(x_num, x_val))
    t           = ax.text(.8, .2 , x_legend, transform = ax.figure.transFigure)
    fig.subplots_adjust(right=.75)

    plt.title(title)
    plt.xlabel('Category (see legend right)')
    plt.ylabel('Number of respondents')
    plt.xticks(ticks = x_num, labels = x_num) # To make sure the number of bars and x-axis-values match.
    plt.show()

    
def count_parental_words(row):
    amount_of_words = 0
    parental_words = ['kid', 'kids', 'child', 'children', \
                     'daughter', 'daughters', 'son', 'sons' \
                     'mom', 'mother', 'dad', 'father', 'baby' \
                     'proud', 'parent']

    for word in row['all_essays'].split():
        if word in parental_words:
            amount_of_words += 1

    return amount_of_words

# Create df with feature columns
def mapping_and_cleaning(df, type):

    print('\n' + 'Approach: ' + str(type) + '\n')

    if type == 'classification':
        smokes_mapping  = {"no"         : 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes"       : 4}
        drugs_mapping   = {"never"      : 0, "sometimes": 1, "often"        : 2}
        drinks_mapping  = {"not at all" : 0, "rarely"   : 1, "socially"     : 2, "often"         : 3, "very often": 4, "desperately": 5}

        df['smokes_num']    = df.smokes.map(smokes_mapping)
        df['drugs_num']     = df.drugs.map(drugs_mapping)
        df['drinks_num']    = df.drinks.map(drinks_mapping)

        #  Convert object last_online to number of seconds.
        df['last_online_dt']    = pd.to_datetime(df['last_online'], format='%Y-%m-%d-%H-%M')
        df['datediff']          = df['last_online_dt'] - df['last_online_dt'].min()
        last_online_dt_ts       = df['last_online_dt'].min().timestamp()
        df['last_online_sec']   = df['datediff'].dt.total_seconds().astype(int)
        df['last_online_sec']   = df['last_online_sec'] + last_online_dt_ts

        df['income_cl'] = df['income'].replace(-1.0, np.NaN)
    
    elif type == 'regression':
        essay_cols = ['essay0', 'essay1', 'essay5', 'essay6']
        df[essay_cols] = df[essay_cols].replace(np.nan, '', regex = True)

        df['all_essays'] = df['essay0'] + df['essay1'] + df['essay5'] + df['essay6']
        df['all_essays'] = df['all_essays'].astype(str)

        df['essay_cnt'] = df.apply(count_parental_words, axis = 1)
        
    return df

def create_labels_and_features(df, type):
    if type == 'classification':
        features    = ['age', 'smokes_num', 'drugs_num', 'drinks_num', 'last_online_sec', 'income_cl', 'height']
    elif type == 'regression':
        features    = ['essay_cnt']

    label           = ['offspring']
    all_columns     = label + features
    df              = df[all_columns]

    # Drop records with at least one empty value
    df = df.dropna(subset = all_columns)

    # Create an array of labels
    df['offspring'] = df['offspring'].replace(regex=['sn&rsquo;t'], value='s not')
    labels_raw      = df['offspring'].tolist()
    labels          = [0 if 'has' not in x else 1 for x in labels_raw]
    print('Has kids: ' + str(labels.count(1)) + ', has no kids: ' + str(labels.count(0)))

    # # Create an array of features
    df = df.drop(columns=['offspring'])

    if type == 'regression':
        feat_raw      = df['essay_cnt'].tolist()
        feat          = [0 if x == 0 else 1 for x in feat_raw]
        print('Rows with parental words: '      + str(feat.count(1)) + \
              ', rows without parental words: ' + str(feat.count(0)))

    return df, features, labels


def scaling(df, features, labels):
    feature_data = df[features]

    x = feature_data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    
    feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
    feature_data = feature_data.to_numpy() # To create an array

    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test


def plot_model_validation(name, accuracy, recall, precision, xlabel, size):

    validations         = [accuracy, recall, precision]
    validation_names    = ['Accuracy', 'Recall', 'Precision']
    
    for i in range(len(validations)):
        plt.xlabel(xlabel)
        plt.ylabel(validation_names[i])
        plt.title(name + ' - ' + validation_names[i])
        plt.plot(range(1, size), validations[i])
        plt.show()
          

def create_classifier(classifier_name, size, parameter):

    if classifier_name == 'KNN':
        return KNeighborsClassifier(n_neighbors = parameter)

    elif classifier_name == 'SVM':
        return svm.SVC(kernel = 'linear', C = 0 + (1 / size) * parameter)


def create_model(approach, X_train, y_train, X_test, y_test, size):

    time_to_train = {}

    if approach == 'classification':
        models = {'KNN': 'k', 'SVM': 'C'}
    elif approach == 'regression':
        models = ['KNR', 'logistic']      

    for model in models:

        accuracy        = []
        recall          = []
        precision       = []
        training_time   = 0

        print('Model: ' + str(model))

        for i in range(1, size):
            start = time.time()

            if approach == 'classification':
                final_model = create_classifier(model, size, i)
            elif approach == 'regression':
                final_model = create_regressor(model, i)

            final_model.fit(X_train, y_train)      
            end = time.time()
            training_time += end - start

            y_pred      = final_model.predict(X_test) 

            if approach == 'classification':
                # Zero_division = 0 to avoid error when divided by zero.
                accuracy.append (accuracy_score (y_pred, y_test))
                recall.append   (recall_score   (y_pred, y_test, zero_division = 0))
                precision.append(precision_score(y_pred, y_test, zero_division = 0))
            
            elif approach == 'regression':
                print(model + ': accuracy ' + str(round(accuracy_score (y_pred, y_test), 2)) \
                            + ', recall '   + str(round(recall_score   (y_pred, y_test), 2))   \
                            + ', precision '+ str(round(precision_score(y_pred, y_test), 2)))

        if approach == 'classification':
            plot_model_validation(model, accuracy, recall, precision, models[model], size)

        time_to_train[model] = round(training_time, 1)
    print('Time_to_train: ' + str(time_to_train))    


def create_regressor(regressor_name, parameter = None):
    if regressor_name == 'KNR':
        return KNeighborsRegressor(n_neighbors = parameter)

    elif regressor_name == 'logistic':
        return LogisticRegression()


def main_script(modelling_approach):

    # Importing and exploring data
    df_original                         = read_csv()
    explore_data(df_original)

    # Making the model
    for approach in modelling_approach:
        df                                  = mapping_and_cleaning(df_original, approach)
        df, features, labels                = create_labels_and_features(df, approach)
        X_train, X_test, y_train, y_test    = scaling(df, features, labels)
        
        create_model(approach, X_train, y_train, X_test, y_test, modelling_approach[approach])


main_script({'classification': 101,
             'regression'    : 2})