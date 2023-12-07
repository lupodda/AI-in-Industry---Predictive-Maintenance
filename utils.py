from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

# Splitting
def split_data(df, comp=None):

  if comp is not None:
    df = df[df[f'rul_comp{comp}'] != -1].reset_index(drop = True)

  gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
  train_ids, test_ids = next(gss.split(X = df, groups = df['machineID']))
  print(train_ids.shape, test_ids.shape)
  df_train = df.loc[train_ids].reset_index(drop=True)
  df_test = df.loc[test_ids].reset_index(drop=True)

  return df_train, df_test

def generate_k_folds(df, k, groups, comp=None):

  if comp is not None:
    df = df[df[f'rul_comp{comp}'] != -1].reset_index(drop = True)

  skf = StratifiedGroupKFold(n_splits=k)
  output = skf.split(df, df['binary_failure'], groups=df[groups])

  folds = np.zeros(df.shape[0])

  for i, ids in enumerate(output):
    folds[ids[1]] = i # 0 are training ids, 1 are test ids

  df['fold'] = folds.astype(np.int64)

  return df

def split_data_fold(df, fold):

  df_train = df[df['fold'] != fold].reset_index(drop=True)
  df_test = df[df['fold'] == fold].reset_index(drop=True)

  return df_train, df_test

def standardise_data_train_test(df_train, df_test, comp=None):

    columns = ['volt', 'rotate', 'pressure', 'vibration', 'age']
    trmean = df_train[columns].mean()
    trstd = df_train[columns].std().replace(to_replace=0, value=1) # handle static fields

    df_train_st = df_train.copy()
    df_train_st[columns] = (df_train[columns] - trmean) / trstd

    df_test_st = df_test.copy()
    df_test_st[columns] = (df_test[columns] - trmean) / trstd

    if comp is not None:

      trmaxrul = df_train[f'rul_comp{comp}'].max()

      df_train_st[f'rul_comp{comp}'] = df_train[f'rul_comp{comp}'] / trmaxrul
      df_train_st[f'rul_comp{comp}'] = np.where(df_train_st[f'rul_comp{comp}'] < 0, -1, df_train_st[f'rul_comp{comp}'])

      df_test_st[f'rul_comp{comp}'] = df_test[f'rul_comp{comp}'] / trmaxrul
      df_test_st[f'rul_comp{comp}'] = np.where(df_test_st[f'rul_comp{comp}'] < 0, -1, df_test_st[f'rul_comp{comp}'])

    return df_train_st, df_test_st




def aggregation(df, comp, hour):

    df = df.drop(['rm_comp1', 'rm_comp2', 'rm_comp3', 'rm_comp4',
                          'rul_comp1', 'max_rul_comp1', 'rul_comp2', 'max_rul_comp2',
                          'rul_comp3', 'max_rul_comp3', 'rul_comp4', 'max_rul_comp4'], axis = 1)

    df = df.reset_index(drop=True).groupby('machineID').resample(f'{hour}H', on='datetime').agg({'volt':'mean',
                                                                                    'rotate':'mean',
                                                                                    'pressure':'mean',
                                                                                    'vibration':'mean',
                                                                                    'age':'first',
                                                                                    'model':'first',
                                                                                    'binary_failure':'sum',
                                                                                    f'fail_comp{comp}':'sum',
                                                                                    'error1': 'sum',
                                                                                    'error2': 'sum',
                                                                                    'error3': 'sum',
                                                                                    'error4': 'sum',
                                                                                    'error5': 'sum'}).reset_index()
    return df



def generate_label_classification(df, w):

    df['fail_within_w_comp1'] = np.where((df['rul_comp1'] <= w) & (df['rul_comp1'] > -1), 1, 0 )
    df['fail_within_w_comp2'] = np.where((df['rul_comp2'] <= w) & (df['rul_comp2'] > -1), 1, 0 )
    df['fail_within_w_comp3'] = np.where((df['rul_comp3'] <= w) & (df['rul_comp3'] > -1), 1, 0 )
    df['fail_within_w_comp4'] = np.where((df['rul_comp4'] <= w) & (df['rul_comp4'] > -1), 1, 0 )

    df['fail_within_w'] = df[['fail_within_w_comp1','fail_within_w_comp2','fail_within_w_comp3','fail_within_w_comp4']].values.tolist()

    return df


def createXY(dataset,n_past,comp,columns):
    X = []
    Y = []
    for i in range(n_past, len(dataset)):
            X.append(dataset.loc[i - n_past:i][columns])
            Y.append(dataset.loc[i][f'rul_comp{comp}'])
    return np.array(X), np.array(Y)

def data_per_machine(df, x, n, comp, columns):
    data_machine_x = df[df["machineID"] == x]
    return createXY(data_machine_x.reset_index(drop=True), n-1, comp, columns)

def createXY_classification(dataset,n_past,columns):
    X = []
    Y = []
    for i in range(n_past, len(dataset)):
            X.append(dataset.loc[i - n_past:i][columns])
            Y.append(dataset.loc[i][f'fail_within_w'])
    return np.array(X),np.array(Y)

def data_per_machine_classification(df, x, n, columns):
    data_machine_x = df[df["machineID"] == x]
    return createXY_classification(data_machine_x.reset_index(drop=True), n-1, columns)


def create_data_sequence(time_steps, df, comp, columns):

    # Create window of train data, with dimension = time_steps
    y = []
    x = []

    unique_machine_ids = df['machineID'].unique()

    # Using list comprehensions for improved performance
    data = [(data_per_machine(df, i, time_steps, comp, columns)) for i in unique_machine_ids]

    # Extracting trainX and trainY arrays
    x = [X for X, _ in data]
    y = [Y for _, Y in data]

    # Concatenating arrays using numpy
    x = np.concatenate(x, dtype=np.float32)
    y = np.concatenate(y, dtype=np.float32)

    print('Data Shape: ', x.shape)
    print('Label Shape: ', y.shape)

    return x, y

def create_data_sequence_classification(time_steps, df, comp, columns):

    # Create window of train data, with dimension = time_steps
    y = []
    x = []

    unique_machine_ids = df['machineID'].unique()

    # Using list comprehensions for improved performance
    data = [(data_per_machine_classification(df, i, time_steps, columns)) for i in unique_machine_ids]

    # Extracting trainX and trainY arrays
    x = [X for X, _ in data]
    y = [Y for _, Y in data]

    # Concatenating arrays using numpy
    x = np.concatenate(x, dtype=np.float32)
    y = np.concatenate(y, dtype=np.float32)

    print('Data Shape: ', x.shape)
    print('Label Shape: ', y.shape)

    return x, y

def standardise_rul_train_test(df_train, df_test, comp):

    df_train_st = df_train.copy()
    df_test_st = df_test.copy()

    trmaxrul = df_train[f'rul_comp{comp}'].max()

    df_train_st[f'rul_comp{comp}'] = df_train[f'rul_comp{comp}'] / trmaxrul
    df_train_st[f'rul_comp{comp}'] = np.where(df_train_st[f'rul_comp{comp}'] < 0, -1, df_train_st[f'rul_comp{comp}'])

    df_test_st[f'rul_comp{comp}'] = df_test[f'rul_comp{comp}'] / trmaxrul
    df_test_st[f'rul_comp{comp}'] = np.where(df_test_st[f'rul_comp{comp}'] < 0, -1, df_test_st[f'rul_comp{comp}'])

    return df_train_st, df_test_st


def get_loss_plot(hist):
    df = pd.DataFrame(hist.history)
    plt.figure(figsize=(8, 8))
    plt.plot(list(range(df.shape[0])), df['loss'], marker='o', color='green', label='training')
    plt.plot(list(range(df.shape[0])), df['val_loss'], marker='^', color='purple', label='validation')
    plt.legend()
    plt.title("LOSS COMPARISON")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.savefig("loss.png")
    plt.show()

def plot_rul(pred=None, target=None, #feature=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target', color='tab:orange')
        #plt.plot(range(len(feature)), feature, label='feature', color='tab:green')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.grid(linestyle=':')
    plt.tight_layout()

def get_best_threshold(probs,y_test):
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_tr = thresholds[0]
    best_f1 = 0.0
    for tr in thresholds:
        tr = round(tr, 2)
        pred = np.where(probs >= tr, 1, 0)
        y_true = np.array(y_test)
        f1_macro = f1_score(y_true, pred,average= 'weighted')
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_tr = tr

    print('BEST THRESHOLD', best_tr)
    print('F1 SCORE MACRO:', best_f1)
    return best_tr, best_f1

def flatted_labels(y_test, tr_pred_test_as_th):
  
  true_labels = y_test.copy()
  for i in range(0, y_test.shape[0]):
    for j in range(0, y_test.shape[1]):
      if y_test[i,j] != 0:
        true_labels[i,j] = j+1
      else:
        true_labels[i,j] = 0

  predicted_labels = tr_pred_test_as_th.copy()
  for i in range(0, tr_pred_test_as_th.shape[0]):
    for j in range(0, tr_pred_test_as_th.shape[1]):
      if tr_pred_test_as_th[i,j] != 0:
        predicted_labels[i,j] = j + 1
      else:
        predicted_labels[i,j] = 0

  true_labels = true_labels.flatten()
  predicted_labels = predicted_labels.flatten()

  return true_labels, predicted_labels

# create a RUL colum for each component

def generate_rul_classification(df, limit_rul):

  comps = [1,2,3,4]
  df_tot = pd.DataFrame(columns=df.columns)
  machine_ids = df['machineID'].unique()

  for comp in comps:

    rul = np.array([-1]*df.shape[0])
    max_rul = np.array([-1]*df.shape[0])

    for idx in machine_ids:

      df_machine = df[df['machineID']==idx].reset_index(drop=True)
      ids_fail = df_machine[df_machine[f'fail_comp{comp}']==1].index.values
      pos_machine = df[df['machineID']==idx].index.values

      if len(ids_fail) != 0:

        tot_life = [ids_fail[0]]

        for i in range(1, len(ids_fail)):
          tot_life.append(ids_fail[i] - ids_fail[i-1] - 1) # -1 otherwise they overlap

        k = 0
        for i in tot_life:
          for j in range(0,i+1):
            rul[pos_machine[k]] = (i - j) # 96-0, 96-1, ..., 96-96_rul = []
            max_rul[pos_machine[k]] = i
            k+=1

        df[f'rul_comp{comp}'] = np.where(rul <= limit_rul, rul, limit_rul)
        df[f'max_rul_comp{comp}'] = max_rul

  # drop rows without RUL information (after that time no comp failed)

  df.drop((df[(df.rul_comp1 == -1) & (df.rul_comp2 == -1) &
              (df.rul_comp3 == -1) & (df.rul_comp4 == -1)].index), inplace=True)
  df = df.reset_index(drop=True)

  return df


def aggregation_classification(df, hour):

    df = df.drop(['rm_comp1', 'rm_comp2', 'rm_comp3', 'rm_comp4',
                          'rul_comp1', 'max_rul_comp1', 'rul_comp2', 'max_rul_comp2',
                          'rul_comp3', 'max_rul_comp3', 'rul_comp4', 'max_rul_comp4'], axis = 1)

    df = df.reset_index(drop=True).groupby('machineID').resample(f'{hour}H', on='datetime').agg({'volt':'mean',
                                                                                'rotate':'mean',
                                                                                'pressure':'mean',
                                                                                'vibration':'mean',
                                                                                'age':'first',
                                                                                'model':'first',
                                                                                'binary_failure':'sum',
                                                                                'fail_comp1':'sum',
                                                                                'fail_comp2':'sum',
                                                                                'fail_comp3':'sum',
                                                                                'fail_comp4':'sum',
                                                                                'error1': 'sum',
                                                                                'error2': 'sum',
                                                                                'error3': 'sum',
                                                                                'error4': 'sum',
                                                                                'error5': 'sum'}).reset_index()
    return df