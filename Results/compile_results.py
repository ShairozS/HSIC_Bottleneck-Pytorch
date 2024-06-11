import pandas as pd
import numpy as np
import os

RESULTS_FOLDER = './Resuls'
result_files = [os.path.join(RESULTS_FOLDER, x) for x in RESULTS_FOLDER if x.endswith('.csv')]
result_rows = []
for r in result_files:
    
    filename = os.path.split(r)[-1]
    print("Processing: ", filename)

    # Extract relevent information from filename
    params = filename.split("_")
    dataset = params[0]
    backprop = params[1]
    model = params[2]
    num_layers = params[3]
    batchsize = params[5]
    lr = params[7]
    epochs = params[9]
    num_parameters = params[11]
    optimizer = params[13]

    # Get best test performance and when it was reached
    data = pd.read_csv(r)
    best_perf_test =  np.argmax(data['Test_loss'])
    best_perf_train = np.argmax(data['Train_loss'])
    best_perf_epoch_test = np.where(data['Test_loss'] == best_perf_test)


    # Get the average time taken per epoch
    avg_time = np.mean(data['Time'])

    result_rows.append(dataset, backprop, model, num_layers, batchsize, lr, epochs, num_parameters, 
                       optimizer, best_perf_train, best_perf_test, best_perf_epoch_test, avg_time)


df = pd.DataFrame(result_rows)
df.columns = ["Dataset", "Backprop", "Model", "Num_layers", "Batchsize", "LearningRate", "TotalEpochs", "Num_parameters",
              "Optimizer", "Best_train", "Best_test", "Best_test_at", "Avg_time"]

print(df.head())

    