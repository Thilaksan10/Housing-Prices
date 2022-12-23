from data import prepare_data
from model import build_model
import numpy as np
import csv

if __name__ == '__main__':
    # build model and load weigths of pretained model
    model = build_model()
    model.load_weights('./checkpoints/my_checkpoint5')

    # prepare test data for prediction
    _, _, _, _, X_test = prepare_data(['train.csv', 'test.csv'], test_split=0.2)

    # open submission.csv and write into it
    with open('submission.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        # write first line
        writer.writerow(['Id', 'SalePrice'])

        # write Id and predited Sale price of each House
        for i, data in enumerate(X_test):
            y_pred = model.predict([np.expand_dims(data, 0)])[0][0]
            print(f'Id: {i+1461}, SalePrice: {y_pred}')
            writer.writerow([i+1461, y_pred])
