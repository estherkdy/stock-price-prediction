# importing the csv files 
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor


# trade from 449 after
def trade_simulation(predicted, lstm_test, actual, rf_preds, threshold, initial_cash):
    # predicted from 0 - n-1
    cash = initial_cash
    trades = 0
    success = 0;
    fail = 0
    start_index = len(actual) - len(lstm_test) # index 712
    #start_index = 448
    # loop until n-1 index
    for i in range(len(lstm_test)-1):
        
        # Calculate predicted and actual gain
        #print(f"Predicted price:{predicted[i]}   Actual price:{actual[i+start_index]} ")#predicted 1 -178

        # actual 1 - 899 -> 
        predict_gain = ((predicted[i+1] - actual[i+start_index]) / actual[i+start_index]) * 100
        predict_gain_2 = ((lstm_test[i+1] - actual[i+start_index]) / actual[i+start_index]) * 100
        actual_gain = ((actual[i+1+start_index] - actual[i+start_index]) / actual[i+start_index]) * 100
        #rf_vote = rf_preds[i]
        #predict_gain_dec
        a = 1000 // actual[i+start_index] 

        #preduct_gain_lstm
        # Check if predicted gain mets the threshold
        if predict_gain >= threshold and rf_preds[i] == 1 and predict_gain_2 >= threshold:
        #if (predict_gain >= threshold and predict_gain_2 >= threshold):
            if cash >= actual[i + start_index]: # if cash is enough
                if (actual_gain > 0):
                    success+=1;
                else:
                    fail += 1;
                # subtract its actual price. mimic buy stock
                cash -= a*actual[i+start_index]
                # sell the n+1 actual price. mimic sell stock
                cash += a*actual[i + 1 + start_index]
                trades += 1
                print(f"Trade Success at {i+1}: Bought at {actual[i+start_index]:.2f} and sold at {actual[i+1+start_index]:.2f} Predicted at: {predicted[i+1]}, actual gains {actual_gain:.2f}%")
            else: # if you too poor :)
                print(f"Trade failed at {i+1}: Not enough cash to buy priced at {actual[i]:.2f} womp womp")
        else:
            #print(f"Trade failed at {i+1}: Skipped (gain {predict_gain:.2f}% is less than the required {threshold}%)")
            h=0
            
    print(f"success trade: {success}")
    print(f"fail trade: {fail}")
    return cash, trades

def regression_model(df_amazon, d):
    start_size = (len(df_amazon)) // 2

    predictions = []    
    actuals = [] 

    for i in range(start_size, len(df_amazon)):
        train_data = df_amazon.iloc[i - start_size:i]
        X_train = np.arange(start_size).reshape(-1, 1) 
        y_train = train_data['Open'].to_numpy()  
    
        # Use a degree 3 polynomial regression model.
        degree = d  
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
    
        # Predict the next value (the next apple's open price).
        next_index = np.array([[start_size]])  # the next index after the training window
        pred = model.predict(next_index)
    
        predictions.append(pred[0])
        actuals.append(df_amazon.iloc[i]['Open'])
    
    results = pd.DataFrame({
        'Actual_Open': actuals,
        'Predicted_Open': predictions
    }, index=df_amazon.index[start_size:])
    

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R² Score:", r2)
    print(results)
    return results  # Return the DataFrame with actual and predicted values.

def random_forest_model(df_amazon):
    split_index = len(df_amazon) // 2

    train_df = df_amazon.iloc[:split_index]
    test_df = df_amazon.iloc[split_index:].copy()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'change_value']
    target = 'marginal_change'
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    
    model = RandomForestClassifier(n_estimators=1000, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    test_df['predicted'] = predictions
    # Column 1 is the probability that the price will increase.
    test_df['pct_increase'] = probabilities[:, 1] * 100
    test_df['pct_decrease'] = probabilities[:, 0] * 100
    print(test_df.head())
    return test_df

if __name__ == "__main__":
    # read csv
    df_amazon = pd.read_csv('amazon_data.csv')

    df_amazon = df_amazon.iloc[::-1]
    df_amazon = df_amazon.reset_index(drop=True)

    df_amazon['change_value']   = df_amazon['Open'].diff().fillna(0)
    df_amazon['marginal_change'] = (df_amazon['change_value'] > 0).astype(int)

    # run the models
    degree = 6
    regression_data = regression_model(df_amazon, degree)
    # variables 
    # regression variable 

    tester = [234.24, 233.16, 232.56, 232.07, 232.03, 232.17, 232.08, 232.06, 231.98, 232.27, 232.04, 232.19, 231.97, 231.61, 231.39, 231.3, 231.45, 231.17, 230.86, 230.65, 230.36, 230.25, 230.11, 230.51, 230.11, 229.86, 229.97, 229.69, 229.38, 229.1, 229.61, 229.56, 229.64, 229.32, 229.05, 228.92, 228.6, 228.85, 228.98, 229.12, 228.91, 228.08, 227.56, 226.8, 226.63, 226.38, 226.33, 225.91, 225.72, 225.45, 225.29, 225.63, 225.58, 225.42, 225.12, 224.68, 224.21, 223.8, 223.37, 223.39, 223.51, 223.32, 222.86, 222.63, 221.94, 221.08, 220.54, 220.19, 219.32, 218.34, 217.62, 217.1, 216.7, 216.76, 215.87, 215.56, 215.75, 215.01, 214.2, 213.71, 213.4, 213.87, 214.61, 215.92, 216.67, 217.09, 217.11, 216.76, 216.55, 217.45, 216.86, 216.77, 216.38, 215.95, 216.06, 216.02, 214.6, 214.0, 213.49, 213.35, 212.79, 212.22, 211.52, 212.05, 213.01, 212.95, 212.76, 212.19, 212.01, 211.47, 211.36, 210.17, 208.73, 207.87, 206.72, 206.51, 206.08, 205.96, 206.67, 206.64, 206.88, 207.48, 207.61, 208.29, 208.73, 209.48, 209.33, 208.93, 208.57, 208.34, 207.6, 207.08, 206.64, 206.24, 205.17, 204.56, 204.42, 204.02, 203.09, 202.15, 202.12, 202.31, 201.81, 201.95, 200.84, 200.03, 199.28, 198.24, 198.39, 198.31, 199.26, 199.04, 199.11, 199.1, 198.96, 199.88, 199.16, 200.35, 200.8, 200.67, 200.87, 200.93, 200.75, 201.14, 201.11, 200.54, 200.23, 199.88, 199.03, 198.1, 197.86, 196.93, 197.51, 197.74, 198.1, 198.41, 198.67, 198.62]
    print(len(tester))
    
    

    predicted_prices = regression_data['Predicted_Open'].tolist()
    print("Length of predicted ")
    #print(len(predicted_prices))
    #print(predicted_prices[0])
    predicted_prices = predicted_prices[-len(tester):]
    actual_prices = regression_data['Actual_Open'].tolist()
    actual_price = df_amazon['Open'].tolist()

    # decision tree variable   
    rf_results = random_forest_model(df_amazon)
    rf_preds = rf_results['predicted'].tolist()
    #rf_preds = 0
    # lstm


    # init value
    threshold_percent = 0.1
    starting_cash = 1000

    # I can do a variable n-amount of stock buys based on the formula on kelly criterion
    # which requires decision tree to give information on what percentage picks bull market over bear
    # and picked side how much percentage on average ex. if 55% choose bull with 5% predicted gain 
    # and 45 bet lose  for 4% 

    # k% = (bp-q)b would be (.05*0.55-0.45)/0.05
    # i was wrong 
    # k = (bp-ql)/b would be (.05*0.55-0.45*0.04)/0.05 = 4.75  anything above a 1 means bet is favorable in one side
    # if negative then its not worth betting on

    # to-do with decision tree - make a 2d array [n][4] where it holds  n being the index
    # 1. percentage of population bet gain
    # 2. average of those percentage gains 
    # 3. percentage of population bet loss
    # 4. average of those percentage loss 

    # decision tree will only train on first half
    
    #test case
    #final_cash, total_trades = trade_simulation(predicted_prices, predicted_prices , actual_prices, rf_preds,threshold_percent, starting_cash)
    

    #final_cash, total_trades = trade_simulation(predicted_prices, predicted_prices , actual_prices, rf_preds,threshold_percent, starting_cash)
    final_cash, total_trades = trade_simulation(predicted_prices, tester, actual_price, rf_preds,threshold_percent, starting_cash)
    print(f"Final cash in the basket: {final_cash:.2f}")
    print(f"Earned Amount: {final_cash-starting_cash:.2f}")
    print(f"Total trades made: {total_trades}")
    
