# importing the csv files 
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# trade from 449 after
def trade_simulation(predicted, actual, rf_preds, threshold, initial_cash):
    cash = initial_cash
    trades = 0
    
    # loop until n-1 index
    for i in range(len(actual) - 1):
        # Calculate predicted and actual gain
        predict_gain = ((predicted[i] - actual[i]) / actual[i]) * 100
        actual_gain = ((actual[i+1] - actual[i]) / actual[i]) * 100
        rf_vote = rf_preds[i]
        #predict_gain_dec
        
        #preduct_gain_lstm

        # Check if predicted gain mets the threshold
        if predict_gain >= threshold and rf_vote == 1:
            if cash >= actual[i]: # if cash is enough
                # subtract its actual price. mimic buy stock
                cash -= actual[i]
                # sell the n+1 actual price. mimic sell stock
                cash += actual[i + 1]
                trades += 1
                print(f"Trade Success at {i+1}: Bought at {actual[i]:.2f} and sold at {actual[i+1]:.2f}, calculated gain {predict_gain:.2f}%, actual gains {actual_gain:.2f}%")
            else: # if you too poor :)
                print(f"Trade failed at {i+1}: Not enough cash to buy priced at {actual[i]:.2f} womp womp")
        else:
            #print(f"Trade failed at {i+1}: Skipped (gain {predict_gain:.2f}% is less than the required {threshold}%)")
            h=8
    
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
    df_amazon['change_value']   = df_amazon['Open'].diff().fillna(0)
    df_amazon['marginal_change'] = (df_amazon['change_value'] > 0).astype(int)
    # run the models
    degree = 5
    regression_data = regression_model(df_amazon, degree)
    # variables 
    # regression variable 

    predicted_prices = regression_data['Predicted_Open'].tolist()
    actual_prices = regression_data['Actual_Open'].tolist()



    # decision tree variable   
    rf_results = random_forest_model(df_amazon)
    rf_preds = rf_results['predicted'].tolist()
    # lstm


    # init value
    threshold_percent = 2
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
    
    final_cash, total_trades = trade_simulation(predicted_prices, actual_prices, rf_preds,threshold_percent, starting_cash)
    print(f"Final cash in the basket: {final_cash:.2f}")
    print(f"Earned Amount: {final_cash-starting_cash:.2f}")
    print(f"Total trades made: {total_trades}")
    

