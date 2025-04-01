def trade_simulation(predicted, actual, threshold, initial_cash):
    cash = initial_cash
    trades = 0
    
    # loop until n-1 index
    for i in range(len(actual) - 1):
        # Calculate predicted and actual gain
        predict_gain = ((predicted[i] - actual[i]) / actual[i]) * 100
        actual_gain = ((actual[i+1] - actual[i]) / actual[i]) * 100
        # Check if predicted gain mets the threshold
        if predict_gain >= threshold:
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
            print(f"Trade failed at {i+1}: Skipped (gain {predict_gain:.2f}% is less than the required {threshold}%)")
    
    return cash, trades

if __name__ == "__main__":
    # variables 
    predicted_prices = [105, 150, 170]
    actual_prices = [100, 120, 140]
    threshold_percent = 5  
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
    final_cash, total_trades = trade_simulation(predicted_prices, actual_prices, threshold_percent, starting_cash)
    print(f"Final cash in the basket: {final_cash:.2f}")
    print(f"Total trades made: {total_trades}")


