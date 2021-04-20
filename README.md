# Machine-Learning-Fantasy-Prediction-App

This Python program generates models to predict an NBA player's preformance depending on their preformance in previous game/seasons. Perfect for Fantasy Sports.

# Setting Up

This project uses Ananconda

'''
conda create -n nbaApp python=3.7
activate nbaApp 
'''

Install the nessesary packages

'''
pip install tensorflow sklearn numpy keras
'''

Run program
'''
//Navigate to directory with main.py
python main.py
'''


# Description: Static Model

This is used to predict how a player will preform through an entire season. The Model is trained from taking historical data (Can be changed to range up to the 1950s
depending on CPU preformance) of all NBA players. Contains an option to filter the training data by model if user believes that it would lead to a more accurate result
(Positionless basketball is the currently the META so I decided to leave this optional). This Model currently predicts player stats for the 2020-2021 season.

# DDescription: Dynamic Model

This is used to predict how a player will preform for their next game. The model is trained from taking data from the static model along with game stats of a certain player
for the current season. When indicated to enter a URL, enter the per game stats of given player from ESPN.com.

