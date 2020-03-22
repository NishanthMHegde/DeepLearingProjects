import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pylab import bone, colorbar, plot, show, pcolor 
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom 

"""
What is a self organizing map?
SOM is a grid of nodes where each of the the node is color coded and represents one observation or category in our dataset.
By making use of colors, it categorizes each of the nodes. For example, we could draw an SOM of the countries in the world.
Here each country can have different attributes (independent variables) like GDP, per capita income, education, health, hunger index, etc.
Depending on each of these atributes, a country is assigned a color which can depict if it is a well off country or a country in poor conditions.
Each color here belongs to one winning node (node with the least mean interneuron distance) for an attribute.

Steps involved in drawing a basic SOM.
1. Consider 3 independent variables or attributes which are present for each observation. These are called visible input nodes.
2. Consider n neurons present in the first layer veritically. These are called Visible Output Nodes.
3. The Euclidean distance from the node to each of the visible input node is calculated. This is done for every Visible Output node.
4. The node which has the least Euclidea distance is called BMU (Best matching Unit).
5. We repeat this for all columns of nodes to find BMUs. 
6. Everytime a BMU is updated, the weight of all the nodes within a specific distance around the BMU are also updated.
7. If there are multiple color BMUs around a region, then the nodes around them get pulled towards the closest BMU and obtains the shade of the most nearest BMU.

Features of SOMs:
1. They classify data without any supervision (there is no need to have a training data which has classification like 0/1 or Yes/No)
2. It brings out different types of correlation between attributes.
3. It retains the topology of the dataset.
4. The output nodes are not connected to each other as in ANN or CNN. 
5. There is no target vector and hence there is no backpropogation of weights.

"""

"""
BUSINESS DESCRIPTON
Our dataset consists of a list of customers of a reputed bank. Each customer has a list of attributes associated with the account he owns.
The last column represents whether the customer got advanced credit or not.
The aim of this exercise is to ascertain which customers managed to get an advanced credit without actually desrving to get a credit or fraudulent
customers who have tricked the bank in obtaining advanced credit.

"""

df = pd.read_csv('Credit_Card_Applications.csv')
print(df.head())
X = df.iloc[:, 0:-1].values
y = df.iloc[:,-1].values

#Let us feature scale the data to have all data in range (0,1)
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Let us now create the SOM object
#input_len is the number of indpendent variables or attributes in our dataset X
#x and y hold the num of rows and columns in the SOM grid
print("Let us now create the SOM object")
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5, random_seed=0)
#let us now initialize random weights which are initially small and in range (0,1)
print("let us now initialize random weights which are initially small and in range (0,1)")
som.random_weights_init(X)
#Let us now train our data
print("Let us now train our data")
som.train_random(data=X, num_iteration=100)

#Let us now plot our SOM using pylab
print("Let us now plot our SOM using pylab")
bone() #Plots the white background where we can place our graph
pcolor(som.distance_map().T) #distance map plots the SOM which has colorwise representation of the Mean Interneuron Distances (MID) between that node to its visible input nodes
colorbar() #The color bar plots a scale which signifies what each color level means. In our distance map, white means highest MID and black means lowest MID
#we can assume that the nodes with highest MID have the customers who do not have a good track record with the bank and are most likey to perform fraudulent activities.
#Let us now verify how many customers fall under the "Got Credit" and "Did not Get Credit" categories and to which MID on the distance map they belong to.
print("we can assume that the nodes with highest MID have the customers who do not have a good track record with the bank and are most likey to perform fraudulent activities.")
print("Let us now verify how many customers fall under the \"Got Credit\" and \"Did not Get Credit\" categories and to which MID on the distance map they belong to.")
for i, x in enumerate(X):
	#i holds the customer id
	#x holds independent variable
	#red circle means did not get credit
	#green square means they got the credit
	markers = ['o', 's']
	colors = ['r', 'g']
	w = som.winner(x) #for each attribute in an observation, get the winner neuron x-y coordinates
	#plot the winning neuron for each observation at the center (0.5 units to the right and 0.5 units top so that it lies in center)
	plot(w[0]+0.5, w[1]+0.5, markers[y[i]],
		markerfacecolor='None',
		markeredgecolor=colors[y[i]],
		markersize=10,
		markeredgewidth=2)
show()

#Finding the frauds
print("Finding the frauds")

#In the above SOM, we can see that the customers in the white nodes (MID) got apporval as indicated by green square. This is fraudulent. 
#The customers in outlier nodes (MID ~ 1.0) should NEVER get approval. So let us catch the culrpits.

print("In the above SOM, we can see that the customers in the white nodes (MID) got apporval as indicated by green square. This is fraudulent.")
print("The customers in outlier nodes (MID ~ 1.0) should NEVER get approval. So let us catch the culrpits.")

#win_map retruns a dictionary where the key is the co-ordinates of the winning nodes and the value is a list of the customers belonging to that winning node.
print("win_map retruns a dictionary where the key is the co-ordinates of the winning nodes and the value is a list of the customers belonging to that winning node.")
mapping = som.win_map(X)
#There are 3 outlier nodes (MID=1.0) but 2 of those nodes have customers who got approval. Let us find those customers belonging to those 2 winning nodes.
print("There are 3 outlier nodes (MID=1.0) but 2 of those nodes have customers who got approval. Let us find those customers belonging to those 2 winning nodes.")
frauds = np.concatenate((mapping[(2,3)], mapping[(2, 5)]), axis=0) #axis=0 means we need to concatenate along the rows so that customers list are placed one beneath the other

#We need to now inverse transform our feature scaled data to get the actual data
frauds = sc.inverse_transform(frauds)
print("Our fradulent customers who get Credit Approval despite having bad stats are")
print(frauds)
