import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pylab import bone, colorbar, plot, show, pcolor 

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
